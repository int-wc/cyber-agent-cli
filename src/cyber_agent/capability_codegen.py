"""动态 capability 代码生成、校验与落盘。

从 CapabilityRegistry 中提取的纯代码生成逻辑，不依赖注册表实例。
"""
from __future__ import annotations

import ast
import py_compile
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CAPABILITY_ENTRYPOINT_FILENAME = "capability.py"
CAPABILITY_TOOL_LAUNCHER_CMD = "run_tool.cmd"
CAPABILITY_TOOL_LAUNCHER_SH = "run_tool.sh"
CAPABILITY_SKILL_LAUNCHER_CMD = "render_skill.cmd"
CAPABILITY_SKILL_LAUNCHER_SH = "render_skill.sh"
CAPABILITY_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{2,63}$")


@dataclass(slots=True)
class CapabilityArtifacts:
    """描述已落盘的代码产物位置与校验结果。"""

    artifact_dir: Path
    entrypoint_path: Path
    source_code: str
    tool_launcher_path: Path | None = None
    skill_launcher_path: Path | None = None
    validation_issues: list[str] = field(default_factory=list)


def _strip_markdown_code_fence(raw_text: str) -> str:
    """兼容模型偶尔返回 ```json ... ``` 或 ```python ... ``` 的情况。"""
    stripped_text = raw_text.strip()
    if not stripped_text.startswith("```"):
        return stripped_text
    lines = stripped_text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return stripped_text


def default_tool_python_code() -> str:
    """返回工具逻辑缺失时的最小可运行骨架。"""
    return textwrap.dedent(
        """
        def handle_request(request: str, context: str) -> str:
            \"\"\"默认工具骨架：在真实逻辑补全前先返回结构化结果。\"\"\"
            cleaned_request = request.strip()
            cleaned_context = context.strip()
            lines = [
                "当前生成工具尚未补全专用实现，先返回结构化骨架。",
                f"请求: {cleaned_request or '无'}",
            ]
            if cleaned_context:
                lines.append(f"上下文: {cleaned_context}")
            lines.append("TODO(人工实现): 在此补充 handle_request 的核心逻辑。")
            return "\\n".join(lines)
        """
    ).strip()


def default_skill_python_code() -> str:
    """返回技能提示词缺失时的最小可运行骨架。"""
    return textwrap.dedent(
        """
        def build_skill_prompt() -> str:
            \"\"\"默认 skill 骨架：提示后续开发者继续补全。\"\"\"
            return (
                "当前生成 skill 仍是最小骨架。\\n"
                "TODO(人工实现): 根据真实业务需求补充 build_skill_prompt 的内容。"
            )
        """
    ).strip()


def build_capability_source(
    *,
    name: str,
    kind: str,
    description: str,
    register_as_tool: bool,
    tool_python_code: str,
    skill_python_code: str,
) -> str:
    """将模型返回的代码片段包装成可直接执行的 Python 文件。"""
    normalized_tool_code = (
        textwrap.dedent(_strip_markdown_code_fence(tool_python_code)).strip()
        or default_tool_python_code()
    )
    normalized_skill_code = (
        textwrap.dedent(_strip_markdown_code_fence(skill_python_code)).strip()
        or default_skill_python_code()
    )
    source_parts = [
        '"""自动生成的 capability 代码文件。"""',
        "",
        "from __future__ import annotations",
        "",
        "import argparse",
        "import sys",
        "from typing import Final",
        "",
        f"CAPABILITY_NAME: Final[str] = {name!r}",
        f"CAPABILITY_KIND: Final[str] = {kind!r}",
        f"CAPABILITY_DESCRIPTION: Final[str] = {description!r}",
        f"CAPABILITY_REGISTER_AS_TOOL: Final[bool] = {register_as_tool!r}",
        "",
        normalized_tool_code,
        "",
        normalized_skill_code,
        "",
        "def _main() -> int:",
        "    parser = argparse.ArgumentParser(",
        '        description=f"动态 capability 运行入口: {CAPABILITY_NAME}"',
        "    )",
        '    parser.add_argument("mode", choices=["run", "prompt"])',
        '    parser.add_argument("--request", default="")',
        '    parser.add_argument("--context", default="")',
        "    args = parser.parse_args()",
        "",
        "    try:",
        '        if args.mode == "run":',
        "            print(handle_request(args.request, args.context))",
        "            return 0",
        "        print(build_skill_prompt())",
        "        return 0",
        "    except Exception as exc:  # pragma: no cover - 生成代码运行时兜底",
        '        print(f"执行失败: {exc}", file=sys.stderr)',
        "        return 1",
        "",
        "",
        'if __name__ == "__main__":',
        "    raise SystemExit(_main())",
    ]
    return "\n".join(source_parts).rstrip() + "\n"


def validate_capability_source(
    source_code: str,
    *,
    requires_tool: bool,
    requires_skill: bool,
) -> list[str]:
    """校验生成代码至少满足基础语法和约定函数存在。"""
    issues: list[str] = []
    try:
        module = ast.parse(source_code)
    except SyntaxError as exc:
        return [f"生成代码存在语法错误：{exc.msg}，第 {exc.lineno} 行。"]

    function_names = {
        node.name
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    if requires_tool and "handle_request" not in function_names:
        issues.append("生成代码未定义 handle_request(request, context)。")
    if requires_skill and "build_skill_prompt" not in function_names:
        issues.append("生成代码未定义 build_skill_prompt()。")
    return issues


def write_launcher_files(
    artifact_dir: Path,
    entrypoint_path: Path,
    *,
    needs_tool_launcher: bool,
    needs_skill_launcher: bool,
) -> tuple[Path | None, Path | None]:
    """为生成的 capability 写入便于人工直接运行的启动脚本。"""
    tool_launcher_path: Path | None = None
    skill_launcher_path: Path | None = None
    python_executable = str(Path(sys.executable).resolve())

    if needs_tool_launcher:
        tool_launcher_path = artifact_dir / CAPABILITY_TOOL_LAUNCHER_CMD
        tool_launcher_path.write_text(
            f'@echo off\r\n"{python_executable}" "{entrypoint_path}" run %*\r\n',
            encoding="utf-8",
        )
        tool_launcher_sh = artifact_dir / CAPABILITY_TOOL_LAUNCHER_SH
        tool_launcher_sh.write_text(
            f'#!/usr/bin/env sh\nexec "{python_executable}" "{entrypoint_path}" run "$@"\n',
            encoding="utf-8",
        )
        tool_launcher_sh.chmod(0o755)

    if needs_skill_launcher:
        skill_launcher_path = artifact_dir / CAPABILITY_SKILL_LAUNCHER_CMD
        skill_launcher_path.write_text(
            f'@echo off\r\n"{python_executable}" "{entrypoint_path}" prompt %*\r\n',
            encoding="utf-8",
        )
        skill_launcher_sh = artifact_dir / CAPABILITY_SKILL_LAUNCHER_SH
        skill_launcher_sh.write_text(
            f'#!/usr/bin/env sh\nexec "{python_executable}" "{entrypoint_path}" prompt "$@"\n',
            encoding="utf-8",
        )
        skill_launcher_sh.chmod(0o755)

    return tool_launcher_path, skill_launcher_path


def materialize_capability_artifacts(
    *,
    storage_dir: Path,
    name: str,
    kind: str,
    description: str,
    register_as_tool: bool,
    generated_spec: dict[str, Any],
) -> CapabilityArtifacts:
    """将 capability 真实落盘为 Python 代码文件与启动脚本。"""
    artifact_dir = storage_dir / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source_code = build_capability_source(
        name=name,
        kind=kind,
        description=description,
        register_as_tool=register_as_tool,
        tool_python_code=str(generated_spec.get("tool_python_code", "")),
        skill_python_code=str(generated_spec.get("skill_python_code", "")),
    )
    entrypoint_path = artifact_dir / CAPABILITY_ENTRYPOINT_FILENAME
    entrypoint_path.write_text(source_code, encoding="utf-8")

    validation_issues = validate_capability_source(
        source_code,
        requires_tool=(kind == "tool" or register_as_tool),
        requires_skill=(kind == "skill"),
    )
    try:
        py_compile.compile(str(entrypoint_path), doraise=True)
    except py_compile.PyCompileError as exc:
        validation_issues.append(f"生成代码编译失败：{exc.msg}")

    tool_launcher_path, skill_launcher_path = write_launcher_files(
        artifact_dir,
        entrypoint_path,
        needs_tool_launcher=(kind == "tool" or register_as_tool),
        needs_skill_launcher=(kind == "skill"),
    )
    return CapabilityArtifacts(
        artifact_dir=artifact_dir,
        entrypoint_path=entrypoint_path,
        source_code=source_code,
        tool_launcher_path=tool_launcher_path,
        skill_launcher_path=skill_launcher_path,
        validation_issues=validation_issues,
    )
