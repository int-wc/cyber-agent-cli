from __future__ import annotations

import ast
import json
import py_compile
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool

from ._lazy_imports import load_chat_openai, load_llm_for_api, is_anthropic_api
from .capability_codegen import (
    CAPABILITY_ENTRYPOINT_FILENAME,
    CAPABILITY_NAME_RE,
    CAPABILITY_SKILL_LAUNCHER_CMD,
    CAPABILITY_SKILL_LAUNCHER_SH,
    CAPABILITY_TOOL_LAUNCHER_CMD,
    CAPABILITY_TOOL_LAUNCHER_SH,
    CapabilityArtifacts,
    _strip_markdown_code_fence,
    build_capability_source,
    default_skill_python_code,
    default_tool_python_code,
    materialize_capability_artifacts,
    validate_capability_source,
)
from .config import settings
from .execution_control import ExecutionController, ExecutionInterruptedError
from .logging import log_capability_operation, log_error
from .openai_compat import ensure_deepseek_reasoning_content_compat
from .tools.metadata import attach_tool_risk
from .tools.system import _run_process_with_controller
from .capability_models import *  # noqa: F403  # 向后兼容：模型已拆分至独立模块


class CapabilityRegistry:
    """管理动态 skill/tool 的代码生成、审计、持久化与运行时注入。"""

    def __init__(
        self,
        *,
        execution_controller: ExecutionController | None = None,
        base_dir: Path | None = None,
        service_name: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from .local_config import get_application_home

        self.execution_controller = execution_controller
        self.base_dir = (base_dir or get_application_home()).resolve()
        self.storage_dir = get_capability_storage_dir(base_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.service_name = settings.normalize_service_name(service_name)
        self.model_name = settings.get_model_name(
            model_name,
            service_name=self.service_name,
        )
        self.api_key = settings.get_api_key(self.service_name, api_key=api_key)
        self.base_url = settings.resolve_base_url(self.service_name, base_url=base_url)
        self._capabilities: dict[str, GeneratedCapability] = {}
        self._refresh_callback = None
        self._llm: Any | None = None
        # 缓存：避免每次构建 prompt/工具集都重新 exec Python 代码
        self._skill_prompt_mtime_cache: dict[str, float] = {}
        self._dynamic_tools_cache: list[BaseTool] | None = None
        self._dynamic_tools_version: int = 0
        self._load_capabilities()

    def update_llm_config(
        self,
        *,
        service_name: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """更新 capability 生成和审计所使用的模型配置。"""
        if service_name is not None:
            self.service_name = settings.normalize_service_name(service_name)
        if model_name is not None:
            self.model_name = settings.get_model_name(
                model_name,
                service_name=self.service_name,
            )
        elif service_name is not None:
            self.model_name = settings.get_model_name(service_name=self.service_name)
        if api_key is not None or service_name is not None:
            self.api_key = settings.get_api_key(self.service_name, api_key=api_key)
        if service_name is not None or base_url is not None:
            self.base_url = settings.resolve_base_url(
                self.service_name,
                base_url=base_url,
            )
        self._llm = None

    def register_refresh_callback(self, callback) -> None:
        """注册 capability 变更后的刷新回调，用于重建当前 runner 工具集。"""
        self._refresh_callback = callback

    def list_capabilities(self) -> list[GeneratedCapability]:
        """返回按名称排序的已保存 capability 列表。"""
        return sorted(self._capabilities.values(), key=lambda item: item.name.lower())

    def get_capability(self, name: str) -> GeneratedCapability:
        """读取单个 capability。"""
        normalized_name = name.strip()
        capability = self._capabilities.get(normalized_name)
        if capability is None:
            raise ValueError(f"未找到 capability：{normalized_name}")
        return capability

    def build_skill_prompt(self, file_skills: list | None = None) -> str:
        """将当前启用的 skill 能力拼接为系统提示补充，可混入文件加载的 skill。"""
        parts: list[str] = []

        # 文件加载的 skill（SKILL.md）
        if file_skills:
            from .skill_loader import build_skill_system_prompt

            file_prompt = build_skill_system_prompt(file_skills)
            if file_prompt:
                parts.append(file_prompt)

        # 动态生成的 skill capability
        skill_capabilities = [
            capability
            for capability in self.list_capabilities()
            if capability.enabled and capability.kind == "skill"
        ]
        if skill_capabilities:
            lines = ["以下是当前会话已激活的动态 skills："]
            for index, capability in enumerate(skill_capabilities, start=1):
                prompt_text = capability.system_prompt.strip()
                if not prompt_text:
                    refreshed_prompt = self._refresh_skill_prompt_from_artifacts(capability)
                    prompt_text = refreshed_prompt.strip()
                lines.append(f"{index}. {capability.name}: {capability.description}")
                if prompt_text:
                    lines.append(f"   技能提示: {prompt_text}")
                if capability.usage_hint:
                    lines.append(f"   使用提示: {capability.usage_hint}")
            parts.append("\n".join(lines))

        return "\n\n".join(part for part in parts if part.strip())

    def get_dynamic_tools(self) -> list[BaseTool]:
        """返回当前所有启用的动态工具与 capability 管理工具，结果按能力版本缓存。"""
        current_version = sum(
            cap.revision for cap in self._capabilities.values() if cap.enabled and cap.register_as_tool
        )
        if self._dynamic_tools_cache is not None and self._dynamic_tools_version == current_version:
            return self._dynamic_tools_cache

        tools: list[BaseTool] = [
            self._create_generated_capability_tool(),
            self._create_revise_capability_tool(),
            self._create_list_capabilities_tool(),
            self._create_show_capability_tool(),
            self._create_mark_capability_satisfied_tool(),
        ]
        for capability in self.list_capabilities():
            if not capability.enabled or not capability.register_as_tool:
                continue
            tools.append(self._build_runtime_tool(capability))

        self._dynamic_tools_cache = tools
        self._dynamic_tools_version = current_version
        return tools

    def create_or_update_capability(
        self,
        *,
        name: str,
        kind: str,
        description: str,
        register_as_tool: bool,
        feedback: str = "",
    ) -> GeneratedCapability:
        """根据用户描述生成或修订代码型 capability，并完成审计和持久化。"""
        normalized_name = self._validate_capability_name(name)
        normalized_kind = kind.strip().lower()
        if normalized_kind not in {"skill", "tool"}:
            raise ValueError("capability_kind 仅支持 skill 或 tool。")

        description = description.strip()
        if not description:
            raise ValueError("description 不能为空。")

        previous_capability = self._capabilities.get(normalized_name)
        generated_spec = self._generate_capability_spec(
            name=normalized_name,
            kind=normalized_kind,
            description=description,
            register_as_tool=register_as_tool,
            feedback=feedback,
            previous_capability=previous_capability,
        )
        artifacts = self._materialize_capability_artifacts(
            name=normalized_name,
            kind=normalized_kind,
            description=description,
            register_as_tool=register_as_tool,
            generated_spec=generated_spec,
        )
        skill_prompt_output = ""
        if normalized_kind == "skill":
            prompt_result = self._execute_generated_capability(
                artifacts.entrypoint_path,
                mode="prompt",
            )
            if prompt_result.returncode == 0:
                skill_prompt_output = prompt_result.stdout.strip()
            else:
                artifacts.validation_issues.append(
                    f"skill 提示词渲染失败：{_truncate_output(prompt_result.combined_output)}"
                )

        audit_result = self._audit_capability(
            capability_spec=generated_spec,
            artifacts=artifacts,
            skill_prompt_output=skill_prompt_output,
            register_as_tool=register_as_tool,
            kind=normalized_kind,
        )

        timestamp = datetime.now().astimezone().isoformat()
        revision_number = (
            previous_capability.revision + 1 if previous_capability is not None else 1
        )
        created_at = previous_capability.created_at if previous_capability else timestamp
        feedback_history = (
            list(previous_capability.feedback_history) if previous_capability else []
        )
        if feedback.strip():
            feedback_history.append(feedback.strip())

        revisions = list(previous_capability.revisions) if previous_capability else []
        revisions.append(
            CapabilityRevision(
                revision=revision_number,
                created_at=timestamp,
                description=description,
                feedback=feedback,
                audit_score=int(audit_result.get("score", 0)),
                audit_summary=str(audit_result.get("summary", "")).strip(),
                audit_issues=[
                    str(item)
                    for item in audit_result.get("issues", [])
                    if str(item).strip()
                ],
            )
        )

        capability = GeneratedCapability(
            name=normalized_name,
            kind=normalized_kind,
            register_as_tool=register_as_tool,
            description=description,
            system_prompt=skill_prompt_output.strip(),
            tool_description=str(generated_spec.get("tool_description", "")).strip(),
            usage_hint=str(generated_spec.get("usage_hint", "")).strip(),
            quality_checklist=[
                str(item)
                for item in generated_spec.get("quality_checklist", [])
                if str(item).strip()
            ],
            smoke_requests=[
                str(item)
                for item in generated_spec.get("smoke_requests", [])
                if str(item).strip()
            ][:2],
            audit_score=int(audit_result.get("score", 0)),
            audit_summary=str(audit_result.get("summary", "")).strip(),
            audit_issues=[
                str(item)
                for item in audit_result.get("issues", [])
                if str(item).strip()
            ],
            audit_recommendations=[
                str(item)
                for item in audit_result.get("recommendations", [])
                if str(item).strip()
            ],
            status=(
                "awaiting_user_feedback"
                if int(audit_result.get("score", 0)) >= settings.capability_audit_min_score
                else "needs_feedback"
            ),
            enabled=True,
            revision=revision_number,
            created_at=created_at,
            updated_at=timestamp,
            feedback_history=feedback_history,
            revisions=revisions,
            source_code=artifacts.source_code,
            artifact_dir=str(artifacts.artifact_dir),
            entrypoint_path=str(artifacts.entrypoint_path),
            tool_launcher_path=str(artifacts.tool_launcher_path or ""),
            skill_launcher_path=str(artifacts.skill_launcher_path or ""),
        )
        self._capabilities[normalized_name] = capability
        self._save_capability(capability)
        self._trigger_refresh()
        log_capability_operation(
            "create_or_update",
            normalized_name,
            kind=normalized_kind,
            audit_score=capability.audit_score,
        )
        return capability

    def mark_capability_satisfied(self, name: str, *, notes: str = "") -> GeneratedCapability:
        capability = self.get_capability(name.strip())
        # 重新审计以更新分数
        if capability.kind == "skill":
            self._refresh_skill_prompt_from_artifacts(capability)
        artifacts = CapabilityArtifacts(
            artifact_dir=Path(capability.artifact_dir),
            entrypoint_path=Path(capability.entrypoint_path),
            source_code=capability.source_code,
            tool_launcher_path=None,
            skill_launcher_path=None,
        )
        audit_result = self._audit_capability(
            capability_spec=capability.to_dict(),
            artifacts=artifacts,
            skill_prompt_output=capability.system_prompt,
            register_as_tool=capability.register_as_tool,
            kind=capability.kind,
        )
        capability.audit_score = int(audit_result.get("score", 0))
        capability.audit_summary = audit_result.get("summary", "")
        capability.audit_issues = audit_result.get("issues", [])
        capability.audit_recommendations = audit_result.get("recommendations", [])
        capability.status = "satisfied"
        capability.updated_at = datetime.now().astimezone().isoformat()
        if notes.strip():
            capability.feedback_history.append(notes.strip())
        self._save_capability(capability)
        return capability

    def _trigger_refresh(self) -> None:
        """在 capability 集合变更后刷新当前 runner 并清空缓存。"""
        self._dynamic_tools_cache = None
        self._dynamic_tools_version = 0
        if self._refresh_callback is not None:
            self._refresh_callback()

    def _load_capabilities(self) -> None:
        """启动时从磁盘恢复已生成的 capability。"""
        for capability_file in self.storage_dir.glob("*.json"):
            try:
                raw_data = json.loads(capability_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(raw_data, dict):
                continue
            capability = GeneratedCapability.from_dict(raw_data)
            if not capability.name:
                continue
            if capability.kind == "skill":
                self._refresh_skill_prompt_from_artifacts(capability)
            self._capabilities[capability.name] = capability

    def _save_capability(self, capability: GeneratedCapability) -> None:
        """将 capability 落盘为 JSON 文件。"""
        target_file = self.storage_dir / f"{capability.name}.json"
        target_file.write_text(
            json.dumps(capability.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _validate_capability_name(self, raw_name: str) -> str:
        """校验 capability 名称，确保它既适合落盘又能作为 tool 名。"""
        normalized_name = raw_name.strip()
        if not CAPABILITY_NAME_RE.fullmatch(normalized_name):
            raise ValueError(
                "capability 名称必须以字母开头，且仅允许字母、数字、下划线和短横线，长度 3-64。"
            )
        if normalized_name in RESERVED_TOOL_NAMES:
            raise ValueError(f"capability 名称与现有工具冲突：{normalized_name}")
        return normalized_name

    def _get_llm(self) -> Any:
        """懒加载用于生成与审计 capability 的模型实例。"""
        if self._llm is None:
            try:
                llm_cls, is_anthropic = load_llm_for_api(self.base_url)
            except ModuleNotFoundError as exc:
                missing = "langchain_anthropic" if is_anthropic_api(self.base_url) else "langchain_openai"
                raise ModuleNotFoundError(
                    f"缺少 `{missing}` 依赖，当前环境无法创建 capability 模型客户端。"
                ) from exc
            if not is_anthropic and self.service_name == "deepseek":
                ensure_deepseek_reasoning_content_compat()
            kwargs = settings.get_chat_openai_kwargs(
                self.service_name,
                model_name=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
            )
            if is_anthropic:
                kwargs["anthropic_api_key"] = kwargs.pop("api_key", "")
                kwargs.pop("openai_api_key", None)
                kwargs.pop("openai_proxy", None)
            self._llm = llm_cls(**kwargs)
        return self._llm

    def _ensure_not_cancelled(self) -> None:
        """在长链路 capability 操作中响应 /stop。"""
        if self.execution_controller is not None:
            self.execution_controller.ensure_not_cancelled()

    def _invoke_json_prompt(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """调用模型并强制解析 JSON 输出。"""
        self._ensure_not_cancelled()
        response = self._get_llm().invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        self._ensure_not_cancelled()
        raw_text = _extract_response_text(getattr(response, "content", ""))
        try:
            parsed_data = json.loads(_strip_markdown_code_fence(raw_text))
        except json.JSONDecodeError as exc:
            raise ValueError(f"模型未返回合法 JSON：{exc}") from exc
        if not isinstance(parsed_data, dict):
            raise ValueError("模型返回的 capability 结构必须是 JSON 对象。")
        return parsed_data

    def invoke_json_prompt(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """对外复用统一的 JSON 模型调用封装。"""
        return self._invoke_json_prompt(system_prompt, user_prompt)

    def _generate_capability_spec(
        self,
        *,
        name: str,
        kind: str,
        description: str,
        register_as_tool: bool,
        feedback: str,
        previous_capability: GeneratedCapability | None,
    ) -> dict[str, Any]:
        """让模型根据用户描述生成结构化的 capability 代码规格。"""
        previous_capability_json = (
            json.dumps(previous_capability.to_dict(), ensure_ascii=False, indent=2)
            if previous_capability is not None
            else "无"
        )
        system_prompt = """
你是动态 capability 代码生成器。你要根据用户需求，产出可落盘为真实 Python 文件并注入当前 agent 的 capability 规格。
输出必须是 JSON 对象，不要输出 Markdown，不要输出额外说明。

JSON 字段要求：
- name: string，必须保留用户给定名称
- kind: string，只能是 skill 或 tool
- register_as_tool: boolean
- tool_description: string，描述该能力被注册为工具后应如何被模型调用
- usage_hint: string，提示该能力适合在什么场景使用
- quality_checklist: string[]，列出 3 到 5 条质量检查点
- smoke_requests: string[]，列出 1 到 2 条烟雾测试输入
- tool_python_code: string，当 kind=tool 或 register_as_tool=true 时必须提供
  约束：这段代码必须定义 handle_request(request: str, context: str) -> str
- skill_python_code: string，当 kind=skill 时必须提供
  约束：这段代码必须定义 build_skill_prompt() -> str

代码生成约束：
1. 仅使用 Python 标准库，不要依赖额外第三方包。
2. 注释、TODO、占位说明全部使用中文。
3. 如果业务细节不足，输出最小可运行骨架，并显式写出 TODO(人工实现)。
4. 不要虚构系统权限、外部密钥、数据库字段或未声明的环境能力。
5. 不要生成入口 main，也不要重复定义 argparse 包装层，平台会自动拼装外层运行模板。
6. 如果用户提供了反馈，必须吸收反馈并修订代码与说明。
""".strip()
        user_prompt = (
            f"capability 名称: {name}\n"
            f"capability 类型: {kind}\n"
            f"是否注册为 tool: {str(register_as_tool).lower()}\n\n"
            f"用户描述:\n{description}\n\n"
            f"历史版本:\n{previous_capability_json}\n\n"
            f"用户反馈:\n{feedback or '无'}"
        )
        generated_spec = self._invoke_json_prompt(system_prompt, user_prompt)
        generated_spec["name"] = name
        generated_spec["kind"] = kind
        generated_spec["register_as_tool"] = register_as_tool
        return generated_spec

    def _build_capability_source(
        self,
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

    def _validate_capability_source(
        self,
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

    def _write_launcher_files(
        self,
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

    def _materialize_capability_artifacts(
        self,
        *,
        name: str,
        kind: str,
        description: str,
        register_as_tool: bool,
        generated_spec: dict[str, Any],
    ) -> CapabilityArtifacts:
        """将 capability 真实落盘为 Python 代码文件与启动脚本。"""
        artifact_dir = self.storage_dir / name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        source_code = self._build_capability_source(
            name=name,
            kind=kind,
            description=description,
            register_as_tool=register_as_tool,
            tool_python_code=str(generated_spec.get("tool_python_code", "")),
            skill_python_code=str(generated_spec.get("skill_python_code", "")),
        )
        entrypoint_path = artifact_dir / CAPABILITY_ENTRYPOINT_FILENAME
        entrypoint_path.write_text(source_code, encoding="utf-8")

        validation_issues = self._validate_capability_source(
            source_code,
            requires_tool=(kind == "tool" or register_as_tool),
            requires_skill=(kind == "skill"),
        )
        try:
            py_compile.compile(str(entrypoint_path), doraise=True)
        except py_compile.PyCompileError as exc:
            validation_issues.append(f"生成代码编译失败：{exc.msg}")

        tool_launcher_path, skill_launcher_path = self._write_launcher_files(
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

    def _execute_generated_capability(
        self,
        entrypoint_path: Path,
        *,
        mode: str,
        request: str = "",
        context: str = "",
    ) -> CapabilityExecutionResult:
        """执行已生成的 capability 代码文件。"""
        command = [str(Path(sys.executable).resolve()), str(entrypoint_path), mode]
        if mode == "run":
            command.extend(["--request", request, "--context", context])

        self._ensure_not_cancelled()
        try:
            completed_process = _run_process_with_controller(
                command,
                working_directory=entrypoint_path.parent,
                timeout_seconds=CAPABILITY_EXECUTION_TIMEOUT_SECONDS,
                execution_controller=self.execution_controller,
            )
        except ExecutionInterruptedError:
            raise
        except Exception as exc:
            return CapabilityExecutionResult(
                returncode=1,
                stdout="",
                stderr=f"执行生成代码失败：{exc}",
            )
        self._ensure_not_cancelled()
        return CapabilityExecutionResult(
            returncode=int(completed_process.returncode),
            stdout=completed_process.stdout or "",
            stderr=completed_process.stderr or "",
        )

    def _refresh_skill_prompt_from_artifacts(self, capability: GeneratedCapability) -> str:
        """从真实代码文件重新提取 skill 提示词，仅在文件修改后才重新 exec。"""
        if capability.kind != "skill" or not capability.entrypoint_path:
            return capability.system_prompt

        entrypoint_path = Path(capability.entrypoint_path)
        if not entrypoint_path.exists():
            return capability.system_prompt

        try:
            current_mtime = entrypoint_path.stat().st_mtime
        except OSError:
            return capability.system_prompt

        cached_mtime = self._skill_prompt_mtime_cache.get(capability.name)
        if cached_mtime is not None and cached_mtime >= current_mtime and capability.system_prompt:
            return capability.system_prompt

        prompt_result = self._execute_generated_capability(entrypoint_path, mode="prompt")
        if prompt_result.returncode != 0:
            return capability.system_prompt

        capability.system_prompt = prompt_result.stdout.strip()
        self._skill_prompt_mtime_cache[capability.name] = current_mtime
        return capability.system_prompt

    def _audit_capability(
        self,
        *,
        capability_spec: dict[str, Any],
        artifacts: CapabilityArtifacts,
        skill_prompt_output: str,
        register_as_tool: bool,
        kind: str,
    ) -> dict[str, Any]:
        """对生成的代码型 capability 做结构检查与烟雾测试审计。"""
        smoke_outputs: list[dict[str, str]] = []
        if not artifacts.validation_issues and (kind == "tool" or register_as_tool):
            for smoke_request in capability_spec.get("smoke_requests", [])[:2]:
                smoke_request_text = str(smoke_request).strip()
                if not smoke_request_text:
                    continue
                smoke_result = self._execute_generated_capability(
                    artifacts.entrypoint_path,
                    mode="run",
                    request=smoke_request_text,
                )
                smoke_outputs.append(
                    {
                        "request": smoke_request_text,
                        "returncode": str(smoke_result.returncode),
                        "output": _truncate_output(smoke_result.combined_output),
                    }
                )

        system_prompt = """
你是 capability 代码审计器。请审查候选 capability 的规格、生成代码、skill 提示词输出与烟雾测试结果。
输出必须是 JSON 对象，不要输出额外说明。

JSON 字段要求：
- score: integer，0 到 100
- summary: string，用中文概括当前质量
- issues: string[]，列出主要问题，没有则返回空数组
- recommendations: string[]，列出可执行改进建议，没有则返回空数组

审计重点：
1. 是否生成了真实、可运行、可继续维护的代码文件。
2. tool 逻辑或 skill 提示词是否与用户目标一致。
3. 输出边界是否清晰，是否保留了必要的 TODO(人工实现)。
4. 烟雾测试是否稳定，错误信息是否可诊断。
""".strip()
        user_prompt = (
            "候选 capability 规格:\n"
            f"{json.dumps(capability_spec, ensure_ascii=False, indent=2)}\n\n"
            "代码文件路径:\n"
            f"{artifacts.entrypoint_path}\n\n"
            "skill 提示词输出:\n"
            f"{skill_prompt_output or '无'}\n\n"
            "烟雾测试输出:\n"
            f"{json.dumps(smoke_outputs, ensure_ascii=False, indent=2)}\n\n"
            "基础校验问题:\n"
            f"{json.dumps(artifacts.validation_issues, ensure_ascii=False, indent=2)}"
        )
        audit_result = self._invoke_json_prompt(system_prompt, user_prompt)

        heuristic_issues: list[str] = list(artifacts.validation_issues)
        if not str(capability_spec.get("tool_description", "")).strip():
            heuristic_issues.append("tool_description 为空。")
        if not capability_spec.get("quality_checklist"):
            heuristic_issues.append("quality_checklist 为空。")
        if kind == "skill" and not skill_prompt_output.strip():
            heuristic_issues.append("skill 提示词输出为空。")
        if (kind == "tool" or register_as_tool) and not str(
            capability_spec.get("tool_python_code", "")
        ).strip():
            heuristic_issues.append("tool_python_code 为空。")
        if kind == "skill" and not str(capability_spec.get("skill_python_code", "")).strip():
            heuristic_issues.append("skill_python_code 为空。")

        if heuristic_issues:
            audit_result.setdefault("issues", [])
            audit_result["issues"] = [
                *[str(item) for item in audit_result.get("issues", [])],
                *heuristic_issues,
            ]
            audit_result["score"] = min(int(audit_result.get("score", 0)), 60)
        return audit_result

    def _build_runtime_tool(self, capability: GeneratedCapability) -> BaseTool:
        """将 capability 包装成当前 agent 可直接调用的动态工具。
        工具函数在每次调用时从 registry 重新获取 capability 对象，
        确保修订后的路径和描述能即时生效。"""
        capability_name = capability.name

        @tool(capability_name)
        def generated_capability_tool(
            request: str,
            context: str = "",
        ) -> str:
            """
            基于当前注入的动态 capability 执行请求。
            request 用于描述本次具体需求，context 可补充额外上下文。
            """
            if not request.strip():
                return "❌ request 不能为空。"

            current_capability = self._capabilities.get(capability_name)
            if current_capability is None:
                return f"❌ 未找到 capability：{capability_name}"

            entrypoint_path = self._resolve_capability_entrypoint(current_capability)
            if entrypoint_path is None:
                return (
                    f"❌ capability `{capability_name}` 缺少可执行代码文件。"
                    f" 请通过 revise_generated_capability 重新生成。"
                )

            execution_result = self._execute_generated_capability(
                entrypoint_path,
                mode="run",
                request=request,
                context=context,
            )
            if execution_result.returncode != 0:
                return (
                    "❌ 生成工具执行失败：\n"
                    f"{_truncate_output(execution_result.combined_output)}"
                )
            return execution_result.stdout.strip() or "无输出。"

        generated_capability_tool.description = capability.tool_description
        return attach_tool_risk(generated_capability_tool, "execute")

    def _resolve_capability_entrypoint(
        self,
        capability: GeneratedCapability,
    ) -> Path | None:
        """解析 capability 的入口文件路径，在存储路径不存在时执行兜底策略。"""
        if capability.entrypoint_path:
            saved_path = Path(capability.entrypoint_path)
            if saved_path.exists():
                return saved_path

        # 兜底：从 storage_dir 按名称重建路径
        fallback_path = self.storage_dir / capability.name / "capability.py"
        if fallback_path.exists():
            return fallback_path

        # 最后尝试：如果源码还在，可以重新落盘
        if capability.source_code:
            artifact_dir = self.storage_dir / capability.name
            artifact_dir.mkdir(parents=True, exist_ok=True)
            entrypoint = artifact_dir / "capability.py"
            try:
                entrypoint.write_text(capability.source_code, encoding="utf-8")
            except OSError:
                return None
            if entrypoint.exists():
                return entrypoint

        return None

    def _create_generated_capability_tool(self) -> BaseTool:
        """创建用于生成 capability 的管理工具。"""

        @tool("create_generated_capability")
        def create_generated_capability(
            capability_name: str,
            capability_kind: str,
            description: str,
            register_as_tool: bool = True,
        ) -> str:
            """
            根据用户描述生成真实代码文件形式的动态 capability，
            并在需要时注册成当前 agent 可直接调用的 tool。
            capability_kind 仅支持 skill 或 tool。
            这一步只负责首次生成与自动审计，不代表当前生成任务已经完成。
            """
            capability = self.create_or_update_capability(
                name=capability_name,
                kind=capability_kind,
                description=description,
                register_as_tool=register_as_tool,
            )
            return self._render_capability_summary(capability)

        return attach_tool_risk(create_generated_capability, "write")

    def _create_revise_capability_tool(self) -> BaseTool:
        """创建用于根据用户反馈修订 capability 的管理工具。"""

        @tool("revise_generated_capability")
        def revise_generated_capability(
            capability_name: str,
            feedback: str,
        ) -> str:
            """
            根据用户反馈修订已有 capability 的真实代码文件，并重新完成审计。
            适合在用户指出不足后持续改进，直到用户满意。
            """
            existing_capability = self.get_capability(capability_name)
            updated_capability = self.create_or_update_capability(
                name=existing_capability.name,
                kind=existing_capability.kind,
                description=existing_capability.description,
                register_as_tool=existing_capability.register_as_tool,
                feedback=feedback,
            )
            return self._render_capability_summary(updated_capability)

        return attach_tool_risk(revise_generated_capability, "write")

    def _create_list_capabilities_tool(self) -> BaseTool:
        """创建 capability 列表查询工具。"""

        @tool("list_generated_capabilities")
        def list_generated_capabilities() -> str:
            """
            列出当前工作目录下已保存的动态 capability。
            适合在首次生成前检查是否已有相近能力，或在继续修订前确认状态。
            """
            capabilities = self.list_capabilities()
            if not capabilities:
                return "当前没有已保存的动态 capability。"

            lines = ["已保存 capability 列表："]
            for capability in capabilities:
                lines.append(
                    f"- {capability.name} | kind={capability.kind} | "
                    f"tool={str(capability.register_as_tool).lower()} | "
                    f"status={capability.status} | revision={capability.revision}"
                )
            return "\n".join(lines)

        return attach_tool_risk(list_generated_capabilities, "read")

    def _create_show_capability_tool(self) -> BaseTool:
        """创建 capability 详情查看工具。"""

        @tool("show_generated_capability")
        def show_generated_capability(capability_name: str) -> str:
            """查看指定 capability 的完整定义、文件路径和最新审计结果，供向用户说明与验收。"""
            capability = self.get_capability(capability_name)
            return self._render_capability_summary(capability, detailed=True)

        return attach_tool_risk(show_generated_capability, "read")

    def _create_mark_capability_satisfied_tool(self) -> BaseTool:
        """创建 capability 满意度标记工具。"""

        @tool("mark_generated_capability_satisfied")
        def mark_generated_capability_satisfied(
            capability_name: str,
            notes: str = "",
        ) -> str:
            """
            只有当用户明确确认 capability 已满足需求时，才将其标记为 satisfied。
            """
            capability = self.mark_capability_satisfied(capability_name, notes=notes)
            return (
                f"已将 capability `{capability.name}` 标记为 satisfied。\n"
                f"当前 revision: {capability.revision}\n"
                f"审计分数: {capability.audit_score}"
            )

        return attach_tool_risk(mark_generated_capability_satisfied, "write")

    def _render_capability_summary(
        self,
        capability: GeneratedCapability,
        *,
        detailed: bool = False,
    ) -> str:
        """将 capability 压缩为便于当前模型继续利用的文本摘要。"""
        lines = [
            f"capability: {capability.name}",
            f"kind: {capability.kind}",
            f"register_as_tool: {str(capability.register_as_tool).lower()}",
            f"status: {capability.status}",
            f"revision: {capability.revision}",
            f"audit_score: {capability.audit_score}",
            f"description: {capability.description}",
            f"tool_description: {capability.tool_description}",
            f"usage_hint: {capability.usage_hint}",
            f"entrypoint_path: {capability.entrypoint_path or '无'}",
            f"tool_launcher_path: {capability.tool_launcher_path or '无'}",
            f"skill_launcher_path: {capability.skill_launcher_path or '无'}",
            f"audit_summary: {capability.audit_summary or '无'}",
        ]
        if capability.audit_issues:
            lines.append("audit_issues:")
            lines.extend(f"- {issue}" for issue in capability.audit_issues)
        if capability.audit_recommendations:
            lines.append("audit_recommendations:")
            lines.extend(f"- {item}" for item in capability.audit_recommendations)
        if capability.status != "satisfied":
            if capability.status == "needs_feedback":
                lines.append("workflow_status: 自动审计认为当前实现仍需继续修订。")
            else:
                lines.append("workflow_status: 自动审计通过，当前仍需等待用户验收确认。")
            lines.append("next_action:")
            lines.append("- 如需查看完整定义、源代码和审计细节，请调用 show_generated_capability。")
            lines.append("- 需要继续完善时，请结合用户反馈调用 revise_generated_capability。")
            lines.append("- 只有当用户明确满意时，才能调用 mark_generated_capability_satisfied。")
        if detailed:
            if capability.system_prompt:
                lines.append("system_prompt:")
                lines.append(capability.system_prompt)
            if capability.quality_checklist:
                lines.append("quality_checklist:")
                lines.extend(f"- {item}" for item in capability.quality_checklist)
            if capability.smoke_requests:
                lines.append("smoke_requests:")
                lines.extend(f"- {item}" for item in capability.smoke_requests)
            if capability.source_code:
                lines.append("source_code:")
                lines.append(capability.source_code)
        return "\n".join(lines)
