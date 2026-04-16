import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from .filesystem import display_path, normalize_allowed_roots, resolve_permitted_path

HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


@dataclass(slots=True)
class HunkLine:
    kind: Literal[" ", "+", "-"]
    text: str


@dataclass(slots=True)
class PatchHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[HunkLine]


@dataclass(slots=True)
class FilePatch:
    old_path: str | None
    new_path: str | None
    hunks: list[PatchHunk]


def _normalize_patch_path(raw_path: str) -> str | None:
    value = raw_path.split("\t", 1)[0].strip()
    if value == "/dev/null":
        return None
    if value.startswith("a/") or value.startswith("b/"):
        return value[2:]
    return value


def parse_unified_patch(patch_text: str) -> list[FilePatch]:
    """解析统一 diff 文本。"""
    lines = patch_text.splitlines()
    index = 0
    file_patches: list[FilePatch] = []

    while index < len(lines):
        current_line = lines[index]
        if current_line.startswith(("diff --git ", "index ", "new file mode ", "deleted file mode ", "similarity index ", "rename from ", "rename to ")):
            index += 1
            continue

        if not current_line.startswith("--- "):
            index += 1
            continue

        old_path = _normalize_patch_path(current_line[4:])
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise ValueError("补丁格式错误：缺少 +++ 文件头。")
        new_path = _normalize_patch_path(lines[index][4:])
        index += 1
        hunks: list[PatchHunk] = []

        while index < len(lines):
            current_line = lines[index]
            if current_line.startswith("--- "):
                break
            if current_line.startswith(("diff --git ", "index ", "new file mode ", "deleted file mode ", "similarity index ", "rename from ", "rename to ")):
                index += 1
                continue

            header_match = HUNK_HEADER_RE.match(current_line)
            if not header_match:
                if current_line.strip() == "":
                    index += 1
                    continue
                raise ValueError(f"补丁格式错误：无法识别的 hunk 头：{current_line}")

            old_start = int(header_match.group("old_start"))
            old_count = int(header_match.group("old_count") or "1")
            new_start = int(header_match.group("new_start"))
            new_count = int(header_match.group("new_count") or "1")
            index += 1
            hunk_lines: list[HunkLine] = []

            while index < len(lines):
                current_line = lines[index]
                if current_line.startswith(("@@ ", "@@")):
                    break
                if current_line.startswith("--- "):
                    break
                if current_line == r"\ No newline at end of file":
                    index += 1
                    continue
                if current_line and current_line[0] in {" ", "+", "-"}:
                    hunk_lines.append(HunkLine(current_line[0], current_line[1:]))
                    index += 1
                    continue
                raise ValueError(f"补丁格式错误：无法识别的 hunk 行：{current_line}")

            hunks.append(
                PatchHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=hunk_lines,
                )
            )

        file_patches.append(FilePatch(old_path=old_path, new_path=new_path, hunks=hunks))

    if not file_patches:
        raise ValueError("未解析到任何文件补丁。")

    return file_patches


def _render_new_text(updated_lines: list[str], had_trailing_newline: bool) -> str:
    if not updated_lines:
        return ""
    rendered_text = "\n".join(updated_lines)
    if had_trailing_newline:
        rendered_text += "\n"
    return rendered_text


def apply_file_patch(file_patch: FilePatch, allowed_roots: list[Path]) -> str:
    """将单文件统一补丁应用到目标文件。"""
    old_path = (
        resolve_permitted_path(file_patch.old_path, allowed_roots)
        if file_patch.old_path is not None
        else None
    )
    new_path = (
        resolve_permitted_path(file_patch.new_path, allowed_roots)
        if file_patch.new_path is not None
        else None
    )
    old_existed = old_path.exists() if old_path is not None else False

    if old_path is None and new_path is None:
        raise ValueError("补丁格式错误：旧路径和新路径不能同时为空。")

    if old_path is not None and old_existed:
        original_text = old_path.read_text(encoding="utf-8", errors="replace")
    else:
        original_text = ""

    original_lines = original_text.splitlines()
    had_trailing_newline = original_text.endswith("\n")
    output_lines: list[str] = []
    cursor = 0

    for hunk in file_patch.hunks:
        hunk_start_index = max(hunk.old_start - 1, 0)
        if hunk_start_index < cursor:
            raise ValueError("补丁格式错误：hunk 顺序重叠。")

        output_lines.extend(original_lines[cursor:hunk_start_index])
        cursor = hunk_start_index

        for hunk_line in hunk.lines:
            if hunk_line.kind == " ":
                if cursor >= len(original_lines) or original_lines[cursor] != hunk_line.text:
                    raise ValueError(
                        f"补丁应用失败：上下文不匹配，文件 {file_patch.old_path or file_patch.new_path}"
                    )
                output_lines.append(original_lines[cursor])
                cursor += 1
                continue
            if hunk_line.kind == "-":
                if cursor >= len(original_lines) or original_lines[cursor] != hunk_line.text:
                    raise ValueError(
                        f"补丁应用失败：删除行不匹配，文件 {file_patch.old_path or file_patch.new_path}"
                    )
                cursor += 1
                continue
            output_lines.append(hunk_line.text)

    output_lines.extend(original_lines[cursor:])

    if new_path is None:
        if old_path is not None and old_path.exists():
            old_path.unlink()
        return f"已删除文件：{display_path(old_path)}"

    new_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_text = _render_new_text(output_lines, had_trailing_newline or old_path is None)
    new_path.write_text(rendered_text, encoding="utf-8")

    if old_path is not None and new_path != old_path and old_path.exists():
        old_path.unlink()
        return (
            f"已重命名并更新文件：{display_path(old_path)} -> {display_path(new_path)}"
        )

    action = "已创建文件" if old_path is None or not old_existed else "已更新文件"
    return f"{action}：{display_path(new_path)}"


def create_apply_unified_patch_tool(allowed_roots: list[Path]):
    """创建统一补丁应用工具。"""
    normalized_roots = normalize_allowed_roots(allowed_roots)

    @tool("apply_unified_patch", extras={"risk": "write"})
    def apply_unified_patch(patch_text: str) -> str:
        """
        应用 unified diff 补丁到允许访问范围内的文本文件。
        适合进行多文件、结构化的批量修改。
        """
        try:
            file_patches = parse_unified_patch(patch_text)
        except ValueError as exc:
            return f"❌ {exc}"

        try:
            results = [
                apply_file_patch(file_patch, normalized_roots)
                for file_patch in file_patches
            ]
        except ValueError as exc:
            return f"❌ {exc}"

        return "\n".join(results)

    return apply_unified_patch
