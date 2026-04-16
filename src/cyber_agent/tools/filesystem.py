from collections.abc import Iterable, Sequence
from pathlib import Path

from langchain_core.tools import tool
from .metadata import attach_tool_risk

MAX_FILE_READ_CHARS = 4000
MAX_DIRECTORY_ENTRIES = 200


def normalize_allowed_roots(allowed_roots: Iterable[Path | str]) -> list[Path]:
    """规范化允许访问的根路径，并按顺序去重。"""
    normalized_roots: list[Path] = []
    seen_roots: set[Path] = set()

    for raw_root in allowed_roots:
        root_path = Path(raw_root).expanduser().resolve()
        if root_path in seen_roots:
            continue
        normalized_roots.append(root_path)
        seen_roots.add(root_path)

    return normalized_roots


def describe_allowed_roots(allowed_roots: Sequence[Path]) -> list[str]:
    """返回适合状态展示的允许路径列表。"""
    return [str(root) for root in allowed_roots]


def resolve_permitted_path(path: str, allowed_roots: Sequence[Path]) -> Path:
    """
    将目标路径解析为绝对路径，并校验是否位于允许访问的根路径之下。
    """
    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        target_path = raw_path.resolve()
        for allowed_root in allowed_roots:
            if target_path == allowed_root or allowed_root in target_path.parents:
                return target_path
    else:
        candidate_paths: list[Path] = []
        cwd_candidate = (Path.cwd() / raw_path).resolve()
        candidate_paths.append(cwd_candidate)

        for allowed_root in allowed_roots:
            root_candidate = (allowed_root / raw_path).resolve()
            if root_candidate not in candidate_paths:
                candidate_paths.append(root_candidate)

        for candidate_path in candidate_paths:
            for allowed_root in allowed_roots:
                if (
                    candidate_path == allowed_root
                    or allowed_root in candidate_path.parents
                ) and candidate_path.exists():
                    return candidate_path

        for candidate_path in candidate_paths:
            for allowed_root in allowed_roots:
                if candidate_path == allowed_root or allowed_root in candidate_path.parents:
                    return candidate_path

    allowed_root_descriptions = "；".join(str(root) for root in allowed_roots)
    raise ValueError(
        "目标路径超出允许访问范围。"
        f"当前允许的根路径有：{allowed_root_descriptions}"
    )


def display_path(target_path: Path) -> str:
    """优先显示相对当前工作目录的路径，无法相对化时显示绝对路径。"""
    workspace_root = Path.cwd().resolve()
    if target_path == workspace_root:
        return "."
    if target_path == target_path.anchor:
        return str(target_path)

    try:
        return str(target_path.relative_to(workspace_root))
    except ValueError:
        return str(target_path)


def create_list_directory_tool(allowed_roots: Sequence[Path]):
    """创建带路径范围约束的目录读取工具。"""
    normalized_roots = normalize_allowed_roots(allowed_roots)

    @tool("list_directory")
    def list_directory(path: str = ".") -> str:
        """
        列出允许访问范围内指定目录的文件和子目录。
        适合在分析代码仓库结构或查找目标文件时使用。
        """
        try:
            target_path = resolve_permitted_path(path, normalized_roots)
        except ValueError as exc:
            return f"❌ {exc}"

        if not target_path.exists():
            return f"❌ 路径不存在：{path}"
        if not target_path.is_dir():
            return f"❌ 路径不是目录：{path}"

        entries = sorted(
            target_path.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.lower()),
        )
        if not entries:
            return f"目录 {display_path(target_path)} 为空。"

        lines = [f"目录: {display_path(target_path)}"]
        for entry in entries[:MAX_DIRECTORY_ENTRIES]:
            marker = "[DIR]" if entry.is_dir() else "[FILE]"
            lines.append(f"{marker} {display_path(entry)}")

        if len(entries) > MAX_DIRECTORY_ENTRIES:
            lines.append(
                f"... 结果过多，已截断为前 {MAX_DIRECTORY_ENTRIES} 项。"
            )

        return "\n".join(lines)

    return attach_tool_risk(list_directory, "read")


def create_read_text_file_tool(allowed_roots: Sequence[Path]):
    """创建带路径范围约束的文本文件读取工具。"""
    normalized_roots = normalize_allowed_roots(allowed_roots)

    @tool("read_text_file")
    def read_text_file(path: str, max_chars: int = MAX_FILE_READ_CHARS) -> str:
        """
        读取允许访问范围内的文本文件内容。
        适合在回答代码或配置问题前先查看真实文件内容。
        """
        if max_chars <= 0:
            return "❌ max_chars 必须大于 0。"

        try:
            target_path = resolve_permitted_path(path, normalized_roots)
        except ValueError as exc:
            return f"❌ {exc}"

        if not target_path.exists():
            return f"❌ 文件不存在：{path}"
        if not target_path.is_file():
            return f"❌ 路径不是文件：{path}"

        file_content = target_path.read_text(encoding="utf-8", errors="replace")
        shown_path = display_path(target_path)

        if len(file_content) <= max_chars:
            return f"文件: {shown_path}\n{file_content}"

        truncated_content = file_content[:max_chars]
        return (
            f"文件: {shown_path}\n"
            f"内容过长，已截断为前 {max_chars} 个字符：\n"
            f"{truncated_content}"
        )

    return attach_tool_risk(read_text_file, "read")


def create_write_text_file_tool(allowed_roots: Sequence[Path]):
    """创建受路径范围限制的整文件写入工具。"""
    normalized_roots = normalize_allowed_roots(allowed_roots)

    @tool("write_text_file")
    def write_text_file(
        path: str,
        content: str,
        create_directories: bool = True,
    ) -> str:
        """
        写入文本文件内容。
        适合在明确知道目标文件最终内容时直接覆盖写入。
        """
        try:
            target_path = resolve_permitted_path(path, normalized_roots)
        except ValueError as exc:
            return f"❌ {exc}"

        if create_directories:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        elif not target_path.parent.exists():
            return f"❌ 父目录不存在：{display_path(target_path.parent)}"

        target_path.write_text(content, encoding="utf-8")
        return (
            f"已写入文件：{display_path(target_path)}\n"
            f"字符数：{len(content)}"
        )

    return attach_tool_risk(write_text_file, "write")


def create_replace_in_file_tool(allowed_roots: Sequence[Path]):
    """创建基于文本替换的文件编辑工具。"""
    normalized_roots = normalize_allowed_roots(allowed_roots)

    @tool("replace_in_file")
    def replace_in_file(
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> str:
        """
        在文本文件中替换指定片段。
        适合对已有文件做小范围、可定位的精确修改。
        """
        if not old_text:
            return "❌ old_text 不能为空。"

        try:
            target_path = resolve_permitted_path(path, normalized_roots)
        except ValueError as exc:
            return f"❌ {exc}"

        if not target_path.exists():
            return f"❌ 文件不存在：{path}"
        if not target_path.is_file():
            return f"❌ 路径不是文件：{path}"

        original_content = target_path.read_text(encoding="utf-8", errors="replace")
        occurrence_count = original_content.count(old_text)
        if occurrence_count == 0:
            return "❌ 未找到要替换的文本片段。"
        if occurrence_count > 1 and not replace_all:
            return (
                "❌ 命中多个位置。"
                "如确认全部替换，请将 replace_all 设为 true。"
            )

        if replace_all:
            updated_content = original_content.replace(old_text, new_text)
            replaced_count = occurrence_count
        else:
            updated_content = original_content.replace(old_text, new_text, 1)
            replaced_count = 1

        target_path.write_text(updated_content, encoding="utf-8")
        return (
            f"已更新文件：{display_path(target_path)}\n"
            f"替换次数：{replaced_count}"
        )

    return attach_tool_risk(replace_in_file, "write")
