"""从当前工作目录及父目录查找并读取 CLAUDE.md 文件，注入系统提示词。

遵循 CLAUDE.md 规范中的优先级规则：
- 更近层级（子目录）的 CLAUDE.md 优先级更高
- 当前目录的 CLAUDE.md 优先于父目录
"""

from __future__ import annotations

from pathlib import Path

# 最多向上查找的层级数
_MAX_PARENT_DEPTH = 10
# CLAUDE.md 文件最大读取字符数（避免占用过多上下文窗口）
_MAX_CLAUDE_MD_CHARS = 8000


def find_claude_md_files(base_dir: Path | None = None) -> list[Path]:
    """从当前目录向上查找所有 CLAUDE.md 文件，按由远到近排序（父目录在前）。"""
    current = (base_dir or Path.cwd()).resolve()
    found: list[Path] = []
    for _ in range(_MAX_PARENT_DEPTH):
        candidate = current / "CLAUDE.md"
        if candidate.exists() and candidate.is_file():
            found.append(candidate)
        parent = current.parent
        if parent == current:
            break
        current = parent
    # 反转使最近的目录排在最后（外层的先被读取，内层的可覆盖）
    found.reverse()
    return found


def read_claude_md_content(file_path: Path, max_chars: int = _MAX_CLAUDE_MD_CHARS) -> str:
    """读取 CLAUDE.md 文件内容，超出限制时截断并标注。"""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n... CLAUDE.md 内容过长已截断。"
    return content


def build_claude_md_prompt(base_dir: Path | None = None) -> str:
    """构建要注入系统提示词的 CLAUDE.md 上下文块。"""
    files = find_claude_md_files(base_dir)
    if not files:
        return ""

    parts: list[str] = []
    for file_path in files:
        content = read_claude_md_content(file_path)
        if not content.strip():
            continue
        # 显示相对路径便于辨认
        try:
            rel_path = file_path.relative_to(Path.cwd())
        except ValueError:
            rel_path = file_path
        parts.append(f"<!-- CLAUDE.md: {rel_path} -->\n{content}")

    if not parts:
        return ""

    return "\n\n---\n\n".join(parts)
