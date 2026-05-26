"""跨会话持久化记忆系统。

记忆以 Markdown 文件存储在 .cyber-agent-memory/ 目录下，
通过 MEMORY.md 索引文件统一管理。每次会话可读取并更新记忆。

结构:
  .cyber-agent-memory/
    MEMORY.md               ← 索引文件，每行一条记忆摘要
    user_preferences.md      ← 用户偏好与习惯
    project_context.md       ← 项目背景与决策
    <custom>.md              ← 任意主题记忆

记忆文件格式（frontmatter + body）:
  ---
  name: short-kebab-slug
  description: 一行摘要
  metadata:
    type: user_preference | project_context | decision | feedback
  ---
  记忆正文...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MEMORY_STORAGE_DIRNAME = ".cyber-agent-memory"
MEMORY_INDEX_FILENAME = "MEMORY.md"
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(slots=True)
class MemoryEntry:
    """一条已解析的记忆条目。"""

    name: str
    description: str
    memory_type: str
    body: str
    file_path: Path
    updated_at: str = ""


@dataclass(slots=True)
class MemorySearchResult:
    """记忆检索命中结果。"""

    entry: MemoryEntry
    relevance: float
    excerpt: str


def get_memory_dir(base_dir: Path | None = None) -> Path:
    """返回记忆存储目录。"""
    return (base_dir or Path.cwd()).resolve() / MEMORY_STORAGE_DIRNAME


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """从 Markdown 内容中提取 frontmatter 和正文。"""
    match = FRONTMATTER_RE.match(content)
    if not match:
        return {}, content
    raw_yaml = match.group(1)
    body = content[match.end():]
    metadata: dict[str, Any] = {}
    current_key = ""
    for line in raw_yaml.splitlines():
        if not line.strip():
            continue
        if ":" in line and not line.startswith(" "):
            key, _, value = line.partition(":")
            current_key = key.strip()
            metadata[current_key] = value.strip()
        elif current_key and line.startswith("  "):
            # 列表/缩进值
            pass
    return metadata, body


def _build_frontmatter(name: str, description: str, memory_type: str) -> str:
    """构建记忆文件的 frontmatter。"""
    lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
        "metadata:",
        f"  type: {memory_type}",
        "---",
        "",
    ]
    return "\n".join(lines)


def save_memory(
    name: str,
    description: str,
    body: str,
    *,
    memory_type: str = "project_context",
    base_dir: Path | None = None,
) -> Path:
    """保存一条记忆到独立文件，并更新索引。"""
    memory_dir = get_memory_dir(base_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{name}.md"
    file_path = memory_dir / filename
    content = _build_frontmatter(name, description, memory_type) + body.rstrip() + "\n"
    file_path.write_text(content, encoding="utf-8")

    _update_memory_index(memory_dir)
    return file_path


def _update_memory_index(memory_dir: Path) -> None:
    """从所有 .md 文件重建 MEMORY.md 索引。"""
    index_path = memory_dir / MEMORY_INDEX_FILENAME
    entries: list[str] = []
    for md_file in sorted(memory_dir.glob("*.md")):
        if md_file.name == MEMORY_INDEX_FILENAME:
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            metadata, body = _parse_frontmatter(content)
            name = metadata.get("name", md_file.stem)
            description = metadata.get("description", "")
            if not description:
                description = body[:120].replace("\n", " ").strip()
            entries.append(f"- [{name}]({md_file.name}) — {description}")
        except (OSError, UnicodeDecodeError):
            continue

    index_content = (
        "# 记忆索引\n\n"
        + "\n".join(entries)
        + "\n"
    )
    index_path.write_text(index_content, encoding="utf-8")


def load_all_memories(base_dir: Path | None = None) -> list[MemoryEntry]:
    """读取所有已保存的记忆。"""
    memory_dir = get_memory_dir(base_dir)
    if not memory_dir.exists():
        return []

    entries: list[MemoryEntry] = []
    for md_file in sorted(memory_dir.glob("*.md")):
        if md_file.name == MEMORY_INDEX_FILENAME:
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            metadata, body = _parse_frontmatter(content)
            entries.append(MemoryEntry(
                name=metadata.get("name", md_file.stem),
                description=metadata.get("description", ""),
                memory_type=metadata.get("type", "unknown"),
                body=body.strip(),
                file_path=md_file,
                updated_at=datetime.fromtimestamp(
                    md_file.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            ))
        except (OSError, UnicodeDecodeError):
            continue
    return entries


def search_memories(
    query: str,
    *,
    base_dir: Path | None = None,
    limit: int = 5,
) -> list[MemorySearchResult]:
    """按关键词检索记忆，返回相关性排序的结果。"""
    normalized_query = query.strip().lower()
    if not normalized_query:
        return []

    results: list[MemorySearchResult] = []
    for entry in load_all_memories(base_dir):
        searchable = f"{entry.name} {entry.description} {entry.body}".lower()
        if normalized_query not in searchable:
            continue
        # 简单相关性：命中次数 / 总词数（粗略估算）
        hit_count = searchable.count(normalized_query)
        relevance = min(1.0, hit_count / max(1, len(searchable.split())))
        excerpt = _build_memory_excerpt(entry.body, normalized_query)
        results.append(MemorySearchResult(entry=entry, relevance=relevance, excerpt=excerpt))

    results.sort(key=lambda r: r.relevance, reverse=True)
    return results[:limit]


def _build_memory_excerpt(text: str, query: str, *, max_chars: int = 200) -> str:
    """围绕查询关键词生成记忆内容摘要。"""
    normalized_text = " ".join(text.split())
    if len(normalized_text) <= max_chars:
        return normalized_text

    lowered = normalized_text.lower()
    idx = lowered.find(query)
    if idx < 0:
        return f"{normalized_text[:max_chars]}..."

    window = max((max_chars - len(query)) // 2, 20)
    start = max(0, idx - window)
    end = min(len(normalized_text), idx + len(query) + window)
    excerpt = normalized_text[start:end]
    if start > 0:
        excerpt = f"...{excerpt}"
    if end < len(normalized_text):
        excerpt = f"{excerpt}..."
    return excerpt


def build_memory_system_prompt(base_dir: Path | None = None) -> str:
    """将已保存的记忆构建为可注入系统提示词的文本。

    仅供模型了解历史决策和用户偏好，不包含文件系统操作指令。
    """
    entries = load_all_memories(base_dir)
    if not entries:
        return ""

    type_groups: dict[str, list[MemoryEntry]] = {}
    for entry in entries:
        group_key = entry.memory_type or "other"
        type_groups.setdefault(group_key, []).append(entry)

    lines = ["## 持久化记忆", ""]
    type_labels = {
        "user_preference": "用户偏好",
        "project_context": "项目背景",
        "decision": "历史决策",
        "feedback": "用户反馈",
    }

    for mem_type, group in sorted(type_groups.items()):
        label = type_labels.get(mem_type, mem_type)
        lines.append(f"### {label}")
        for entry in group:
            lines.append(f"- **{entry.description}**")
            if entry.body:
                first_line = entry.body.strip().splitlines()[0][:150]
                lines.append(f"  {first_line}")
        lines.append("")

    lines.append("以上记忆来自历史会话，请在相关场景下参考。")
    return "\n".join(lines)


def delete_memory(name: str, *, base_dir: Path | None = None) -> bool:
    """删除指定记忆文件，更新索引。"""
    memory_dir = get_memory_dir(base_dir)
    file_path = memory_dir / f"{name}.md"
    if not file_path.exists():
        return False
    file_path.unlink()
    _update_memory_index(memory_dir)
    return True
