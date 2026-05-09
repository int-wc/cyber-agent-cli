from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SKILL_DIRS = [
    Path.home() / ".claude" / "skills",
    Path(".claude/skills").resolve(),
]

SKILL_FILENAME = "SKILL.md"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(slots=True)
class LoadedSkill:
    """描述一个从 SKILL.md 文件加载的技能。"""

    name: str
    description: str
    source_path: Path
    body: str
    version: str = ""
    model: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def build_prompt_text(self) -> str:
        """构建注入系统提示的技能文本。"""
        lines = [
            f"## 技能: {self.name}",
            f"描述: {self.description}",
        ]
        if self.body.strip():
            lines.append(f"指令:\n{self.body.strip()}")
        return "\n".join(lines)


def _parse_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    """解析 SKILL.md 的 YAML 风格 frontmatter 和 Markdown 正文。"""
    frontmatter_match = _FRONTMATTER_RE.match(raw_text)
    if not frontmatter_match:
        return {}, raw_text

    frontmatter_text = frontmatter_match.group(1)
    body = raw_text[frontmatter_match.end():].strip()
    metadata: dict[str, Any] = {}

    lines = frontmatter_text.splitlines()
    line_index = 0
    while line_index < len(lines):
        stripped_line = lines[line_index].strip()
        line_index += 1
        if not stripped_line or stripped_line.startswith("#"):
            continue
        if ":" not in stripped_line:
            # 可能是 YAML 多行值的延续行
            continue
        key, _, value = stripped_line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue

        # 处理 YAML 多行指示符 (> | >- |-
        if value in (">", "|", ">-", "|-", ">+", "|+"):
            multiline_parts: list[str] = []
            while line_index < len(lines):
                next_line = lines[line_index]
                # 非缩进行且有内容 → 新的 key，停止收集
                if next_line and not next_line[0].isspace() and next_line.strip():
                    break
                if next_line.strip():
                    multiline_parts.append(next_line.strip())
                line_index += 1
            value = " ".join(multiline_parts)

        if not value:
            continue
        metadata[key] = value

    return metadata, body


def discover_skill_files(extra_dirs: list[str | Path] | None = None) -> list[Path]:
    """扫描默认和额外目录，返回按优先级排序的 SKILL.md 文件路径列表。
    项目级 (.claude/skills/) 优先于个人级 (~/.claude/skills/)，同目录内按名称排序。
    """
    scan_dirs: list[Path] = []
    # 项目级优先
    project_dir = Path(".claude/skills").resolve()
    scan_dirs.append(project_dir)
    # 个人级
    scan_dirs.append(Path.home() / ".claude" / "skills")
    # 额外目录
    for extra_dir in extra_dirs or []:
        scan_dirs.append(Path(extra_dir).expanduser().resolve())

    seen_names: set[str] = set()
    skill_files: list[Path] = []

    for scan_dir in scan_dirs:
        if not scan_dir.is_dir():
            continue
        for skill_dir in sorted(scan_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / SKILL_FILENAME
            if not skill_file.is_file():
                continue
            skill_name = skill_dir.name
            if skill_name in seen_names:
                continue
            seen_names.add(skill_name)
            skill_files.append(skill_file)

    return skill_files


def load_skill_from_file(skill_path: Path) -> LoadedSkill | None:
    """从单个 SKILL.md 文件加载技能定义。"""
    try:
        raw_text = skill_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    metadata, body = _parse_frontmatter(raw_text)
    name = metadata.get("name", "").strip()
    if not name:
        name = skill_path.parent.name

    description = metadata.get("description", "").strip()
    if not description:
        # 从正文首行提取描述
        first_paragraph = body.strip().split("\n\n")[0].strip()
        description = first_paragraph[:200]

    allowed_tools_str = metadata.get("allowed-tools", "")
    allowed_tools = [
        tool.strip() for tool in allowed_tools_str.split() if tool.strip()
    ]

    extra_metadata: dict[str, str] = {}
    for key in ("author", "category", "license", "compatibility"):
        value = metadata.get(key, "")
        if value:
            extra_metadata[key] = value

    return LoadedSkill(
        name=name,
        description=description,
        source_path=skill_path.resolve(),
        body=body,
        version=metadata.get("version", ""),
        model=metadata.get("model", ""),
        allowed_tools=allowed_tools,
        metadata=extra_metadata,
    )


def load_all_skills(extra_dirs: list[str | Path] | None = None) -> list[LoadedSkill]:
    """扫描并加载所有可用技能。"""
    skill_files = discover_skill_files(extra_dirs)
    skills: list[LoadedSkill] = []
    for skill_path in skill_files:
        skill = load_skill_from_file(skill_path)
        if skill is not None:
            skills.append(skill)
    return skills


def build_skill_system_prompt(skills: list[LoadedSkill]) -> str:
    """将所有已加载技能拼接为系统提示扩展文本。"""
    if not skills:
        return ""

    lines = ["以下是当前会话已激活的扩展 skills（从 SKILL.md 加载）："]
    for skill in skills:
        lines.append(skill.build_prompt_text())
    return "\n\n".join(lines)
