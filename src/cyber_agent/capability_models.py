"""能力注册表数据模型与工具函数。

从 capability_registry.py 拆分以控制单文件行数。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

CAPABILITY_STORAGE_DIRNAME = ".cyber-agent-cli-capabilities"
CAPABILITY_EXECUTION_TIMEOUT_SECONDS = 30
MAX_GENERATED_OUTPUT_CHARS = 4000
RESERVED_TOOL_NAMES = {
    "scan_port",
    "list_directory",
    "read_text_file",
    "write_text_file",
    "replace_in_file",
    "apply_unified_patch",
    "run_shell_command",
    "run_registered_tool",
    "search_web",
    "fetch_web_page",
    "create_generated_capability",
    "revise_generated_capability",
    "list_generated_capabilities",
    "show_generated_capability",
    "mark_generated_capability_satisfied",
}


@dataclass(slots=True)
class CapabilityRevision:
    """记录一次 capability 生成或修订的审计结果。"""

    revision: int
    created_at: str
    description: str
    feedback: str
    audit_score: int
    audit_summary: str
    audit_issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GeneratedCapability:
    """描述一个可持久化的动态 skill/tool 能力。"""

    name: str
    kind: str
    register_as_tool: bool
    description: str
    system_prompt: str
    tool_description: str
    usage_hint: str
    quality_checklist: list[str] = field(default_factory=list)
    smoke_requests: list[str] = field(default_factory=list)
    audit_score: int = 0
    audit_summary: str = ""
    audit_issues: list[str] = field(default_factory=list)
    audit_recommendations: list[str] = field(default_factory=list)
    status: str = "draft"
    enabled: bool = True
    revision: int = 1
    created_at: str = ""
    updated_at: str = ""
    feedback_history: list[str] = field(default_factory=list)
    revisions: list[CapabilityRevision] = field(default_factory=list)
    source_code: str = ""
    artifact_dir: str = ""
    entrypoint_path: str = ""
    tool_launcher_path: str = ""
    skill_launcher_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """将 dataclass 转为适合 JSON 落盘的结构。"""
        data = asdict(self)
        data["revisions"] = [asdict(revision) for revision in self.revisions]
        return data

    @classmethod
    def from_dict(cls, raw_data: dict[str, Any]) -> "GeneratedCapability":
        """从 JSON 结构恢复 capability。"""
        revisions = [
            CapabilityRevision(**revision_data)
            for revision_data in raw_data.get("revisions", [])
            if isinstance(revision_data, dict)
        ]
        return cls(
            name=str(raw_data.get("name", "")),
            kind=str(raw_data.get("kind", "skill")),
            register_as_tool=bool(raw_data.get("register_as_tool", False)),
            description=str(raw_data.get("description", "")),
            system_prompt=str(raw_data.get("system_prompt", "")),
            tool_description=str(raw_data.get("tool_description", "")),
            usage_hint=str(raw_data.get("usage_hint", "")),
            quality_checklist=[
                str(item)
                for item in raw_data.get("quality_checklist", [])
                if str(item).strip()
            ],
            smoke_requests=[
                str(item)
                for item in raw_data.get("smoke_requests", [])
                if str(item).strip()
            ],
            audit_score=int(raw_data.get("audit_score", 0)),
            audit_summary=str(raw_data.get("audit_summary", "")),
            audit_issues=[str(item) for item in raw_data.get("audit_issues", [])],
            audit_recommendations=[
                str(item) for item in raw_data.get("audit_recommendations", [])
            ],
            status=str(raw_data.get("status", "draft")),
            enabled=bool(raw_data.get("enabled", True)),
            revision=int(raw_data.get("revision", 1)),
            created_at=str(raw_data.get("created_at", "")),
            updated_at=str(raw_data.get("updated_at", "")),
            feedback_history=[str(item) for item in raw_data.get("feedback_history", [])],
            revisions=revisions,
            source_code=str(raw_data.get("source_code", "")),
            artifact_dir=str(raw_data.get("artifact_dir", "")),
            entrypoint_path=str(raw_data.get("entrypoint_path", "")),
            tool_launcher_path=str(raw_data.get("tool_launcher_path", "")),
            skill_launcher_path=str(raw_data.get("skill_launcher_path", "")),
        )




@dataclass(slots=True)
class CapabilityExecutionResult:
    """描述一次生成代码执行的结果。"""

    returncode: int
    stdout: str
    stderr: str

    @property
    def combined_output(self) -> str:
        combined_output = "\n".join(
            part for part in [self.stdout.strip(), self.stderr.strip()] if part
        ).strip()
        return combined_output or "无输出。"


def get_capability_storage_dir(base_dir: Path | None = None) -> Path:
    """返回 capability 存储目录，支持从任意目录启动时回溯查找。"""
    from .local_config import find_data_dir
    return find_data_dir(CAPABILITY_STORAGE_DIRNAME, base_dir)


def _extract_response_text(raw_content: object) -> str:
    """将模型响应内容统一压缩为文本。"""
    if isinstance(raw_content, list):
        return "".join(
            item if isinstance(item, str) else str(item.get("text", ""))
            for item in raw_content
        )
    return str(raw_content)


def _truncate_output(output: str, *, limit: int = MAX_GENERATED_OUTPUT_CHARS) -> str:
    """限制生成代码执行输出长度，避免审计提示词无限膨胀。"""
    truncated_output = output[:limit]
    if len(output) > limit:
        truncated_output += "\n... 输出过长，已截断。"
    return truncated_output
