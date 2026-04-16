from dataclasses import dataclass
from enum import StrEnum


class ApprovalPolicy(StrEnum):
    """定义高风险工具调用的审批策略。"""

    PROMPT = "prompt"
    AUTO = "auto"
    NEVER = "never"


APPROVAL_POLICY_LABELS: dict[ApprovalPolicy, str] = {
    ApprovalPolicy.PROMPT: "交互审批",
    ApprovalPolicy.AUTO: "自动批准",
    ApprovalPolicy.NEVER: "全部拒绝",
}


@dataclass(slots=True)
class ApprovalDecision:
    """描述一次审批的最终结果。"""

    approved: bool
    reason: str


def parse_approval_policy(raw_value: str) -> ApprovalPolicy:
    """将外部输入解析为审批策略枚举。"""
    normalized_value = raw_value.strip().lower()
    try:
        return ApprovalPolicy(normalized_value)
    except ValueError as exc:
        supported_policies = ", ".join(policy.value for policy in ApprovalPolicy)
        raise ValueError(
            f"不支持的审批策略：{raw_value}。可选值：{supported_policies}"
        ) from exc


def get_approval_policy_label(policy: ApprovalPolicy) -> str:
    """返回适合 CLI 展示的审批策略名称。"""
    return APPROVAL_POLICY_LABELS[policy]
