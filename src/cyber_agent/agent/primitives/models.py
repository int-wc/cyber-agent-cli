"""业务原语模型：原语判定、作用对象、端点与链的数据结构。

从 BSRC_SKILLS_V1 的「业务原语解析 + 原语链利用」方法论移植而来。
核心思想：不按 API 名字或参数名判断一个端点，而是判定它「到底对
什么东西做什么操作」——即核心业务原语 business_attr，再按原语选
攻击基元；多个原语通过业务信任串联成链（primitive-chains.json）
才可能构成有效危害。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BusinessPrimitive(StrEnum):
    """业务原语（business_attr）：端点「对什么东西做什么操作」的本质。"""

    READ_FILE = "read_file"        # 读文件/本地资源
    WRITE_FILE = "write_file"      # 写文件/上传/持久化
    EXEC_CODE = "exec_code"        # 执行代码/表达式/模板
    MODIFY_STATE = "modify_state"  # 修改状态/流程/配置
    QUERY_DATA = "query_data"      # 查询数据/列表/详情
    TRANSFER = "transfer"          # 拉取远程 URL / 转发
    AUTH = "auth"                  # 认证/token 签发/登录

    @classmethod
    def parse(cls, value: str) -> BusinessPrimitive | None:
        """宽松解析：容忍大小写与未知值，未知返回 None。"""
        if not value:
            return None
        v = str(value).strip().lower()
        for member in cls:
            if member.value == v:
                return member
        return None


class AttrTarget(StrEnum):
    """原语作用对象（attr_target）：原语施加在什么上面。"""

    LOCAL_FS = "local_fs"      # 本地文件系统
    REMOTE_URL = "remote_url"  # 远程 URL
    DB = "db"                  # 数据库
    TEMPLATE = "template"      # 模板/表达式/配置渲染
    USER_INPUT = "user_input"  # 用户输入
    WORKER = "worker"          # 后台任务/worker

    @classmethod
    def parse(cls, value: str) -> AttrTarget | None:
        """宽松解析：容忍大小写与未知值，未知返回 None。"""
        if not value:
            return None
        v = str(value).strip().lower()
        for member in cls:
            if member.value == v:
                return member
        return None


@dataclass
class PrimitiveEndpoint:
    """已判定原语的端点。

    对应 BSRC 深度分析 Step5 输出的一行：
    `- /xxx POST business_attr=transfer attr_target=remote_url attr_reason=... params={...} risk=...`
    """

    endpoint: str
    method: str = "GET"
    business_attr: BusinessPrimitive | None = None
    attr_target: AttrTarget | None = None
    attr_reason: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    risk: str = ""

    @property
    def attr_key(self) -> str:
        """原语 key（可能为未知值，仍返回原始字符串用于链匹配）。"""
        return self.business_attr.value if self.business_attr else "unknown"

    def to_dict(self) -> dict[str, Any]:
        """转为可写回/可持久化的 dict。"""
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "business_attr": self.attr_key,
            "attr_target": self.attr_target.value if self.attr_target else None,
            "attr_reason": self.attr_reason,
            "params": self.params,
            "risk": self.risk,
        }


@dataclass
class PrimitiveChain:
    """原语链模板：多个原语通过业务信任串联成有效危害。

    对应 primitive-chains.json 的 chains 条目：
    - primitives: 链模板的组成原语
    - logic: 串联逻辑
    - gain: 危害升级（低危原语 → 高危害）
    - instances: 验证成功回写的真实案例
    """

    chain_id: str
    name: str
    primitives: list[str] = field(default_factory=list)
    logic: str = ""
    gain: str = ""
    instances: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PrimitiveChain:
        """从 primitive-chains.json 条目构造。"""
        return cls(
            chain_id=str(raw.get("id", "")),
            name=str(raw.get("name", "")),
            primitives=[str(p) for p in raw.get("primitives", [])],
            logic=str(raw.get("logic", "")),
            gain=str(raw.get("gain", "")),
            instances=list(raw.get("instances", []) or []),
        )


@dataclass
class ChainCandidate:
    """链匹配候选：链模板在目标端点上的原语是否齐备。"""

    chain: PrimitiveChain
    matched_endpoints: dict[str, list[str]] = field(default_factory=dict)

    @property
    def chain_id(self) -> str:
        return self.chain.chain_id

    @property
    def name(self) -> str:
        return self.chain.name

    def to_dict(self) -> dict[str, Any]:
        """转为可写回/可展示的 dict。

        含 instance_count：候选链在链库中积累的真实验证案例数，
        供提示注入作为「该链已被实战验证」的证据。
        """
        return {
            "chain_id": self.chain.chain_id,
            "name": self.chain.name,
            "primitives": sorted(self.chain.primitives),
            "logic": self.chain.logic,
            "gain": self.chain.gain,
            "matched_endpoints": self.matched_endpoints,
            "instance_count": len(self.chain.instances),
        }


@dataclass
class AttackSurface:
    """攻击面模式：端点原语判定后按 signals 匹配，注入攻击基元。

    对应 attack_surfaces.json 的 surfaces 条目。
    """

    surface_id: str
    name: str
    signals: list[str] = field(default_factory=list)
    primitives: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    base_primitives: list[str] = field(default_factory=list)
    risk: str = ""
    instances: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AttackSurface:
        """从 attack_surfaces.json 条目构造。"""
        return cls(
            surface_id=str(raw.get("id", "")),
            name=str(raw.get("name", "")),
            signals=[str(s) for s in raw.get("signals", [])],
            primitives=[str(p) for p in raw.get("primitives", [])],
            targets=[str(t) for t in raw.get("targets", [])],
            base_primitives=[str(b) for b in raw.get("base_primitives", [])],
            risk=str(raw.get("risk", "")),
            instances=list(raw.get("instances", []) or []),
        )


@dataclass
class SurfaceMatch:
    """端点与攻击面的匹配结果：命中的攻击面 + 注入的攻击基元。"""

    surface: AttackSurface
    matched_signals: list[str] = field(default_factory=list)
