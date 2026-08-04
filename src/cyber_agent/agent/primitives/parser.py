"""业务原语解析器：从分析文本/端点清单中解析 business_attr 判定。

移植自 BSRC_SKILLS_V1：
- chain_linking.py 的 EP_LINE 正则（Phase2 原语判定行的解析）
- workflow_runner.js 的 BUSINESS_ATTR_GUIDE（原语判定的思维指导）

解析两种输入：
1. 分析文本中的原语判定行：
   `- /xxx POST business_attr=transfer attr_target=remote_url attr_reason=... params={...} risk=...`
2. 结构化的端点 dict 列表（endpoint/method/business_attr/...）
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from .models import (
    AttrTarget,
    BusinessPrimitive,
    PrimitiveEndpoint,
)

# Phase2 原语判定行格式：- /xxx POST business_attr=transfer ...
EP_LINE = re.compile(
    r"-\s+(\S+)\s+(GET|POST|PUT|DELETE|PATCH|HEAD)\s+business_attr=(\w+)",
    re.I,
)

# attr_reason / params / risk 可选字段的宽松提取
_FIELD_LINE = re.compile(
    r"attr_target=(\w+)"
    r"|attr_reason=([^\s]+(?: [^\s]+)*?)(?=\s+(?:risk|params)=|\s*$)"
    r"|risk=([^\s]+(?: [^\s]+)*?)(?=\s+business_attr=|\s*$)",
    re.I,
)

# 一行的完整原语判定（含 attr_target 与 risk），供整行解析
_FULL_ATTR_LINE = re.compile(
    r"business_attr=(\w+).*?attr_target=(\w+)",
    re.I,
)


# BUSINESS_ATTR_GUIDE —— 原语判定思维指导（供角色 prompt 注入）
BUSINESS_ATTR_GUIDE = """判定核心业务原语 business_attr（端点"到底对什么东西做什么操作"），再选攻击基元，不要只看API名字或参数名。

- 每个端点输出: `- /xxx POST business_attr=<原语> attr_target=<对象> attr_reason=<依据> params={...} risk=<风险>`
- business_attr ∈ {read_file, write_file, exec_code, modify_state, query_data, transfer, auth}
- attr_target（原语作用对象）∈ {local_fs, remote_url, db, template, user_input, worker}

按原语选攻击基元（不按名字）:
- read_file → 任意文件读取/路径遍历
- write_file → 上传/任意写入/路径穿越
- exec_code → RCE/表达式注入/SSTI
- modify_state → 逻辑缺陷/越权增删改/审批绕过
- query_data → IDOR/未授权查询
- transfer → SSRF（含 remote→local 原语切换: 远程URL被白名单拦时，改让服务端读本地路径——读本地不是URL fetch，白名单看不见）；开放重定向
- auth → 认证绕过/JWT伪造/弱口令

强制规则:
1. 名字含 load/parse/import/sync/render/convert/transform/download/upload 等"数据处理"语义的端点，必须强制判定真实原语——同一表面常同时是 read 原语与 exec 原语。
2. 每个端点至少自问三连: "如果我让它读本地文件 / 执行我给的代码 / 写入任意路径，业务上它会不会照做？" → 对应 read/exec/write 三个原语各测一遍。"""


def parse_line(line: str) -> PrimitiveEndpoint | None:
    """解析一行原语判定文本，返回端点对象；无法解析返回 None。"""
    if not line:
        return None
    m = EP_LINE.search(line)
    if not m:
        return None
    endpoint = m.group(1)
    method = m.group(2).upper()
    attr_name = m.group(3)

    attr = BusinessPrimitive.parse(attr_name)
    # 匹配 attr_target
    target = None
    tm = _FULL_ATTR_LINE.search(line)
    if tm:
        target = AttrTarget.parse(tm.group(2))

    # 宽松提取 attr_reason / risk / params
    reason, risk = _extract_reason_risk(line)
    params = _extract_params(line)

    return PrimitiveEndpoint(
        endpoint=endpoint,
        method=method,
        business_attr=attr,
        attr_target=target,
        attr_reason=reason,
        params=params,
        risk=risk,
    )


def _extract_reason_risk(line: str) -> tuple[str, str]:
    """从一行中提取 attr_reason 与 risk 字段（尽力而为，失败返回空）。"""
    reason = ""
    risk = ""
    # 尝试 JSON 风格的键值：attr_reason=xxx risk=yyy
    reason_m = re.search(r"attr_reason=([^\\n\\r]*?)(?=\s+(?:risk|params)=|\s*$)", line, re.I)
    if reason_m:
        reason = reason_m.group(1).strip().strip('"')
    risk_m = re.search(r"risk=([^\\n\\r]*?)(?=\s+(?:business_attr|attr_reason|params)=|\s*$)", line, re.I)
    if risk_m:
        risk = risk_m.group(1).strip().strip('"')
    return reason, risk


def _extract_params(line: str) -> dict[str, Any]:
    """从一行中提取 params={...}（JSON），失败返回空 dict。"""
    m = re.search(r"params=\{(.*?)\}", line, re.I)
    if not m:
        return {}
    try:
        raw = json.loads("{" + m.group(1) + "}")
        return raw if isinstance(raw, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def parse_text(text: str) -> list[PrimitiveEndpoint]:
    """解析整段分析文本，返回所有原语判定行对应的端点。"""
    eps = []
    for line in text.splitlines():
        ep = parse_line(line)
        if ep is not None:
            eps.append(ep)
    return eps


def parse_endpoint_dicts(items: Iterable[dict[str, Any]]) -> list[PrimitiveEndpoint]:
    """从结构化端点 dict 列表构造端点对象（原语判定已由上游给出）。"""
    eps = []
    for item in items:
        endpoint = str(item.get("endpoint", ""))
        if not endpoint:
            continue
        eps.append(
            PrimitiveEndpoint(
                endpoint=endpoint,
                method=str(item.get("method", "GET")).upper(),
                business_attr=BusinessPrimitive.parse(str(item.get("business_attr", ""))),
                attr_target=AttrTarget.parse(str(item.get("attr_target", ""))),
                attr_reason=str(item.get("attr_reason", "")),
                params=dict(item.get("params", {}) or {}),
                risk=str(item.get("risk", "")),
            )
        )
    return eps


def endpoints_to_line(ep: PrimitiveEndpoint) -> str:
    """把一个端点对象序列化回原语判定行（供后续文本传递/回写匹配）。"""
    attr_name = ep.attr_key
    target = ep.attr_target.value if ep.attr_target else ""
    parts = [f"- {ep.endpoint} {ep.method} business_attr={attr_name}"]
    if target:
        parts.append(f"attr_target={target}")
    if ep.attr_reason:
        parts.append(f"attr_reason={ep.attr_reason}")
    if ep.params:
        parts.append(f"params={json.dumps(ep.params, ensure_ascii=False)}")
    if ep.risk:
        parts.append(f"risk={ep.risk}")
    return " ".join(parts)


def serialize_endpoints(eps: Iterable[PrimitiveEndpoint]) -> str:
    """把端点列表序列化为连续的原语判定文本（跨阶段传递用）。"""
    return "\n".join(endpoints_to_line(ep) for ep in eps)