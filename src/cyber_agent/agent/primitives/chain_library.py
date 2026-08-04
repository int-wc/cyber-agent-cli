"""原语链库：加载 primitive-chains.json，把已判定原语的端点清单与链模板匹配。

移植自 BSRC_SKILLS_V1 的 chain_linking.py 联动推理逻辑：
- 单个业务原语(能力点)单独可能无法构成有效危害；
- 若链模板的组成原语在目标端点上都齐备，则存在「业务信任串联→有效危害」的可能；
- 候选链交给后续阶段用只读方式验证实际可串联性。

链库文件：src/cyber_agent/agent/primitives/data/primitive-chains.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ChainCandidate, PrimitiveChain, PrimitiveEndpoint

# 默认链库路径（相对本模块的 data 目录）
DEFAULT_CHAINS_PATH = Path(__file__).resolve().parent / "data" / "primitive-chains.json"


def load_chains(path: str | Path | None = None) -> list[PrimitiveChain]:
    """加载原语链库，返回链模板列表。文件缺失/损坏返回空列表。"""
    p = Path(path) if path else DEFAULT_CHAINS_PATH
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [PrimitiveChain.from_dict(item) for item in raw.get("chains", []) if item]


def load_chain_ids(path: str | Path | None = None) -> set[str]:
    """返回链库中所有链模板 id 集合（供实例回写去重判断）。"""
    return {ch.chain_id for ch in load_chains(path)}


def _attr_to_endpoints(eps: Iterable) -> dict[str, list[str]]:
    """按原语 key 分组去重端点列表。"""
    attr2eps: dict[str, list[str]] = {}
    for ep in eps:
        attr = getattr(ep, "attr_key", None) or ""
        endpoint = getattr(ep, "endpoint", "")
        if not attr or not endpoint:
            continue
        if attr not in attr2eps:
            attr2eps[attr] = []
        if endpoint not in attr2eps[attr]:
            attr2eps[attr].append(endpoint)
    return attr2eps


def link(eps: Iterable[PrimitiveEndpoint], chains: Iterable[PrimitiveChain]) -> list[ChainCandidate]:
    """把端点清单与链模板匹配，返回组成原语齐备的候选链。

    每条候选链记录链模板 + 每条组成原语对应的目标端点（取前 3 个），
    供下游阶段对链的组成端点做只读可串联性验证。
    """
    attr2eps = _attr_to_endpoints(eps)
    available = set(attr2eps.keys())
    candidates: list[ChainCandidate] = []
    for ch in chains:
        need = {p for p in ch.primitives if p}
        if not need or not need.issubset(available):
            continue
        matched = {a: attr2eps[a][:3] for a in sorted(need)}
        candidates.append(ChainCandidate(chain=ch, matched_endpoints=matched))
    return candidates


def build_link_report(
    eps: Iterable[PrimitiveEndpoint],
    chains: Iterable[PrimitiveChain] | None = None,
) -> dict[str, Any]:
    """生成链联动推理报告（含端点数 + 候选链），供日志/追踪使用。"""
    eps_list = list(eps)
    chain_list = list(chains) if chains is not None else load_chains()
    candidates = link(eps_list, chain_list)
    return {
        "total_endpoints": len(eps_list),
        "candidates": [c.to_dict() for c in candidates],
    }
