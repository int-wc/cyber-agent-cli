"""攻击面前匹配：把已判定原语的端点清单与攻击面库匹配，注入攻击基元。

移植自 BSRC_SKILLS_V1 的 attack_surface_match.py：
- 对每个端点，取「endpoint + business_attr + params」拼成 hay，与攻击面 signals 做子串匹配；
- 加分项：端点原语命中攻击面 primitives（prim=2），匹配信号数做排序权重；
- 命中即把该攻击面的 base_primitives（攻击基元）注入 Phase3；
- 未命中的端点交由 agent 对照攻击面库自行补充。

攻击面库文件：src/cyber_agent/agent/primitives/data/attack_surfaces.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import AttackSurface, PrimitiveEndpoint, SurfaceMatch

# 默认攻击面库路径（相对本模块的 data 目录）
DEFAULT_SURFACES_PATH = Path(__file__).resolve().parent / "data" / "attack_surfaces.json"

# 原语命中权重：端点原语出现在攻击面 primitives 中时额外加分
_PRIMITIVE_HIT_WEIGHT = 2


def load_surfaces(path: str | Path | None = None) -> list[AttackSurface]:
    """加载攻击面库，返回攻击面列表。文件缺失/损坏返回空列表。"""
    p = Path(path) if path else DEFAULT_SURFACES_PATH
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [AttackSurface.from_dict(item) for item in raw.get("surfaces", []) if item]


def _endpoint_hay(ep: PrimitiveEndpoint) -> str:
    """构造端点的匹配文本：endpoint + 原语 + 参数键。"""
    keys = list(ep.params.keys()) if isinstance(ep.params, dict) else []
    parts = [ep.endpoint, ep.attr_key, " ".join(map(str, keys))]
    return " ".join(parts).lower()


def match_surfaces(
    ep: PrimitiveEndpoint,
    surfaces: list[AttackSurface] | None = None,
) -> list[SurfaceMatch]:
    """匹配单个端点到攻击面库，按 (原语命中, 信号数) 降序返回。

    原语命中（端点原语 ∈ 攻击面 primitives）权重高于信号数量。
    """
    surf_list = surfaces if surfaces is not None else load_surfaces()
    hay = _endpoint_hay(ep)
    scored: list[tuple[int, int, SurfaceMatch]] = []
    for surf in surf_list:
        matched_signals = [s for s in surf.signals if str(s).lower() in hay]
        primitive_hit = ep.attr_key in surf.primitives
        score = len(matched_signals) + (_PRIMITIVE_HIT_WEIGHT if primitive_hit else 0)
        if score <= 0:
            continue
        scored.append((
            1 if primitive_hit else 0,  # 原语命中优先
            len(matched_signals),        # 其次信号数量
            SurfaceMatch(surface=surf, matched_signals=matched_signals),
        ))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [m for _, _, m in scored]


def build_hint_report(
    eps: list[PrimitiveEndpoint],
    surfaces: list[AttackSurface] | None = None,
) -> dict[str, Any]:
    """生成攻击面匹配报告：命中 + 未命中，供日志/追踪使用。"""
    surf_list = surfaces if surfaces is not None else load_surfaces()
    hints: list[dict[str, Any]] = []
    matched_count = 0
    for ep in eps:
        matches = match_surfaces(ep, surf_list)
        item = {
            "endpoint": ep.endpoint,
            "method": ep.method,
            "business_attr": ep.attr_key,
            "surfaces": [
                {
                    "id": m.surface.surface_id,
                    "name": m.surface.name,
                    "base_primitives": m.surface.base_primitives,
                    "risk": m.surface.risk,
                    "matched_signals": m.matched_signals,
                }
                for m in matches
            ],
        }
        hints.append(item)
        if matches:
            matched_count += 1
    return {
        "total_endpoints": len(eps),
        "matched_count": matched_count,
        "hints": hints,
    }