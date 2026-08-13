"""业务原语解析与原语链利用核心库。

从 BSRC_SKILLS_V1 的 workflow 原语方法论移植到 Python：
- models.py          业务原语/作用对象/端点/链/攻击面的数据模型
- parser.py          business_attr 判定行解析 + 原语判定思维指导
- chain_library.py   原语链库加载 + 端点→链模板匹配（原语链联动推理）
- surface_matcher.py 攻击面前匹配（signals + 原语命中 → 攻击基元注入）

供四柱管线的原语化改造（ANALYST→原语解析、DIFFUSER→攻击面扩散、
JUMPER→链跃迁、REFLECTOR→链裁决）与独立的链利用工作流复用。
"""

from __future__ import annotations

from .models import (
    AttackSurface,
    AttrTarget,
    BusinessPrimitive,
    ChainCandidate,
    PrimitiveChain,
    PrimitiveEndpoint,
    SurfaceMatch,
)
from .parser import (
    BUSINESS_ATTR_GUIDE,
    parse_line,
    parse_text,
    parse_endpoint_dicts,
    serialize_endpoints,
    endpoints_to_line,
)
from .chain_library import (
    DEFAULT_CHAINS_PATH,
    MAX_CHAIN_INSTANCES,
    load_chains,
    load_chain_ids,
    link,
    build_link_report,
    record_chain_instance,
    upsert_chain,
)
from .surface_matcher import (
    DEFAULT_SURFACES_PATH,
    MAX_SURFACE_INSTANCES,
    load_surfaces,
    match_surfaces,
    build_hint_report,
    record_surface_instance,
    upsert_surface,
)

__all__ = [
    "AttackSurface",
    "AttrTarget",
    "BusinessPrimitive",
    "ChainCandidate",
    "PrimitiveChain",
    "PrimitiveEndpoint",
    "SurfaceMatch",
    "BUSINESS_ATTR_GUIDE",
    "parse_line",
    "parse_text",
    "parse_endpoint_dicts",
    "serialize_endpoints",
    "endpoints_to_line",
    "DEFAULT_CHAINS_PATH",
    "MAX_CHAIN_INSTANCES",
    "load_chains",
    "load_chain_ids",
    "link",
    "build_link_report",
    "record_chain_instance",
    "upsert_chain",
    "DEFAULT_SURFACES_PATH",
    "MAX_SURFACE_INSTANCES",
    "load_surfaces",
    "match_surfaces",
    "build_hint_report",
    "record_surface_instance",
    "upsert_surface",
]
