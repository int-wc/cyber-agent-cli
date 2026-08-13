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
import threading
from datetime import date
from pathlib import Path
from typing import Any

from .models import AttackSurface, PrimitiveEndpoint, SurfaceMatch

# 默认攻击面库路径（相对本模块的 data 目录）
DEFAULT_SURFACES_PATH = Path(__file__).resolve().parent / "data" / "attack_surfaces.json"

# 原语命中权重：端点原语出现在攻击面 primitives 中时额外加分
_PRIMITIVE_HIT_WEIGHT = 2

# 弱判别原语：这类原语是「大类」语义，几乎所有端点都命中（如 query_data=查询），
# 单独命中无判别力，必须配合信号命中才入选，否则攻击面匹配会噪声爆炸。
_WEAK_DISCRIMINATOR_PRIMITIVES = frozenset({"query_data"})

# 单个攻击面的实例上限：超过后丢弃最旧实例，避免库文件无限膨胀
MAX_SURFACE_INSTANCES = 50

# 回写操作使用模块级锁，避免多线程并发写坏库文件
_surfaces_write_lock = threading.RLock()


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
        # 弱判别原语（query_data）单独命中不算入选依据，避免噪声爆炸
        if primitive_hit and ep.attr_key in _WEAK_DISCRIMINATOR_PRIMITIVES:
            primitive_hit = False
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


# ═══ 攻击面库回写（实例积累回路）═══

def _read_surfaces_file(path: Path) -> dict[str, Any] | None:
    """读取攻击面库原始 JSON 结构；文件缺失/损坏返回 None。

    直接操作原始 dict 而非 AttackSurface 模型，是为了保留
    version/updated/description 等顶层元数据不被模型序列化丢弃。
    """
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_surfaces_file(path: Path, data: dict[str, Any]) -> bool:
    """原子写攻击面库文件：先写 .tmp 再 replace，避免崩溃导致文件损坏。

    与 session_store.py 的原子写模式保持一致。
    """
    try:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return True
    except OSError:
        return False


def _surface_instance_dedup_key(instance: dict[str, Any]) -> str:
    """实例去重键：目标 + 端点 + 方法（同一目标同一端点的重复案例只保留一条）。"""
    target = str(instance.get("target", "")).strip()
    endpoint = str(instance.get("endpoint", "")).strip()
    method = str(instance.get("method", "")).strip().upper()
    return f"{target}|{method}|{endpoint}"


def record_surface_instance(
    surface_id: str,
    instance: dict[str, Any],
    path: str | Path | None = None,
) -> bool:
    """往指定攻击面条目的 instances 追加一个真实案例（去重 + 限容）。

    实例结构约定（与 record_chain_instance 一致）：
    {
        "target": "厂商/域名",
        "endpoint": "/api/xxx",
        "method": "GET",
        "verdict": "execute",
        "priority": "high",
        "finding": "发现摘要",
        "date": "2026-08-05",
    }

    攻击面不存在或写入失败返回 False；存在且写入成功返回 True。
    """
    p = Path(path) if path else DEFAULT_SURFACES_PATH
    with _surfaces_write_lock:
        data = _read_surfaces_file(p)
        if data is None:
            return False
        surfaces = data.setdefault("surfaces", [])
        for item in surfaces:
            if str(item.get("id", "")) != surface_id:
                continue
            instances = item.setdefault("instances", [])
            dedup_key = _surface_instance_dedup_key(instance)
            if dedup_key:
                instances[:] = [
                    old for old in instances
                    if _surface_instance_dedup_key(old) != dedup_key
                ]
            instances.append(instance)
            # 限容：保留最新 MAX_SURFACE_INSTANCES 条
            if len(instances) > MAX_SURFACE_INSTANCES:
                del instances[: len(instances) - MAX_SURFACE_INSTANCES]
            data["updated"] = date.today().isoformat()
            return _write_surfaces_file(p, data)
        return False


def upsert_surface(
    new_surface: dict[str, Any],
    path: str | Path | None = None,
) -> tuple[bool, str]:
    """新增或更新一个攻击面条目，返回 (是否新增, surface_id)。

    用于把真实运行中「程序化匹配未覆盖」的新攻击面沉淀为正式条目：
    - 已存在同 id → 保留原 instances，用新内容覆盖其他字段；
    - 不存在 → 以新条目追加（instances 置空）。
    """
    surface_id = str(new_surface.get("id", "")).strip()
    if not surface_id:
        return False, ""
    p = Path(path) if path else DEFAULT_SURFACES_PATH
    with _surfaces_write_lock:
        data = _read_surfaces_file(p)
        if data is None:
            data = {
                "version": 1,
                "updated": date.today().isoformat(),
                "description": "攻击面模式库：可前匹配、可积累。Phase2 判定端点原语后，按 signals 匹配此处攻击面，将其 base_primitives 注入 Phase3；挖洞后在 instances 追加真实案例。跨SRC共享（同 api_patterns.json）。",
                "surfaces": [],
            }
        surfaces = data.setdefault("surfaces", [])
        for item in surfaces:
            if str(item.get("id", "")) == surface_id:
                kept_instances = item.get("instances", [])
                item.clear()
                item.update(new_surface)
                item["instances"] = kept_instances
                data["updated"] = date.today().isoformat()
                _write_surfaces_file(p, data)
                return False, surface_id
        entry = dict(new_surface)
        entry.setdefault("instances", [])
        surfaces.append(entry)
        data["updated"] = date.today().isoformat()
        _write_surfaces_file(p, data)
        return True, surface_id