"""原语链库：加载 primitive-chains.json，把已判定原语的端点清单与链模板匹配。

移植自 BSRC_SKILLS_V1 的 chain_linking.py 联动推理逻辑：
- 单个业务原语(能力点)单独可能无法构成有效危害；
- 若链模板的组成原语在目标端点上都齐备，则存在「业务信任串联→有效危害」的可能；
- 候选链交给后续阶段用只读方式验证实际可串联性。

链库文件：src/cyber_agent/agent/primitives/data/primitive-chains.json
"""

from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path
from typing import Any

from .models import ChainCandidate, PrimitiveChain, PrimitiveEndpoint

# 默认链库路径（相对本模块的 data 目录）
DEFAULT_CHAINS_PATH = Path(__file__).resolve().parent / "data" / "primitive-chains.json"

# 链候选文件：管线自动沉淀的「模型自创链」先落此处，
# 由人工/沉淀脚本确认后再 upsert 进正式链库，避免垃圾链污染。
DEFAULT_CHAIN_CANDIDATES_PATH = (
    Path(__file__).resolve().parent / "data" / "primitive-chains.candidates.json"
)

# 单条链的实例上限：超过后丢弃最旧实例，避免链库无限膨胀
MAX_CHAIN_INSTANCES = 50

# 回写操作使用模块级锁，避免多线程并发写坏链库文件
_chains_write_lock = threading.RLock()


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


# ═══ 链库回写（实例积累回路）═══

def _read_chains_file(path: Path) -> dict[str, Any] | None:
    """读取链库原始 JSON 结构；文件缺失/损坏返回 None。

    直接操作原始 dict 而非 PrimitiveChain 模型，是为了保留
    version/updated/description 等顶层元数据不被模型序列化丢弃。
    """
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_chains_file(path: Path, data: dict[str, Any]) -> bool:
    """原子写链库文件：先写 .tmp 再 replace，避免崩溃导致文件损坏。

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


def _chain_instance_dedup_key(instance: dict[str, Any]) -> str:
    """实例去重键：目标 + 端点 + 方法（同一目标同一端点的重复案例只保留一条）。"""
    target = str(instance.get("target", "")).strip()
    endpoint = str(instance.get("endpoint", "")).strip()
    method = str(instance.get("method", "")).strip().upper()
    return f"{target}|{method}|{endpoint}"


def record_chain_instance(
    chain_id: str,
    instance: dict[str, Any],
    path: str | Path | None = None,
) -> bool:
    """往指定链模板的 instances 追加一个真实案例（去重 + 限容）。

    实例结构约定（由管线/沉淀脚本提供）：
    {
        "target": "厂商/域名",
        "endpoint": "/api/xxx",
        "method": "GET",
        "verdict": "execute",      # 链裁决结论
        "priority": "high",        # 优先级
        "finding": "发现摘要",      # 验证结果
        "date": "2026-08-05",      # 发生日期
    }

    链不存在或写入失败返回 False；链存在且写入成功返回 True。
    """
    p = Path(path) if path else DEFAULT_CHAINS_PATH
    with _chains_write_lock:
        data = _read_chains_file(p)
        if data is None:
            return False
        chains = data.setdefault("chains", [])
        for item in chains:
            if str(item.get("id", "")) != chain_id:
                continue
            instances = item.setdefault("instances", [])
            dedup_key = _chain_instance_dedup_key(instance)
            if dedup_key:
                instances[:] = [
                    old for old in instances
                    if _chain_instance_dedup_key(old) != dedup_key
                ]
            instances.append(instance)
            # 限容：保留最新 MAX_CHAIN_INSTANCES 条
            if len(instances) > MAX_CHAIN_INSTANCES:
                del instances[: len(instances) - MAX_CHAIN_INSTANCES]
            data["updated"] = date.today().isoformat()
            return _write_chains_file(p, data)
        return False


def upsert_chain(
    new_chain: dict[str, Any],
    path: str | Path | None = None,
) -> tuple[bool, str]:
    """新增或更新一条链模板，返回 (是否新增, chain_id)。

    用于把模型在真实运行中「自创」的链沉淀为正式模板：
    - 链库已存在同 id → 保留原 instances，用新内容覆盖其他字段；
    - 链库不存在 → 以新条目追加（instances 置空）。
    """
    chain_id = str(new_chain.get("id", "")).strip()
    if not chain_id:
        return False, ""
    p = Path(path) if path else DEFAULT_CHAINS_PATH
    with _chains_write_lock:
        data = _read_chains_file(p)
        if data is None:
            data = {
                "version": 1,
                "updated": date.today().isoformat(),
                "description": "业务原语链库：单个业务原语(能力点)单独可能无法构成有效危害，多个原语通过业务信任串联成可利用链达成有效漏洞。Phase3 联动推理 + Phase4 验证 + 实例积累，跨SRC共享。",
                "chains": [],
            }
        chains = data.setdefault("chains", [])
        for item in chains:
            if str(item.get("id", "")) == chain_id:
                kept_instances = item.get("instances", [])
                item.clear()
                item.update(new_chain)
                item["instances"] = kept_instances
                data["updated"] = date.today().isoformat()
                _write_chains_file(p, data)
                return False, chain_id
        entry = dict(new_chain)
        entry.setdefault("instances", [])
        chains.append(entry)
        data["updated"] = date.today().isoformat()
        _write_chains_file(p, data)
        return True, chain_id


def load_chain_candidates(path: str | Path | None = None) -> list[dict[str, Any]]:
    """读取链候选文件，返回候选链 dict 列表。文件缺失/损坏返回空列表。"""
    p = Path(path) if path else DEFAULT_CHAIN_CANDIDATES_PATH
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    items = raw.get("candidates", []) if isinstance(raw, dict) else []
    return [item for item in items if isinstance(item, dict)]


def append_chain_candidate(
    new_chain: dict[str, Any],
    path: str | Path | None = None,
) -> bool:
    """把模型自创的链追加到链候选文件。

    同 id 候选再次出现时（不同目标的真实运行），合并实例并累计
    seen_count（promote 提炼的频次依据），而不是丢弃。返回是否追加成功。
    """
    chain_id = str(new_chain.get("id", "")).strip()
    if not chain_id:
        return False
    p = Path(path) if path else DEFAULT_CHAIN_CANDIDATES_PATH
    with _chains_write_lock:
        data = {}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
        if not isinstance(data, dict):
            data = {}
        candidates = data.setdefault("candidates", [])
        new_instances = list(new_chain.get("instances", []) or [])
        for item in candidates:
            if str(item.get("id", "")) == chain_id:
                # 已有候选：合并实例（去重）+ 累计频次
                existing = item.setdefault("instances", [])
                existing_keys = {
                    f"{i.get('target','')}|{i.get('method','')}|{i.get('endpoint','')}"
                    for i in existing
                }
                for inst in new_instances:
                    key = f"{inst.get('target','')}|{inst.get('method','')}|{inst.get('endpoint','')}"
                    if key and key not in existing_keys:
                        existing.append(inst)
                        existing_keys.add(key)
                item["seen_count"] = int(item.get("seen_count", 1)) + 1
                data["updated"] = date.today().isoformat()
                return _write_chains_file(p, data)
        entry = dict(new_chain)
        entry.setdefault("instances", [])
        entry.setdefault("first_seen", date.today().isoformat())
        entry["seen_count"] = 1
        candidates.append(entry)
        data["updated"] = date.today().isoformat()
        return _write_chains_file(p, data)


def promote_chain_candidates(
    chains_path: str | Path | None = None,
    candidates_path: str | Path | None = None,
    *,
    min_seen: int = 2,
) -> tuple[list[str], list[str]]:
    """把出现频次足够的候选链提炼进正式链库，返回 (已入库 id 列表, 跳过 id 列表)。

    候选链在多个目标的真实运行中反复出现（seen_count ≥ min_seen），
    说明它是可复用的通用链型，自动 upsert 进正式链库；候选链的实例
    一并迁移（实例是跨目标验证证据）。已存在同 id 的候选跳过。

    - min_seen=2：至少 2 次独立运行出现才提炼，避免单次噪声入库。
    - 提炼后候选条目从候选文件移除。
    """
    cp = Path(candidates_path) if candidates_path else DEFAULT_CHAIN_CANDIDATES_PATH
    chains_p = Path(chains_path) if chains_path else DEFAULT_CHAINS_PATH
    promoted: list[str] = []
    skipped: list[str] = []
    if not cp.exists():
        return promoted, skipped
    try:
        cand_data = json.loads(cp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return promoted, skipped
    if not isinstance(cand_data, dict):
        return promoted, skipped
    candidates = cand_data.get("candidates", [])
    if not candidates:
        return promoted, skipped

    existing_ids = load_chain_ids(chains_p)
    remaining: list[dict[str, Any]] = []
    with _chains_write_lock:
        for item in candidates:
            if not isinstance(item, dict):
                continue
            chain_id = str(item.get("id", "")).strip()
            if not chain_id:
                continue
            seen = int(item.get("seen_count", 1))
            if seen < min_seen or chain_id in existing_ids:
                # 频次不足或已入库：保留在候选区，等待继续积累
                skipped.append(chain_id)
                remaining.append(item)
                continue
            # 提炼：实例迁移进正式链库（保留 target/endpoint/finding 证据）
            entry = {
                "id": chain_id,
                "name": str(item.get("name", chain_id)),
                "primitives": [str(p) for p in item.get("primitives", []) if p],
                "logic": str(item.get("logic", ""))[:500],
                "gain": str(item.get("gain", "候选链自动提炼"))[:300],
                "instances": list(item.get("instances", []) or []),
            }
            if not entry["primitives"]:
                # 无原语信息时无法自动归入链库分类，保留候选等待人工
                skipped.append(chain_id)
                remaining.append(item)
                continue
            added, _ = upsert_chain(entry, chains_p)
            if added:
                promoted.append(chain_id)
            else:
                skipped.append(chain_id)
                remaining.append(item)
        # 写回候选文件：移除已提炼条目
        cand_data["candidates"] = remaining
        cand_data["updated"] = date.today().isoformat()
        _write_chains_file(cp, cand_data)
    return promoted, skipped
