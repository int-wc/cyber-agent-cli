#!/usr/bin/env python3
"""从执行轨迹沉淀原语链实例（积累回路离线批量回写）。

扫描 ~/.cyber-agent-cli-traces/*.trace.json（或 --traces-dir），对每条
含 primitive_verdict 的真实运行：

- 从 pipeline_start 提取目标摘要（厂商/域名），作为实例 target；
- 解析 verdict JSON 中的 top_chains，取 verdict=execute 的链；
- 链已在正式链库 → record_chain_instance 追加实例（去重+限容）；
- 链不在正式链库（模型自创）→ 打印到候选报告，不自动 upsert，
  由人工确认后调用 upsert_chain 或直接编辑链库合并；
- 输出沉淀统计：新增实例数、候选链清单。

用法:
  python3 scripts/sediment_primitive_instances.py            # 沉淀全部 trace
  python3 scripts/sediment_primitive_instances.py --dry-run  # 只报告不写入
  python3 scripts/sediment_primitive_instances.py --traces-dir /path/to/traces
  python3 scripts/sediment_primitive_instances.py --chain-id ch_ssrf_to_auth
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# 允许以源码方式直接运行（未 pip install -e）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cyber_agent.agent.primitives import (  # noqa: E402
    DEFAULT_CHAINS_PATH,
    load_chain_ids,
    promote_chain_candidates,
    record_chain_instance,
)

# 默认轨迹目录（与 pipeline.py _save_trace 一致）
DEFAULT_TRACES_DIR = Path.home() / ".cyber-agent-cli-traces"

# verdict detail 的 JSON 围栏：```json ... ``` 或裸 JSON
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

# 目标摘要提取：优先取【厂商】或域名字样
_TARGET_RE = re.compile(
    r"[【\[]([^】\]]{2,40})[】\]]"
    r"|([a-zA-Z0-9][-a-zA-Z0-9]*\.[-a-zA-Z0-9.]+)"
)

# 测试占位目标：真实挖洞的目标不会是这些字样（fake LLM 测试/实验运行）
_PLACEHOLDER_TARGETS = ("目标A", "目标B", "测试", "xxx", "example", "example.com")


def _strip_fence(text: str) -> str:
    """去掉 ```json 围栏，返回纯 JSON 文本。"""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _parse_verdict(detail: str) -> list[dict]:
    """解析 primitive_verdict 事件 detail，返回 top_chains 列表。"""
    raw = _strip_fence(detail)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 宽松兜底：找 top_chains 数组
        m = re.search(r'"top_chains"\s*:\s*(\[.*\])', raw, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(1))
            return [t for t in data if isinstance(t, dict)]
        except json.JSONDecodeError:
            return []
    if not isinstance(data, dict):
        return []
    chains = data.get("top_chains", [])
    return [t for t in chains if isinstance(t, dict)]


def _extract_target(start_detail: str) -> str:
    """从 pipeline_start detail 提取目标标识（厂商/域名）。"""
    if not start_detail:
        return ""
    m = _TARGET_RE.search(start_detail)
    if m:
        return m.group(1) or m.group(2) or ""
    return start_detail[:40].strip()


def _extract_evidence_commands(trace_events: list[dict]) -> list[str]:
    """从 tool_call 事件提取 curl 探测命令，作为实例证据线索。"""
    commands: list[str] = []
    for e in trace_events:
        if e.get("event") != "tool_call":
            continue
        meta = e.get("metadata") or {}
        args = meta.get("args") or {}
        cmd = str(args.get("command", "")) or str(e.get("detail", ""))
        if cmd.strip():
            commands.append(cmd.strip())
    return commands[:20]


def sediment_trace_file(
    trace_path: Path,
    *,
    known_ids: set[str],
    dry_run: bool = False,
    only_chain: str = "",
) -> tuple[int, list[dict], list[dict]]:
    """沉淀单个 trace 文件，返回 (新增实例数, 沉淀链列表, 候选链列表)。"""
    try:
        events = json.loads(trace_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ⚠️ 跳过 {trace_path.name}: {exc}")
        return 0, [], []
    if not isinstance(events, list):
        return 0, [], []

    start_detail = ""
    verdict_detail = ""
    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get("event") == "pipeline_start":
            start_detail = str(e.get("detail", ""))
        elif e.get("event") == "primitive_verdict":
            verdict_detail = str(e.get("detail", ""))
    if not verdict_detail:
        return 0, [], []

    target = _extract_target(start_detail)
    top_chains = _parse_verdict(verdict_detail)
    evidence_commands = _extract_evidence_commands(events)

    # 过滤无实际执行的运行：只有执行过工具（tool_call）的 trace 才算验证，
    # 纯思考/测试运行（fake LLM）不沉淀为「验证实例」，避免污染链库。
    has_execution = any(
        isinstance(e, dict) and e.get("event") == "tool_call" for e in events
    )
    if not has_execution:
        return 0, [], []

    # 过滤测试占位目标
    if any(p in target for p in _PLACEHOLDER_TARGETS):
        return 0, [], []

    sedimented: list[dict] = []
    candidates: list[dict] = []
    added = 0
    for item in top_chains:
        if str(item.get("verdict", "")).strip().lower() != "execute":
            continue
        chain_id = str(item.get("chain_id", "")).strip()
        if not chain_id:
            continue
        if only_chain and chain_id != only_chain:
            continue
        plan = str(item.get("exploitation_plan", ""))[:300]
        finding = plan or "; ".join(evidence_commands)[:300]
        instance = {
            "target": target,
            "endpoint": str(item.get("key_endpoints") or "")[:200],
            "method": "",
            "verdict": "execute",
            "priority": str(item.get("priority", "medium")),
            "finding": finding,
            "date": trace_path.name[:8],
            "source": trace_path.name,
        }
        if chain_id in known_ids:
            if not dry_run:
                ok = record_chain_instance(chain_id, instance)
            else:
                ok = True
            if ok:
                added += 1
                sedimented.append({"chain_id": chain_id, "target": target, "priority": instance["priority"]})
        else:
            candidates.append({
                "chain_id": chain_id,
                "name": chain_id,
                "target": target,
                "priority": instance["priority"],
                "plan": plan,
                "source": trace_path.name,
            })
    return added, sedimented, candidates


def main() -> int:
    ap = argparse.ArgumentParser(description="从执行轨迹沉淀原语链实例")
    ap.add_argument("--traces-dir", default=str(DEFAULT_TRACES_DIR), help="轨迹目录，默认 ~/.cyber-agent-cli-traces")
    ap.add_argument("--dry-run", action="store_true", help="只报告不写入链库")
    ap.add_argument("--chain-id", default="", help="只沉淀指定链 id")
    ap.add_argument("--promote", action="store_true", help="沉淀后把频次足够的候选链自动提炼入库")
    ap.add_argument("--min-seen", type=int, default=2, help="候选链提炼的最低出现频次，默认 2")
    args = ap.parse_args()

    traces_dir = Path(args.traces_dir).expanduser()
    if not traces_dir.is_dir():
        print(f"❌ 轨迹目录不存在: {traces_dir}")
        return 1

    trace_files = sorted(traces_dir.glob("*.trace.json"))
    if not trace_files:
        print(f"❌ 轨迹目录没有 .trace.json: {traces_dir}")
        return 1

    known_ids = load_chain_ids()
    print(f"🧬 原语链实例沉淀（{'dry-run' if args.dry_run else '写入链库'}）")
    print(f"   轨迹目录: {traces_dir}  ({len(trace_files)} 个文件)")
    print(f"   正式链库: {DEFAULT_CHAINS_PATH}  ({len(known_ids)} 条链)")
    print()

    total_added = 0
    all_sedimented: list[dict] = []
    all_candidates: list[dict] = []
    processed = 0
    for tf in trace_files:
        added, sedimented, candidates = sediment_trace_file(
            tf,
            known_ids=known_ids,
            dry_run=args.dry_run,
            only_chain=args.chain_id,
        )
        if added or candidates:
            processed += 1
            total_added += added
            all_sedimented.extend(sedimented)
            all_candidates.extend(candidates)
            print(f"  📄 {tf.name}: 沉淀 {added} 条，候选 {len(candidates)} 条")

    print()
    print(f"✅ 完成：处理 {processed} 个含裁决的 trace，沉淀 {total_added} 条实例")
    if all_sedimented:
        print("\n📌 已沉淀实例:")
        for s in all_sedimented:
            print(f"   - {s['chain_id']} [{s['priority']}] target={s['target'][:50]}")
    if all_candidates:
        print("\n🔶 候选链（不在正式链库，模型自创）:")
        seen: set[str] = set()
        for c in all_candidates:
            if c["chain_id"] in seen:
                continue
            seen.add(c["chain_id"])
            print(f"   - {c['chain_id']} [{c['priority']}] target={c['target'][:50]}")
            print(f"     plan: {c['plan'][:120]}")
        if not args.promote:
            print("\n   用 --promote 自动提炼频次足够的候选链入库。")

    # 候选链自动提炼入库（频次足够、有原语信息的通用链型）
    if args.promote:
        if args.dry_run:
            print("\n⚠️ --promote 与 --dry-run 冲突：dry-run 不执行提炼。")
        else:
            promoted, skipped = promote_chain_candidates(min_seen=args.min_seen)
            if promoted:
                print(f"\n✅ 自动提炼 {len(promoted)} 条候选链入库:")
                for cid in promoted:
                    print(f"   - {cid}")
            if skipped:
                print(f"\n⏳ 跳过 {len(skipped)} 条候选（频次不足/无原语信息/已存在，保留在候选区）:")
                for cid in skipped:
                    print(f"   - {cid}")
            if not promoted and not skipped:
                print("\nℹ️ 候选区为空，无可提炼。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
