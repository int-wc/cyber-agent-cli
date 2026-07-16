"""Benchmark 运行态、自动提交、移交和终止逻辑。"""
from __future__ import annotations

import json
import re as _re_mod
import subprocess
from typing import Any

from .pipeline_constants import BENCHMARK_REDUNDANT_BLOCK_LIMIT


class BenchmarkRuntimeMixin:
    """Benchmark 运行态维护与移交控制能力。"""

    def _benchmark_profiled_service_fingerprints(self) -> set[str]:
        return (
            set(self._benchmark_service_action_profiles())
            | set(self._benchmark_service_handoff_profiles())
        )

    def _benchmark_fingerprint_has_profiled_handoff(self, fingerprint: Any) -> bool:
        return (
            isinstance(fingerprint, str)
            and fingerprint in self._benchmark_profiled_service_fingerprints()
        )

    def _benchmark_run_service_action_step(
        self,
        fingerprint: str,
        action: str,
    ) -> str:
        profile = self._benchmark_service_action_profiles()[fingerprint]
        label = str(profile.get("label") or fingerprint)
        code, addrs = self._benchmark_active_challenge_from_state()
        if not code:
            verb = action if action != "handoff" else "handoff"
            return f"确定性 {label} {verb}：当前没有 active 容器，跳过。"
        action_profile = dict((profile.get("actions") or {}).get(action) or {})
        if action == "close":
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            closed = self._benchmark_close_local(code)
            if code in completed:
                return f"确定性 {label} close：{code} 已完成，close 返回: {closed[:200]}"
            return f"确定性 {label} close：{code} 已 close 释放资源: {closed[:300]}"

        if not addrs:
            return f"确定性 {label} {action}：当前 active 容器无地址，跳过。"
        addr = addrs[0]
        if not _re_mod.fullmatch(r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}", addr):
            return f"确定性 {label} {action}：容器地址格式异常: {addr}"
        base = f"http://{addr}/"
        self._benchmark_set_service_fingerprint(code, fingerprint)
        probe_func = profile.get("probe")
        if callable(probe_func):
            probe = str(probe_func(code, base, ""))
        else:
            probe = self._benchmark_probe_handoff_followup_local(code, addrs)
        with self._benchmark_state_lock:
            completed = set(self._benchmark_state.get("completed_challenges", set()))
        if code in completed:
            closed = self._benchmark_close_local(code)
            return (
                f"确定性 {label} {action}：{code} 已提交成功并关闭。\n"
                f"close 返回: {closed[:200]}\n探测摘要:\n{probe[:3000]}"
            )
        if action == "handoff":
            self._benchmark_mark_reasoning_needed(
                code,
                str(action_profile.get("reasoning_reason") or f"{label} handoff 需要服务专项深挖"),
            )
        else:
            self._benchmark_mark_abandoned(
                code,
                str(action_profile.get("abandon_reason") or f"{label} bounded exploit 未发现 flag"),
            )
        summary = str(action_profile.get("summary") or "{code} 已完成 bounded 探测。").format(
            code=code
        )
        return (
            f"确定性 {label} {action}：{summary}\n"
            f"探测摘要:\n{probe[:3500]}"
        )

    @staticmethod
    def _benchmark_fast_action_from_task(
        task: dict[str, Any] | None = None,
        desc: str = "",
    ) -> str:
        if isinstance(task, dict):
            action = str(
                task.get("benchmark_action")
                or task.get("benchmark_step")
                or ""
            ).strip().lower()
            if action in {"setup", "schedule", "probe"}:
                return action
        if "Benchmark fast setup" in desc:
            return "setup"
        if "Benchmark fast step 1" in desc:
            return "schedule"
        if "Benchmark fast step 2" in desc:
            return "probe"
        return ""

    def _benchmark_deterministic_fast_step(
        self,
        desc: str,
        reason: str = "",
        *,
        action: str | None = None,
    ) -> str | None:
        """Run the policy fast path without an LLM when the step is mechanical."""
        if not self._is_benchmark_aggressive():
            return None
        fast_action = (
            action
            if action in {"setup", "schedule", "probe"}
            else self._benchmark_fast_action_from_task(desc=desc)
        )
        service_action = self._benchmark_service_action_from_desc(desc)
        if service_action:
            fingerprint, action = service_action
            return self._benchmark_run_service_action_step(fingerprint, action)
        if "Benchmark handoff step 1" in desc:
            code, addrs = self._benchmark_active_challenge_from_state()
            if not code:
                return "确定性 handoff：当前没有 active 容器，跳过。"
            probe = self._benchmark_probe_container_local(code, addrs)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                closed = self._benchmark_close_local(code)
                return (
                    f"确定性 handoff：{code} 已提交成功并关闭。\n"
                    f"close 返回: {closed[:200]}\n探测摘要:\n{probe[:3000]}"
                )
            return (
                f"确定性 handoff：已围绕当前题 {code} 执行登录态/LFI/源码派生探测，"
                "未直接发现 flag；保留给 step 2 做一个高置信后续假设。\n"
                f"探测摘要:\n{probe[:3500]}"
            )
        if "Benchmark handoff step 2" in desc:
            code, addrs = self._benchmark_active_challenge_from_state()
            if not code:
                return "确定性 handoff：当前没有 active 容器，跳过。"
            with self._benchmark_state_lock:
                fingerprints = dict(self._benchmark_state.get("service_fingerprints", {}))
            if self._benchmark_fingerprint_has_profiled_handoff(fingerprints.get(code)):
                return None
            probe = self._benchmark_probe_handoff_followup_local(code, addrs)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                closed = self._benchmark_close_local(code)
                return (
                    f"确定性 handoff：{code} 后续假设提交成功并关闭。\n"
                    f"close 返回: {closed[:200]}\n探测摘要:\n{probe[:3000]}"
                )
            if any(
                marker in probe.lower()
                for marker in ("hugegraph", "gremlin", "arthas", "jdwp")
            ):
                self._benchmark_mark_reasoning_needed(
                    code,
                    "服务专项线索仍有利用空间，禁止按普通 Web 低收益题关闭",
                )
                return (
                    f"确定性 handoff：{code} 发现/确认服务专项线索，"
                    "保留 active 给四柱继续深挖，不标记 abandoned。\n"
                    f"探测摘要:\n{probe[:3000]}"
                )
            self._benchmark_mark_abandoned(code, "handoff 高置信后续假设无 flag")
            return (
                f"确定性 handoff：{code} 后续高置信假设未发现 flag。\n"
                "已标记为 abandoned，下一轮应先 close 释放容器。\n"
                f"探测摘要:\n{probe[:3000]}"
            )
        if "Benchmark handoff step 3" in desc:
            code, _addrs = self._benchmark_active_challenge_from_state()
            if not code:
                return "确定性 handoff：当前没有 active 容器，无需 close。"
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
                reasoning = set(self._benchmark_state.get("reasoning_challenges", set()))
                abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            if code in completed:
                closed = self._benchmark_close_local(code)
                return f"确定性 handoff：{code} 已完成，close 返回: {closed[:200]}"
            if code in reasoning and code not in abandoned:
                return (
                    f"确定性 handoff：{code} 仍处于 reasoning 状态且未被判定低收益，"
                    "跳过机械 close，保留 active 给四柱/runner 继续服务专项利用。"
                )
            closed = self._benchmark_close_local(code)
            return f"确定性 handoff：{code} 无直接突破，已 close 释放资源: {closed[:300]}"
        if fast_action == "setup":
            notes: list[str] = []
            detected = self._benchmark_detect_tun_local()
            if detected:
                notes.append(f"复用已连接 VPN：{detected[0]} {detected[1]}")
            else:
                notes.append(self._benchmark_start_vpn_local())
                detected = self._benchmark_detect_tun_local()
            with self._benchmark_state_lock:
                if not self._benchmark_state.get("api_interface"):
                    self._benchmark_state["api_interface"] = "enp0s20f0u3u4"
            try:
                challenges = self._benchmark_list_challenges_local()
                notes.append(f"平台 API 校验成功，题目数={len(challenges)}")
            except Exception as exc:
                notes.append(f"平台 API 校验失败: {exc}")
            self._persist_benchmark_state()
            return "确定性 setup：" + "；".join(notes)

        if fast_action == "schedule":
            challenges = self._benchmark_list_challenges_local()
            if not challenges:
                return "确定性调度：平台已返回终止状态或无题目列表。"
            cleanup_results = self._benchmark_close_completed_active_from_snapshot(challenges)
            cleanup_note = ""
            if cleanup_results:
                cleanup_note = "确定性调度：先关闭已完成 active 容器释放名额: " + " | ".join(cleanup_results)
                challenges = self._benchmark_list_challenges_local()
            active_items = [
                item for item in challenges
                if self._benchmark_is_active_status(item)
                and item.get("is_completed") is not True
            ]
            with self._benchmark_state_lock:
                abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            for item in active_items:
                code = item.get("unique_code")
                if isinstance(code, str) and code in abandoned:
                    closed = self._benchmark_close_local(code)
                    return f"确定性调度：先关闭 stale 当前题 {code}: {closed[:200]}"
            if active_items:
                code = active_items[0].get("unique_code")
                addrs = active_items[0].get("container_addr") or []
                active_msg = f"确定性调度：继续当前 active 题 {code} => {addrs}"
                return f"{cleanup_note}\n{active_msg}" if cleanup_note else active_msg
            next_item = self._benchmark_select_next_candidate(challenges)
            if next_item is None:
                no_candidate_msg = (
                    "确定性调度：未发现未完成 stopped 候选，后续应刷新平台状态或等待资源。"
                )
                return (
                    f"{cleanup_note}\n{no_candidate_msg}"
                    if cleanup_note
                    else no_candidate_msg
                )
            code = str(next_item["unique_code"])
            difficulty = str(next_item.get("difficulty") or "unknown").lower()
            started = self._benchmark_start_local(code)
            if "max active challenge instances reached" in started.lower():
                refreshed = self._benchmark_list_challenges_local()
                retry_cleanup = self._benchmark_close_completed_active_from_snapshot(refreshed)
                if retry_cleanup:
                    challenges = self._benchmark_list_challenges_local()
                    retry_item = next(
                        (
                            item
                            for item in challenges
                            if item.get("unique_code") == code
                            and item.get("is_completed") is not True
                            and self._benchmark_is_startable_status(item)
                        ),
                        None,
                    )
                    if retry_item is not None:
                        retry_started = self._benchmark_start_local(code)
                        retry_msg = (
                            f"确定性调度：start 遇到 active 名额满，已关闭完成题释放名额: "
                            f"{' | '.join(retry_cleanup)}；重试启动 {code} ({difficulty}): "
                            f"{retry_started[:300]}"
                        )
                        return f"{cleanup_note}\n{retry_msg}" if cleanup_note else retry_msg
                blocked_msg = (
                    f"确定性调度：启动下一道 {difficulty} {code} 失败，"
                    "active 名额已满且未发现可清理的已完成容器；"
                    "不要放弃该题，下一步应 close stale active 或刷新状态。"
                )
                return f"{cleanup_note}\n{blocked_msg}" if cleanup_note else blocked_msg
            start_msg = f"确定性调度：启动下一道 {difficulty} {code}: {started[:300]}"
            return f"{cleanup_note}\n{start_msg}" if cleanup_note else start_msg

        if fast_action == "probe":
            code, addrs = self._benchmark_active_challenge_from_state()
            if not code or not addrs:
                refreshed_code, refreshed_addrs = self._benchmark_refresh_active_challenge_from_platform()
                if refreshed_code:
                    code, addrs = refreshed_code, refreshed_addrs
            if not code:
                return "确定性探测：当前没有 active 容器，跳过。"
            if not addrs:
                return (
                    f"确定性探测：{code} 平台显示 active 但没有 container_addr，"
                    "本轮只刷新状态，不 start 新题。"
                )
            with self._benchmark_state_lock:
                abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            if code in abandoned:
                closed = self._benchmark_close_local(code)
                return f"确定性探测：{code} 已 stale，直接 close: {closed[:200]}"
            probe = self._benchmark_probe_container_local(code, addrs)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                self._benchmark_clear_probe_unreachable(code)
                closed = self._benchmark_close_local(code)
                return (
                    f"确定性探测：{code} 已提交成功并关闭。"
                    f"触发原因: {reason or 'fast path'}\n"
                    f"close 返回: {closed[:200]}\n"
                    f"探测摘要:\n{probe[:2500]}"
                )
            if self._benchmark_probe_looks_unreachable(probe):
                streak = self._benchmark_note_probe_unreachable(code)
                retry_limit = self._benchmark_unreachable_retry_limit()
                if streak < retry_limit:
                    return (
                        f"确定性探测：{code} 容器暂不可达（第 {streak} 次），"
                        f"保留 active，下一轮重试 readiness/probe，不 start 新题（阈值 {retry_limit}）。"
                        f"触发原因: {reason or 'fast path'}\n"
                        f"探测摘要:\n{probe[:2500]}"
                    )
                closed = self._benchmark_close_local(code)
                return (
                    f"确定性探测：{code} 连续 {streak} 次不可达，已关闭止损。"
                    f"触发原因: {reason or 'fast path'}\n"
                    f"close 返回: {closed[:200]}\n"
                    f"探测摘要:\n{probe[:2500]}"
                )
            with self._benchmark_state_lock:
                abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
                reasoning = set(self._benchmark_state.get("reasoning_challenges", set()))
                fingerprints = dict(self._benchmark_state.get("service_fingerprints", {}))
            if code in abandoned:
                closed = self._benchmark_close_local(code)
                fingerprint = fingerprints.get(code)
                service_note = f"（服务指纹 {fingerprint}）" if fingerprint else ""
                return (
                    f"确定性探测：{code} 已由 bounded service profile 判定低收益{service_note}，"
                    "已 close 释放资源。\n"
                    f"触发原因: {reason or 'fast path'}\n"
                    f"close 返回: {closed[:200]}\n"
                    f"探测摘要:\n{probe[:2500]}"
                )
            if code in reasoning:
                fingerprint = fingerprints.get(code)
                service_note = f"（服务指纹 {fingerprint}）" if fingerprint else ""
                return (
                    f"确定性探测：{code} 已由 service profile 标记为需要深挖{service_note}，"
                    "保留 active 并切回四柱/runner；本轮不 close、不 start 新题。"
                    f"触发原因: {reason or 'fast path'}\n"
                    f"探测摘要:\n{probe[:2500]}"
                )
            self._benchmark_mark_reasoning_needed(
                code,
                "fast path 已获取可达响应/协议线索但未直接发现 flag",
            )
            return (
                f"确定性探测：{code} 已获取有效响应但未直接发现 flag，"
                "保留 active 并切回四柱/runner 深挖；本轮不 close、不 start 新题。"
                f"触发原因: {reason or 'fast path'}\n"
                f"探测摘要:\n{probe[:2500]}"
            )
        return None

    def _benchmark_deterministic_standard_task(self, desc: str) -> str | None:
        if not self._is_benchmark_aggressive() or "Benchmark " in desc:
            return None
        classification = self._benchmark_classify_standard_task(desc)
        if classification == "setup":
            state_context = self._benchmark_state_context()
            return (
                "确定性标准任务：Benchmark setup 已由 fast path 校验，"
                "直接复用当前运行态；不再重复慢速读取/搜索配置。\n"
                f"{state_context[:1200]}"
            )
        if classification == "probe":
            return self._benchmark_deterministic_fast_step(
                "Benchmark fast step 2：只解当前已启动的 10.x 容器。",
                reason="standard_mechanical_probe",
                action="probe",
            )
        if classification == "schedule":
            return self._benchmark_deterministic_fast_step(
                "Benchmark fast step 1：只做调度。",
                reason="standard_mechanical_schedule",
                action="schedule",
            )
        return None

    @staticmethod
    def _benchmark_standard_task_rules() -> list[dict[str, Any]]:
        return [
            {
                "kind": "setup",
                "any": ("challenges_api", "api文档", "vpn配置"),
                "any_original": ("读取", "获取"),
            },
            {
                "kind": "probe",
                "any_original": ("快速指纹", "寻找flag", "尝试提交"),
            },
            {
                "kind": "probe",
                "any": ("obvious flag", "post submit"),
            },
            {
                "kind": "probe",
                "all": ("submit",),
                "any": ("flag", "close"),
            },
            {
                "kind": "probe",
                "any_original": ("探测",),
                "any_original_2": ("容器", "10.x"),
            },
            {
                "kind": "schedule",
                "any_original": ("列出",),
                "any_original_2": ("题目",),
            },
            {
                "kind": "schedule",
                "any_original": ("题目列表",),
                "any_original_2": ("获取", "筛选", "刷新", "解析"),
            },
            {
                "kind": "schedule",
                "any_original": ("挑战列表",),
                "any_original_2": ("获取", "调用", "确定"),
            },
            {
                "kind": "schedule",
                "any_original": ("可做",),
                "any_original_2": ("题目", "挑战"),
            },
            {
                "kind": "schedule",
                "all": ("list接口",),
                "any_original": ("题目",),
            },
            {
                "kind": "schedule",
                "all": ("list api",),
                "any_original": ("解析", "排序", "排除", "题目"),
            },
            {
                "kind": "schedule",
                "all": ("list接口",),
                "any": ("easy",),
            },
            {
                "kind": "schedule",
                "all": ("平台api",),
                "any_original": ("挑战列表", "题目"),
            },
            {
                "kind": "schedule",
                "any_original": ("筛选",),
                "any": ("easy", "低level"),
            },
            {
                "kind": "schedule",
                "all": ("post start",),
                "any_original": ("题",),
            },
            {
                "kind": "schedule",
                "all": ("post start",),
                "any": ("easy",),
            },
            {
                "kind": "schedule",
                "all": ("start",),
                "any_original": ("重新", "下一个", "下一道", "未完成"),
                "any_original_2": ("题", "容器"),
            },
            {
                "kind": "schedule",
                "all": ("start api",),
                "any_original": ("选中", "未完成", "第一道", "启动"),
            },
            {
                "kind": "schedule",
                "any_original": ("排序", "排除已完成", "解析"),
                "any": ("stopped", "easy", "level"),
            },
            {
                "kind": "schedule",
                "any_original": ("选择一道",),
                "any": ("easy", "低level"),
            },
            {
                "kind": "schedule",
                "any_original": ("启动",),
                "any_original_2": ("容器",),
            },
            {
                "kind": "schedule",
                "any_original": ("启动",),
                "any_original_2": ("题",),
                "any": ("easy", "低level"),
            },
            {
                "kind": "schedule",
                "all": ("start接口",),
                "any_original": ("启动",),
            },
            {
                "kind": "schedule",
                "any_original": ("下一道",),
                "any_original_2": ("题",),
            },
        ]

    @classmethod
    def _benchmark_classify_standard_task(cls, desc: str) -> str | None:
        lowered = desc.lower()
        for rule in cls._benchmark_standard_task_rules():
            if not all(token in lowered for token in rule.get("all", ())):
                continue
            if not all(token in desc for token in rule.get("all_original", ())):
                continue
            if rule.get("any") and not any(token in lowered for token in rule["any"]):
                continue
            if rule.get("any_original") and not any(
                token in desc for token in rule["any_original"]
            ):
                continue
            if rule.get("any_original_2") and not any(
                token in desc for token in rule["any_original_2"]
            ):
                continue
            return str(rule["kind"])
        return None

    def _benchmark_active_submit_code(self, evidence: str = "") -> str | None:
        with self._benchmark_state_lock:
            completed = set(self._benchmark_state.get("completed_challenges", set()))
            closed = set(self._benchmark_state.get("closed_challenges", set()))
            active = dict(self._benchmark_state.get("active_containers", {}))
            current = self._benchmark_state.get("current_challenge")
        excluded = completed | closed
        if isinstance(current, str) and current and current not in excluded:
            return current
        if evidence:
            matched_codes: list[str] = []
            for code, addrs in active.items():
                if not isinstance(code, str) or code in excluded:
                    continue
                for addr in addrs or []:
                    if not isinstance(addr, str) or not addr:
                        continue
                    host = addr.split(":", 1)[0]
                    if addr in evidence or f"http://{addr}" in evidence or host in evidence:
                        matched_codes.append(code)
                        break
            matched_codes = list(dict.fromkeys(matched_codes))
            if len(matched_codes) == 1:
                return matched_codes[0]
        active_candidates = [
            code for code in active
            if isinstance(code, str) and code not in excluded
        ]
        if len(active_candidates) == 1:
            return active_candidates[0]
        return None

    def _benchmark_auto_submit_flags_from_tool_result(self, content: str) -> None:
        """Submit flags immediately when a container tool output exposes one.

        This is intentionally outside the model loop: if the subtask times out
        after discovering a flag, the score is still claimed before close/switch.
        """
        if not self._is_benchmark_aggressive() or not content:
            return
        command = self._extract_command_from_tool_result(content)
        output = self._extract_output_from_tool_result(content)
        lowered_command = command.lower()
        if (
            "/openapi/v1/challenges" in lowered_command
            or "challenges_api.md" in lowered_command
            or ".events.jsonl" in lowered_command
        ):
            return
        if "10." not in command and "http://10." not in output and "https://10." not in output:
            return
        flags = _re_mod.findall(
            r"\b(?:flag|ctf|tsec)\{[^}\s]{4,}\}",
            output,
            flags=_re_mod.IGNORECASE,
        )
        if not flags:
            return
        code = self._benchmark_active_submit_code(f"{command}\n{output}")
        if not code:
            self._record_trace(
                "benchmark_auto_submit_skipped",
                detail="工具输出包含 flag，但无法唯一确定当前 active unique_code。",
                metadata={"flags": flags[:3]},
            )
            return
        api_config = self._benchmark_api_config_from_workspace()
        if api_config is None:
            self._record_trace(
                "benchmark_auto_submit_skipped",
                detail="工具输出包含 flag，但无法读取 CHALLENGES_API.md 中的 API 配置。",
                metadata={"challenge": code},
            )
            return
        base_url, token = api_config
        with self._benchmark_state_lock:
            submitted = set(self._benchmark_state.get("auto_submitted_flags", set()))
            api_interface = self._benchmark_state.get("api_interface") or "enp0s20f0u3u4"

        for flag in dict.fromkeys(flags):
            submit_key = f"{code}:{flag}"
            if submit_key in submitted:
                continue
            body = json.dumps(
                {"unique_code": code, "flag": flag},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            cmd = [
                "curl",
                "-sS",
                "--interface",
                str(api_interface),
                "--connect-timeout",
                "10",
                "--max-time",
                "20",
                "-X",
                "POST",
                f"{base_url}/openapi/v1/challenges/submit",
                "-H",
                f"BENCHMARK_TOKEN: {token}",
                "-H",
                "Content-Type: application/json",
                "-d",
                body,
            ]
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
            except Exception as exc:
                self._record_trace(
                    "benchmark_auto_submit_failed",
                    detail=f"{code} 自动提交 flag 失败：{exc}",
                    metadata={"challenge": code},
                )
                continue

            with self._benchmark_state_lock:
                submitted = set(self._benchmark_state.get("auto_submitted_flags", set()))
                submitted.add(submit_key)
                self._benchmark_state["auto_submitted_flags"] = submitted

            synthetic_content = (
                f"命令: curl --interface {api_interface} "
                "-X POST "
                f"{base_url}/openapi/v1/challenges/submit "
                f"-H 'BENCHMARK_TOKEN: {token}' "
                f"-d '{body}'\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{result.stdout}"
            )
            self._record_trace(
                "benchmark_auto_submit",
                detail=f"{code} 自动提交工具输出中的 flag，返回码 {result.returncode}。",
                metadata={
                    "challenge": code,
                    "stdout": result.stdout[:500],
                    "stderr": result.stderr[:500],
                },
            )
            self._update_benchmark_runtime_state(synthetic_content)
            if result.returncode == 0 and (
                '"correct":true' in result.stdout.lower()
                or '"duplicate"' in result.stdout.lower()
                or '"code":"duplicate"' in result.stdout.lower()
            ):
                break

    @staticmethod
    def _extract_command_from_tool_result(content: str) -> str:
        match = _re_mod.search(r"命令:\s*(.*?)\n工作目录:", content, _re_mod.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_output_from_tool_result(content: str) -> str:
        match = _re_mod.search(r"\n输出:\n(.*)$", content, _re_mod.DOTALL)
        return match.group(1).strip() if match else content

    @staticmethod
    def _iter_json_fragments(text: str) -> list[Any]:
        decoder = json.JSONDecoder()
        results: list[Any] = []
        for idx, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(text[idx:])
            except Exception:
                continue
            results.append(value)
        return results

    def _update_benchmark_runtime_state(self, content: str) -> None:
        if not self._is_benchmark_aggressive() or not content:
            return

        command = self._extract_command_from_tool_result(content)
        output = self._extract_output_from_tool_result(content)
        combined = f"{command}\n{output}"
        lowered = combined.lower()
        command_lowered = command.lower()
        is_platform_challenges_command = "/openapi/v1/challenges" in command_lowered
        is_platform_list_command = (
            is_platform_challenges_command
            and "/openapi/v1/challenges/" not in command_lowered
            and "|" not in command
            and ">" not in command
        )
        is_platform_submit_command = (
            is_platform_challenges_command and "/openapi/v1/challenges/submit" in command_lowered
        )
        updates: dict[str, Any] = {}

        tun_match = _re_mod.search(
            r"\b(tun\d+):[\s\S]{0,240}?\binet\s+([0-9.]+)/\d+",
            combined,
        )
        if tun_match:
            updates["vpn_connected"] = True
            updates["tun_interface"] = tun_match.group(1)
            updates["tun_ip"] = tun_match.group(2)

        vpn_config = _re_mod.search(r"openvpn\s+--config\s+([^\s]+\.ovpn)", command)
        if vpn_config:
            updates["vpn_config"] = vpn_config.group(1)

        if (
            "tsecbench.zc.tencent.com" in lowered
            and "/openapi/v1/challenges" in lowered
            and "退出码: 0" in content
            and "invalid_state" not in lowered
            and "task_not_found" not in lowered
        ):
            used_interface = self._command_uses_interface(command)
            if used_interface and not _re_mod.fullmatch(r"tun\d+", used_interface):
                updates["api_interface"] = used_interface

        if is_platform_challenges_command and "invalid_state" in lowered and "finished" in lowered:
            updates["task_finished"] = True

        completed: set[str] = set()
        completed_scores: dict[str, int] = {}
        closed: set[str] = set()
        active_updates: dict[str, list[str]] = {}
        current_challenge: str | None = None
        last_score: int | None = None
        challenges_snapshot: list[dict[str, Any]] | None = None

        json_fragments = self._iter_json_fragments(output) if is_platform_challenges_command else []
        if is_platform_list_command:
            list_fragment = next(
                (
                    value
                    for value in json_fragments
                    if isinstance(value, list)
                    and all(
                        isinstance(item, dict) and "unique_code" in item
                        for item in value
                    )
                ),
                None,
            )
            json_fragments = [list_fragment] if isinstance(list_fragment, list) else []
        for value in json_fragments:
            if isinstance(value, dict):
                code = value.get("unique_code")
                if isinstance(code, str) and code:
                    addrs = value.get("container_addr")
                    if isinstance(addrs, list) and addrs and (
                        self._benchmark_is_active_status(value)
                        or "/challenges/start" in lowered
                    ):
                        active_updates[code] = [str(addr) for addr in addrs]
                        current_challenge = code
                    if value.get("closed") is True:
                        closed.add(code)
                elif value.get("closed") is True:
                    close_code = self._benchmark_extract_unique_code(command)
                    if close_code:
                        closed.add(close_code)

                if value.get("correct") is True:
                    submit_code = self._benchmark_extract_unique_code(command)
                    if submit_code:
                        completed.add(submit_code)
                        current_challenge = submit_code
                    awarded = value.get("awarded")
                    score = value.get("cumulative_score")
                    if isinstance(score, int) and submit_code:
                        completed_scores[submit_code] = score
                    elif isinstance(awarded, int) and submit_code:
                        completed_scores[submit_code] = awarded
                    if isinstance(score, int):
                        last_score = score

                if value.get("is_completed") is True:
                    code = value.get("unique_code")
                    if isinstance(code, str):
                        completed.add(code)
                        total_score = value.get("total_score")
                        if isinstance(total_score, int):
                            completed_scores.setdefault(code, total_score)

            elif isinstance(value, list):
                if is_platform_list_command and value and all(
                    isinstance(item, dict) and "unique_code" in item
                    for item in value
                ):
                    challenges_snapshot = [dict(item) for item in value]
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    code = item.get("unique_code")
                    if not isinstance(code, str):
                        continue
                    if item.get("is_completed") is True:
                        completed.add(code)
                        total_score = item.get("total_score")
                        if isinstance(total_score, int):
                            completed_scores.setdefault(code, total_score)
                    if self._benchmark_is_active_status(item):
                        addrs = item.get("container_addr")
                        if isinstance(addrs, list) and addrs:
                            active_updates[code] = [str(addr) for addr in addrs]
                            current_challenge = code

        if is_platform_challenges_command and not challenges_snapshot:
            for match in _re_mod.finditer(
                r"(?P<code>\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b)"
                r"\s*:\s*completed=(?P<completed>true|false)"
                r"\s*,\s*status=(?P<status>[A-Za-z_]+)"
                r"\s*,\s*addr=(?P<addr>\[[^\]\n]{0,500}\])",
                output,
                flags=_re_mod.IGNORECASE,
            ):
                code = match.group("code")
                completed_value = match.group("completed").lower() == "true"
                status = match.group("status").lower()
                addrs = _re_mod.findall(
                    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}",
                    match.group("addr"),
                )
                if completed_value:
                    completed.add(code)
                    continue
                if status == "available" and addrs:
                    active_updates[code] = [str(addr) for addr in addrs]
                    current_challenge = code

        state_changed = bool(
            updates
            or completed
            or completed_scores
            or closed
            or active_updates
            or challenges_snapshot is not None
            or current_challenge
            or last_score is not None
        )
        if not state_changed:
            return

        with self._benchmark_state_lock:
            self._benchmark_state.update(updates)
            active = dict(self._benchmark_state.get("active_containers", {}))
            active.update(active_updates)
            completed_set = set(self._benchmark_state.get("completed_challenges", set()))
            closed_set = set(self._benchmark_state.get("closed_challenges", set()))
            abandoned_set = set(self._benchmark_state.get("abandoned_challenges", set()))
            recovered_set = set(
                self._benchmark_state.get("recovery_attempted_challenges", set())
            )
            reasoning_set = set(self._benchmark_state.get("reasoning_challenges", set()))
            service_fingerprints = dict(
                self._benchmark_state.get("service_fingerprints", {})
            )
            unreachable_streaks = dict(
                self._benchmark_state.get("probe_unreachable_streaks", {})
            )
            score_map = dict(self._benchmark_state.get("completed_scores", {}))
            completed_set.update(completed)
            closed_set.update(closed)
            for code, score in completed_scores.items():
                if is_platform_submit_command:
                    score_map[code] = score
                else:
                    score_map.setdefault(code, score)
            if challenges_snapshot is not None:
                self._benchmark_state["last_challenges_snapshot"] = challenges_snapshot
                self._benchmark_state["task_finished"] = False
                snapshot_completed = {
                    str(item["unique_code"])
                    for item in challenges_snapshot
                    if item.get("is_completed") is True
                }
                completed_set.update(snapshot_completed)
                for item in challenges_snapshot:
                    code = item.get("unique_code")
                    total_score = item.get("total_score")
                    if (
                        isinstance(code, str)
                        and code in completed_set
                        and isinstance(total_score, int)
                    ):
                        score_map.setdefault(code, total_score)
                active = {}
                for item in challenges_snapshot:
                    code = item.get("unique_code")
                    addrs = item.get("container_addr")
                    if (
                        isinstance(code, str)
                        and self._benchmark_is_active_status(item)
                        and isinstance(addrs, list)
                        and addrs
                        and code not in completed_set
                    ):
                        closed_set.discard(code)
                        abandoned_set.discard(code)
                        recovered_set.discard(code)
                        active[code] = [str(addr) for addr in addrs]
                        current_challenge = code
            for code in completed_set | closed_set:
                active.pop(code, None)
                abandoned_set.discard(code)
                reasoning_set.discard(code)
                service_fingerprints.pop(code, None)
                unreachable_streaks.pop(code, None)
            if self._benchmark_state.get("task_finished"):
                active = {}
            self._benchmark_state["active_containers"] = active
            self._benchmark_state["completed_challenges"] = completed_set
            self._benchmark_state["completed_scores"] = score_map
            self._benchmark_state["closed_challenges"] = closed_set
            self._benchmark_state["abandoned_challenges"] = abandoned_set
            self._benchmark_state["recovery_attempted_challenges"] = recovered_set
            self._benchmark_state["reasoning_challenges"] = reasoning_set
            self._benchmark_state["service_fingerprints"] = service_fingerprints
            self._benchmark_state["probe_unreachable_streaks"] = unreachable_streaks
            if current_challenge and current_challenge not in completed_set and current_challenge not in closed_set:
                self._benchmark_state["current_challenge"] = current_challenge
                self._benchmark_current_challenge = current_challenge
            else:
                stored_current = self._benchmark_state.get("current_challenge")
                if (
                    isinstance(stored_current, str)
                    and stored_current
                    and (
                        stored_current in completed_set
                        or stored_current in closed_set
                        or stored_current in abandoned_set
                    )
                    and stored_current not in active
                ):
                    self._benchmark_state["current_challenge"] = None
                    if self._benchmark_current_challenge == stored_current:
                        self._benchmark_current_challenge = None
            if last_score is not None:
                self._benchmark_state["last_score"] = last_score
            should_cleanup = (
                bool(self._benchmark_state.get("task_finished"))
                and not bool(self._benchmark_state.get("background_processes_cleaned"))
            )
            snapshot = self._benchmark_state_snapshot_unlocked()

        self._record_trace(
            "benchmark_state_updated",
            detail=self._benchmark_state_summary(snapshot),
            metadata=snapshot,
        )
        if (
            bool(self._benchmark_state.get("task_finished"))
            or updates
            or completed
            or completed_scores
            or closed
            or active_updates
            or challenges_snapshot is not None
            or current_challenge
            or last_score is not None
        ):
            self._persist_benchmark_state()
        if should_cleanup:
            self._cleanup_benchmark_background_processes()

    def _benchmark_state_snapshot_unlocked(self) -> dict[str, Any]:
        last_snapshot = self._benchmark_state.get("last_challenges_snapshot")
        snapshot_summary = None
        if isinstance(last_snapshot, list):
            completed_items = [
                item for item in last_snapshot
                if isinstance(item, dict) and item.get("is_completed") is True
            ]
            active_items = [
                item for item in last_snapshot
                if isinstance(item, dict) and self._benchmark_is_active_status(item)
            ]
            snapshot_summary = {
                "total": len(last_snapshot),
                "completed_count": len(completed_items),
                "completed_total_score": sum(
                    item.get("total_score") or 0 for item in completed_items
                ),
                "completed_challenges": sorted(
                    str(item.get("unique_code")) for item in completed_items
                ),
                "active_challenges": sorted(
                    str(item.get("unique_code")) for item in active_items
                ),
            }
        return {
            "vpn_connected": bool(self._benchmark_state.get("vpn_connected")),
            "tun_interface": self._benchmark_state.get("tun_interface"),
            "tun_ip": self._benchmark_state.get("tun_ip"),
            "vpn_config": self._benchmark_state.get("vpn_config"),
            "api_interface": self._benchmark_state.get("api_interface"),
            "task_finished": bool(self._benchmark_state.get("task_finished")),
            "current_challenge": self._benchmark_state.get("current_challenge"),
            "active_containers": dict(self._benchmark_state.get("active_containers", {})),
            "last_challenges_snapshot": snapshot_summary,
            "completed_challenges": sorted(self._benchmark_state.get("completed_challenges", set())),
            "completed_scores": dict(self._benchmark_state.get("completed_scores", {})),
            "closed_challenges": sorted(self._benchmark_state.get("closed_challenges", set())),
            "abandoned_challenges": sorted(self._benchmark_state.get("abandoned_challenges", set())),
            "recovery_attempted_challenges": sorted(
                self._benchmark_state.get("recovery_attempted_challenges", set())
            ),
            "reasoning_challenges": sorted(
                self._benchmark_state.get("reasoning_challenges", set())
            ),
            "service_fingerprints": dict(
                self._benchmark_state.get("service_fingerprints", {})
            ),
            "observed_probe_paths": sorted(
                self._benchmark_state.get("observed_probe_paths", set())
            ),
            "observed_param_names": sorted(
                self._benchmark_state.get("observed_param_names", set())
            ),
            "probe_unreachable_streaks": dict(
                self._benchmark_state.get("probe_unreachable_streaks", {})
            ),
            "auto_submitted_flags": sorted(self._benchmark_state.get("auto_submitted_flags", set())),
            "last_score": self._benchmark_state.get("last_score"),
            "setup_timeout_streak": int(
                self._benchmark_state.get("setup_timeout_streak") or 0
            ),
            "redundant_block_count": int(
                self._benchmark_state.get("redundant_block_count") or 0
            ),
            "background_processes_cleaned": bool(
                self._benchmark_state.get("background_processes_cleaned")
            ),
        }

    def _benchmark_state_snapshot(self) -> dict[str, Any]:
        with self._benchmark_state_lock:
            return self._benchmark_state_snapshot_unlocked()

    @staticmethod
    def _benchmark_state_summary(state: dict[str, Any]) -> str:
        parts: list[str] = []
        if state.get("task_finished"):
            parts.append(
                "任务状态：平台曾返回 invalid_state/finished；"
                "如目标分未达成，应先刷新平台列表确认真实状态，"
                "不要仅凭一次错误响应结束。"
            )
        snapshot = state.get("last_challenges_snapshot")
        if isinstance(snapshot, dict):
            parts.append(
                "最后平台快照："
                f"{snapshot.get('completed_count', 0)}/{snapshot.get('total', '?')} "
                f"题完成，已完成题总分 {snapshot.get('completed_total_score', 0)}。"
            )
        if state.get("vpn_connected"):
            tun = state.get("tun_interface") or "tun0"
            ip = state.get("tun_ip") or "unknown"
            parts.append(f"VPN：已连接 {tun} {ip}，不要重复启动 OpenVPN。")
        if state.get("api_interface"):
            parts.append(
                f"平台 API：使用 --interface {state['api_interface']}，"
                "认证头必须为 BENCHMARK_TOKEN；不要用 tun0 或 Authorization Bearer。"
            )
        active = state.get("active_containers") or {}
        if active:
            active_text = ", ".join(
                f"{code}=>{','.join(addrs)}"
                for code, addrs in sorted(active.items())
            )
            parts.append(f"当前活跃容器：{active_text}。")
        if state.get("current_challenge"):
            parts.append(f"当前题：{state['current_challenge']}。")
        if state.get("completed_challenges"):
            parts.append(
                "已通关题："
                + ", ".join(state["completed_challenges"])
                + "；禁止重复 start/探测。"
            )
            scores = state.get("completed_scores") or {}
            if scores:
                known_score = sum(
                    score for score in scores.values() if isinstance(score, int)
                )
                parts.append(
                    f"已知通关得分：{known_score}（{len(scores)} 道题有分值记录）。"
                )
        if state.get("closed_challenges"):
            parts.append(
                "本地已关闭/软跳过题："
                + ", ".join(state["closed_challenges"])
                + "；平台真实状态优先，若平台仍显示未完成 stopped，允许恢复一次后继续探测。"
            )
        if state.get("abandoned_challenges"):
            parts.append(
                "已判定低收益待关闭题："
                + ", ".join(state["abandoned_challenges"])
                + "；必须先 close，再换下一题，不要继续探测其容器。"
            )
        if state.get("reasoning_challenges"):
            parts.append(
                "已有有效响应线索、必须深挖的题："
                + ", ".join(state["reasoning_challenges"])
                + "；这是 fast path handoff，不是前置 setup。禁止生成/执行工具链检查、"
                "VPN 启动、重复读取目录、批量 list/start 新题等任务；必须先围绕当前"
                " active 容器的页面/API/源码/配置线索验证。"
            )
        fingerprints = state.get("service_fingerprints") or {}
        if isinstance(fingerprints, dict) and fingerprints:
            parts.append(
                "已识别服务指纹："
                + ", ".join(
                    f"{code}={fingerprint}"
                    for code, fingerprint in sorted(fingerprints.items())
                )
                + "；后续攻击路径必须围绕该真实服务类型，不要套用无关 Web/PHP 模板。"
            )
        observed_paths = state.get("observed_probe_paths") or []
        observed_params = state.get("observed_param_names") or []
        if observed_paths or observed_params:
            parts.append(
                "运行时自增长候选："
                f"{len(observed_paths)} 个路径、{len(observed_params)} 个参数名；"
                "后续探测会在 execution_control_policy 预算内复用这些真实响应来源的候选。"
            )
        if state.get("last_score") is not None:
            parts.append(f"最近题目累计得分：{state['last_score']}。")
        return "\n".join(parts)

    def _benchmark_state_context(self) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        summary = self._benchmark_state_summary(self._benchmark_state_snapshot())
        target_directive = self._benchmark_target_continue_directive()
        if target_directive:
            summary = f"{summary}\n{target_directive}" if summary else target_directive
        if not summary:
            return ""
        return (
            "## Benchmark 已确认运行态（必须信任并复用）\n"
            f"{summary}\n"
            "除非上面的状态被平台 API 明确推翻，否则不要重复做 VPN 启动、"
            "不要回头探测已完成题或已判定低收益题；本地已关闭但未 abandoned、"
            "且平台仍未完成的 stopped 题可恢复一次。"
            "10.x 容器访问必须显式使用 "
            "`curl --interface tun0`；平台 API 才使用物理网卡。"
        )

    @staticmethod
    def _benchmark_builtin_service_handoff_profiles() -> dict[str, dict[str, Any]]:
        return {}

    def _benchmark_normalize_service_handoff_profile(
        self,
        raw: Any,
    ) -> tuple[str, dict[str, Any]] | None:
        if not isinstance(raw, dict):
            return None
        fingerprint = str(raw.get("fingerprint") or "").strip().lower()
        if not _re_mod.fullmatch(r"[a-z0-9_.-]{1,80}", fingerprint):
            return None
        context = str(raw.get("context") or raw.get("handoff_context") or "").strip()
        evidence_focus = self._benchmark_string_tuple(
            raw.get("evidence_focus", raw.get("handoff_focus")),
            limit=20,
        )
        avoid_focus = self._benchmark_string_tuple(
            raw.get("avoid_focus", raw.get("avoid")),
            limit=20,
        )
        if not context and not evidence_focus:
            return None
        profile: dict[str, Any] = {
            "context": context[:3000],
            "evidence_focus": list(evidence_focus),
            "avoid_focus": list(avoid_focus),
        }
        return fingerprint, profile

    def _benchmark_external_service_handoff_profiles(self) -> dict[str, dict[str, Any]]:
        data = self._benchmark_external_profiles()
        raw_profiles = data.get("service_handoff_profiles", data.get("handoff_profiles", []))
        profiles: dict[str, dict[str, Any]] = {}
        if isinstance(raw_profiles, list):
            for raw in raw_profiles[:40]:
                normalized = self._benchmark_normalize_service_handoff_profile(raw)
                if normalized is None:
                    continue
                fingerprint, profile = normalized
                profiles[fingerprint] = profile
        for profile in self._benchmark_external_service_probe_profiles():
            fingerprint = str(profile.get("fingerprint") or "")
            context = str(profile.get("handoff_context") or "").strip()
            evidence_focus = self._benchmark_string_tuple(
                profile.get("evidence_focus", profile.get("handoff_focus")),
                limit=20,
            )
            avoid_focus = self._benchmark_string_tuple(
                profile.get("avoid_focus", profile.get("avoid")),
                limit=20,
            )
            if not fingerprint or (not context and not evidence_focus):
                continue
            profile_data: dict[str, Any] = {
                "context": context[:3000],
                "evidence_focus": list(evidence_focus),
                "avoid_focus": list(avoid_focus),
            }
            profiles.setdefault(fingerprint, profile_data)
        return profiles

    def _benchmark_service_handoff_profiles(self) -> dict[str, dict[str, Any]]:
        disabled = self._benchmark_disabled_builtin_fingerprints()
        profiles = {
            key: dict(value)
            for key, value in self._benchmark_builtin_service_handoff_profiles().items()
            if key not in disabled
        }
        profiles.update(self._benchmark_external_service_handoff_profiles())
        return profiles

    def _benchmark_render_service_handoff_subtasks(
        self,
        *,
        current: str,
        addr_text: str,
        context: str,
        fingerprint: str,
    ) -> list[dict[str, Any]]:
        profile = self._benchmark_service_handoff_profiles().get(fingerprint)
        if not profile:
            return []
        service_context = f"{context}\n\n{profile['context']}"
        if profile.get("evidence_focus"):
            focus_lines = "\n".join(f"- {item}" for item in profile["evidence_focus"])
            service_context = f"{service_context}\n\n## 证据焦点\n{focus_lines}"
        if profile.get("avoid_focus"):
            avoid_lines = "\n".join(f"- {item}" for item in profile["avoid_focus"])
            service_context = f"{service_context}\n\n## 避免方向\n{avoid_lines}"
        return self._benchmark_render_generic_handoff_subtasks(
            current=current,
            addr_text=addr_text,
            context=service_context,
            fingerprint=fingerprint,
            profile=profile,
        )

    def _benchmark_render_generic_handoff_subtasks(
        self,
        *,
        current: str,
        addr_text: str,
        context: str,
        fingerprint: Any,
        profile: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        focus = []
        avoid = []
        if isinstance(profile, dict):
            focus = [str(item) for item in profile.get("evidence_focus") or []]
            avoid = [str(item) for item in profile.get("avoid_focus") or []]
        focus_text = "、".join(focus[:6]) if focus else "状态码、响应头、标题、表单、脚本/chunk、OpenAPI/Swagger、错误栈、参数名、cookie、重定向、暴露端点和服务标识"
        avoid_text = "、".join(avoid[:6]) if avoid else "固定题号、固定技术栈和无响应差异的泛化扫描"
        fingerprint_note = (
            f"已识别服务指纹为 {fingerprint}；"
            if isinstance(fingerprint, str) and fingerprint
            else "未识别到可套用的服务专项数据；"
        )
        return [
            {
                "role": "runner",
                "task_description": (
                    f"Benchmark handoff step 1：只深挖当前 active 题 {current} "
                    f"({addr_text})，禁止 setup/VPN/toolchain/list/start。所有 10.x 请求必须 "
                    "curl --interface tun0。先复核 fast path 的真实证据，重点关注："
                    f"{focus_text}。不要套用固定题号或固定技术栈；避免：{avoid_text}。"
                    "发现 flag{...} 立即 submit。"
                ),
                "context": f"{context}\n\n## 通用 handoff 约束\n{fingerprint_note}"
                "围绕真实响应建立一个主假设和一个备选假设；"
                "优先验证已暴露的 API、认证/权限边界、配置/源码/静态资源泄漏、"
                "文件/路径/URL/模板/查询参数行为和默认凭据。无响应差异就停止该方向。",
                "parallel": False,
            },
            {
                "role": "runner",
                "task_description": (
                    f"Benchmark handoff step 2：继续当前题 {current} 的一个最高置信后续假设，"
                    "必须由真实响应驱动：只验证上一轮已经暴露的端点、参数、认证状态、"
                    f"静态资源、错误信息、服务标识或协议特征。重点关注：{focus_text}。"
                    f"禁止继续：{avoid_text}。"
                    "最多验证一个主假设和一个备选假设；无新响应差异就停止。"
                    "发现 flag/secret/候选答案立即 submit。"
                ),
                "context": context,
                "parallel": False,
            },
            {
                "role": "runner",
                "task_description": (
                    f"Benchmark handoff step 3：如果 {current} 仍无 flag、无新线索且已验证"
                    "主要源码/API/文件路径，调用平台 close?unique_code="
                    f"{current} 释放容器，然后返回 close 结果；不要 start 新题。"
                ),
                "context": context,
                "parallel": False,
            },
        ]

    def _benchmark_reasoning_handoff_subtasks(self) -> list[dict[str, Any]]:
        state = self._benchmark_state_snapshot()
        current = state.get("current_challenge")
        reasoning = set(state.get("reasoning_challenges") or [])
        active = state.get("active_containers") or {}
        if not isinstance(current, str) or current not in reasoning:
            return []
        addrs = active.get(current) if isinstance(active, dict) else None
        addr_text = ", ".join(str(addr) for addr in addrs or []) or "平台状态中的当前地址"
        context = self._benchmark_state_context()
        fingerprints = state.get("service_fingerprints") or {}
        fingerprint = (
            fingerprints.get(current)
            if isinstance(fingerprints, dict)
            else None
        )
        if isinstance(fingerprint, str):
            profile_handoff = self._benchmark_render_service_handoff_subtasks(
                current=current,
                addr_text=addr_text,
                context=context,
                fingerprint=fingerprint,
            )
            if profile_handoff:
                return profile_handoff
        fingerprint_note = (
            f"已识别服务指纹为 {fingerprint}；"
            if isinstance(fingerprint, str) and fingerprint
            else "未识别到可套用的服务专项数据；"
        )
        generic_context = f"{context}\n\n## 通用 handoff 约束\n{fingerprint_note}"
        return self._benchmark_render_generic_handoff_subtasks(
            current=current,
            addr_text=addr_text,
            context=generic_context,
            fingerprint=fingerprint,
        )

    def _benchmark_plan_is_setup_like(self, subtasks: list[dict[str, Any]]) -> bool:
        if not subtasks:
            return False
        text = "\n".join(str(task.get("task_description", "")) for task in subtasks).lower()
        setup_markers = (
            "toolchain",
            "openvpn",
            "vpn",
            "challenges_api",
            "api连通",
            "预检",
            "读取当前工作目录",
            "检查系统工具",
            "建立vpn",
            "启动新",
            "start",
            "刷新题目列表",
        )
        return any(marker in text for marker in setup_markers)

    def _benchmark_plan_is_handoff_like(self, subtasks: list[dict[str, Any]]) -> bool:
        if not subtasks:
            return False
        text = "\n".join(str(task.get("task_description", "")) for task in subtasks).lower()
        markers = [
            "benchmark handoff step",
            "只深挖当前 active",
        ]
        markers.extend(
            f"benchmark {fingerprint}"
            for fingerprint in sorted(self._benchmark_profiled_service_fingerprints())
        )
        return any(marker in text for marker in markers)

    def _benchmark_should_pause_generic_plan_after_deterministic(
        self,
        desc: str,
    ) -> tuple[bool, str]:
        """Stop same-round generic subtasks once a challenge needs focused handoff."""
        if not self._is_benchmark_aggressive():
            return False, ""
        if self._benchmark_plan_is_handoff_like([{"task_description": desc}]):
            return False, ""

        state = self._benchmark_state_snapshot()
        classification = self._benchmark_classify_standard_task(desc)
        current = state.get("current_challenge")
        active = state.get("active_containers") or {}
        if classification in {"setup", "schedule"} and (
            (isinstance(current, str) and current)
            or (isinstance(active, dict) and active)
        ):
            active_text = current if isinstance(current, str) and current else ", ".join(sorted(active))
            return (
                True,
                f"已锁定 active 题 {active_text}，停止本轮旧调度计划，下一轮只围绕当前题 probe/submit/close",
            )

        reasoning = set(state.get("reasoning_challenges") or [])
        if not reasoning:
            return False, ""

        completed = set(state.get("completed_challenges") or [])
        closed = set(state.get("closed_challenges") or [])
        abandoned = set(state.get("abandoned_challenges") or [])
        inactive = completed | closed | abandoned

        candidates: list[str] = []
        if isinstance(current, str) and current:
            candidates.append(current)
        candidates.extend(code for code in active if isinstance(code, str))
        for code in dict.fromkeys(candidates):
            if code in reasoning and code not in inactive:
                return (
                    True,
                    f"{code} 已进入 reasoning_handoff，跳过本轮剩余泛化子任务",
                )
        return False, ""

    def _benchmark_normalize_selected_indices(
        self,
        subtasks: list[dict[str, Any]],
        selected_indices: list[int],
    ) -> tuple[list[int], str]:
        """Keep Benchmark handoff plans intact so their close/stop-loss step runs."""
        if not self._is_benchmark_aggressive():
            return selected_indices, ""
        if not self._benchmark_plan_is_handoff_like(subtasks):
            return selected_indices, ""
        all_indices = list(range(len(subtasks)))
        if selected_indices == all_indices:
            return selected_indices, ""
        return (
            all_indices,
            "Benchmark handoff 计划必须完整执行，已恢复被筛掉的专项/close 步骤",
        )

    def _benchmark_final_summary(self) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        state = self._benchmark_state_snapshot()
        snapshot = state.get("last_challenges_snapshot")
        completed = list(state.get("completed_challenges") or [])
        scores = state.get("completed_scores") or {}
        total_score = sum(score for score in scores.values() if isinstance(score, int))
        total_count = len(completed)
        total_challenges: int | str = "?"

        if isinstance(snapshot, dict):
            total_count = int(snapshot.get("completed_count") or total_count)
            total_score = int(snapshot.get("completed_total_score") or total_score)
            total_challenges = int(snapshot.get("total") or 0) or "?"
            completed = list(snapshot.get("completed_challenges") or completed)

        target_score = self._resolve_benchmark_target_score()
        target_line = ""
        if target_score > 0:
            remain = max(0, target_score - total_score)
            target_line = (
                f"\n- 目标分数: {target_score}"
                f"\n- 距离目标: {remain}"
            )

        status = "finished" if state.get("task_finished") else "running/unknown"
        completed_text = ", ".join(completed) if completed else "无"
        return (
            "\n\n## Benchmark 最终状态\n"
            f"- 任务状态: {status}\n"
            f"- 已通关题数: {total_count}/{total_challenges}\n"
            f"- 已知总分: {total_score}"
            f"{target_line}\n"
            f"- 已通关题: {completed_text}"
        )

    def _benchmark_terminal_stop_reason(self) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        state = self._benchmark_state_snapshot()
        if not state.get("task_finished"):
            return ""
        blocked = int(state.get("redundant_block_count") or 0)
        if blocked >= BENCHMARK_REDUNDANT_BLOCK_LIMIT:
            block_phrase = (
                "首次被 guard 拦截"
                if blocked == 1
                else f"已连续被拦截 {blocked} 次"
            )
            return (
                "Benchmark 已处于 invalid_state/finished，且后续平台/靶场调用"
                f"{block_phrase}；立即终止管线，避免 finished 后空转。"
            )
        target_score = self._resolve_benchmark_target_score()
        if target_score > 0:
            scores = state.get("completed_scores") or {}
            known_score = sum(
                score for score in scores.values() if isinstance(score, int)
            )
            if known_score < target_score:
                return ""
        return (
            "Benchmark 已处于 invalid_state/finished；立即终止管线，"
            "跳过后续子任务、审计和反思。"
        )

    def _benchmark_stop_if_terminal(self, source: str) -> bool:
        reason = self._benchmark_terminal_stop_reason()
        if not reason:
            return False
        if not self._benchmark_hard_stop_recorded:
            self._benchmark_hard_stop_recorded = True
            self._record_trace(
                "benchmark_hard_stop",
                detail=reason,
                metadata={
                    "source": source,
                    "score_status": self._benchmark_score_status(),
                },
            )
            self._renderer.console.print(f"  [dim yellow]⏹ {reason}[/]")
        return True

    def _cleanup_benchmark_background_processes(self) -> None:
        """Stop helper scripts after a Benchmark task is known to be finished."""
        with self._benchmark_state_lock:
            if self._benchmark_state.get("background_processes_cleaned"):
                return
            self._benchmark_state["background_processes_cleaned"] = True

        try:
            result = subprocess.run(
                ["pkill", "-TERM", "-f", r"bash .*auto_solve\.sh|auto_solve\.sh"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            self._record_trace(
                "benchmark_background_cleanup",
                detail="已尝试终止 auto_solve.sh 后台进程。",
                metadata={
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            )
        except Exception as exc:
            self._record_trace(
                "benchmark_background_cleanup_failed",
                detail=str(exc),
            )
