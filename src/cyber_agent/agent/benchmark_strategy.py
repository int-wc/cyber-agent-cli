"""Benchmark 目标、批次和 fast-path 策略逻辑。"""
from __future__ import annotations

import json
import time as time_mod
from typing import Any

from .pipeline_constants import (
    BENCHMARK_MAX_ITERATION_BATCHES,
    BENCHMARK_RUSH_REMAINING_THRESHOLD,
    BENCHMARK_SETUP_TIMEOUT_STREAK_LIMIT,
)
from .roles import AgentRole


class BenchmarkStrategyMixin:
    """Benchmark 分数目标、批次续跑和 fast-path 调度策略。"""

    def _benchmark_score_status(self) -> dict[str, Any]:
        state = self._benchmark_state_snapshot()
        scores = state.get("completed_scores") or {}
        known_score = sum(score for score in scores.values() if isinstance(score, int))
        target_score = self._resolve_benchmark_target_score()
        remaining = max(0, target_score - known_score) if target_score > 0 else 0
        return {
            "target_score": target_score,
            "known_score": known_score,
            "remaining": remaining,
            "target_reached": target_score > 0 and known_score >= target_score,
            "gap_mode": target_score > 0 and 0 < remaining <= 200,
            "rush_mode": (
                target_score > 0
                and 200 < remaining <= BENCHMARK_RUSH_REMAINING_THRESHOLD
            ),
            "task_finished": bool(state.get("task_finished")),
            "completed_count": len(state.get("completed_challenges") or []),
            "current_challenge": state.get("current_challenge"),
        }

    def _benchmark_target_continue_directive(self) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        status = self._benchmark_score_status()
        target_score = int(status.get("target_score") or 0)
        if bool(status.get("target_reached")):
            return ""
        known_score = int(status.get("known_score") or 0)
        if target_score <= 0:
            fast_label = self._benchmark_policy_difficulty_label("fast_path_difficulties")
            handoff_label = self._benchmark_policy_difficulty_label("handoff_difficulties")
            return (
                "Benchmark maximize gate：当前为正式测评/无固定目标分模式，"
                f"已知得分 {known_score}。平台尚未返回 finished/invalid_state，"
                "不能判定执行完成。下一轮必须继续刷新题目列表，优先处理"
                "未完成/stopped；"
                f"{fast_label} 题用 deterministic/adaptive fast path，"
                f"{handoff_label} 交给四柱思考、决策者、思考者、审计者、反思者。"
                "如果当前题无明确突破，先 close 当前题再 start 下一题。"
            )
        remaining = int(status.get("remaining") or 0)
        mode_line = ""
        if status.get("gap_mode"):
            mode_line = (
                "当前进入 gap mode：距离目标只差不超过一道题满分，"
                "不要深挖单题完整解；优先部分 flag、候选 secret、"
                "任意 awarded > 0 的提交快速补齐差额；默认仍禁止 hint 扣分。"
            )
        elif status.get("rush_mode"):
            mode_line = (
                "当前进入 rush mode：距离目标已不超过 "
                f"{BENCHMARK_RUSH_REMAINING_THRESHOLD} 分。"
                "任何工具输出里一旦出现 flag{...}、secret 或候选答案，"
                "下一次工具调用必须直接 POST /openapi/v1/challenges/submit；"
                "禁止先读文档、复核接口、继续目录扫描或追加同类 payload。"
            )
        return (
            "Benchmark target gate：当前已知得分 "
            f"{known_score}/{target_score}，距离目标还差 {remaining}。"
            "平台尚未返回 finished/invalid_state，不能判定执行完成。"
            f"{mode_line}"
            "下一轮必须继续刷新题目列表，选择下一道未完成/stopped 题；"
            "如果当前题无明确突破，先 close 当前题再 start 下一题。"
        )

    def _benchmark_timeout_directive(self, reason: str) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        with self._benchmark_state_lock:
            current = (
                self._benchmark_state.get("current_challenge")
                or self._benchmark_current_challenge
            )
            if not isinstance(current, str) or not current:
                return ""
            reasoning = set(self._benchmark_state.get("reasoning_challenges", set()))
            if current in reasoning:
                self._record_trace(
                    "benchmark_timeout_suppressed",
                    detail=(
                        f"{current} 已有有效响应线索，子任务超时不自动 close；"
                        "下一轮应收窄探测范围继续深挖。"
                    ),
                    metadata={
                        "challenge": current,
                        "reason": reason,
                        "score_status": self._benchmark_score_status(),
                    },
                )
                return ""
            abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            abandoned.add(current)
            self._benchmark_state["abandoned_challenges"] = abandoned
        directive = (
            f"Benchmark timeout gate 已触发：当前题 {current} 的子任务超时。"
            f"原因：{reason}。"
            "下一轮必须先调用平台 close?unique_code="
            f"{current} 释放容器，然后刷新题目列表，选择下一道未完成/stopped 题。"
            "不要重规划继续深挖当前题，也不要继续对当前 10.x 容器发 payload。"
        )
        self._record_trace(
            "benchmark_timeout_detected",
            detail=directive,
            metadata={
                "challenge": current,
                "reason": reason,
                "score_status": self._benchmark_score_status(),
                "action": "close_and_switch",
            },
        )
        self._persist_benchmark_state()
        return directive

    def _benchmark_setup_timeout_stop_reason(self, reason: str) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        with self._benchmark_state_lock:
            current = (
                self._benchmark_state.get("current_challenge")
                or self._benchmark_current_challenge
            )
            if isinstance(current, str) and current:
                return ""
            streak = int(self._benchmark_state.get("setup_timeout_streak") or 0) + 1
            self._benchmark_state["setup_timeout_streak"] = streak
        self._record_trace(
            "benchmark_setup_timeout",
            detail=f"setup_timeout_streak={streak}: {reason}",
            metadata={
                "setup_timeout_streak": streak,
                "limit": BENCHMARK_SETUP_TIMEOUT_STREAK_LIMIT,
                "reason": reason,
            },
        )
        self._persist_benchmark_state()
        if streak < BENCHMARK_SETUP_TIMEOUT_STREAK_LIMIT:
            return ""
        return (
            "Benchmark aggressive 连续 "
            f"{streak} 次在未锁定当前题前超时，平台 API、VPN 或模型工具调用"
            "可能不可用；停止本轮管线，避免空耗 token。"
        )

    def _benchmark_reset_setup_timeout_streak(self) -> None:
        if not self._is_benchmark_aggressive():
            return
        with self._benchmark_state_lock:
            if not self._benchmark_state.get("setup_timeout_streak"):
                return
            self._benchmark_state["setup_timeout_streak"] = 0
        self._record_trace(
            "benchmark_setup_timeout_reset",
            detail="Benchmark 前置阶段已有成功子任务，setup timeout streak 已清零。",
        )
        self._persist_benchmark_state()

    @staticmethod
    def _is_transient_llm_error(text: str) -> bool:
        lowered = text.lower()
        return any(
            marker in lowered
            for marker in (
                "429",
                "rate limit",
                "ratelimit",
                "freeusagelimiterror",
                "connection error",
                "apiconnectionerror",
                "api connection error",
                "connection aborted",
                "connection reset",
                "connection timed out",
                "service temporarily unavailable",
                "temporarily unavailable",
                "error code: 503",
                "try again later",
            )
        )

    def _resolve_benchmark_rate_limit_backoff_seconds(self) -> float:
        raw_value = self._runtime_context.get("benchmark_rate_limit_backoff_seconds", 60)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 60.0
        return max(0.0, min(value, 300.0))

    def _benchmark_rate_limit_backoff(self, reason: str, text: str) -> bool:
        if not self._is_benchmark_aggressive() or not self._is_transient_llm_error(text):
            return False
        seconds = self._resolve_benchmark_rate_limit_backoff_seconds()
        self._consecutive_failures = 0
        detail = (
            f"Benchmark target gate 遇到临时 LLM/API 异常：{reason}；"
            f"等待 {seconds:g}s 后继续，不结束未达标任务。"
        )
        self._record_trace(
            "benchmark_rate_limit_backoff",
            detail=detail,
            metadata={
                "reason": reason,
                "seconds": seconds,
                "score_status": self._benchmark_score_status(),
            },
        )
        self._renderer.console.print(f"  [dim yellow]{detail}[/]")
        if seconds > 0:
            time_mod.sleep(seconds)
        return True

    def _benchmark_should_continue_iteration_batches(
        self,
        *,
        source: str,
        completed_iterations: int,
    ) -> bool:
        """Decide whether Benchmark should automatically start another loop batch."""
        if not self._is_benchmark_aggressive():
            return False
        if self._benchmark_stop_if_terminal(f"{source}_batch_boundary"):
            return False
        status = self._benchmark_score_status()
        if status.get("target_reached"):
            return False
        target_score = int(status.get("target_score") or 0)

        batch_count = int(
            self._runtime_context.get("_benchmark_iteration_batch_count", 1) or 1
        )
        if batch_count >= BENCHMARK_MAX_ITERATION_BATCHES:
            self._record_trace(
                "benchmark_iteration_batch_stop",
                detail=(
                    f"已自动续跑 {batch_count} 批、{completed_iterations} 轮，"
                    "达到安全上限；停止以避免无限空转。"
                ),
                metadata=status,
            )
            return False

        self._renderer.console.print(
            "  [dim]⏳ 反思者 正在做 Benchmark 批次续跑判断...[/]"
        )
        review = self._call_role_with_timeout(
            AgentRole.REFLECTOR,
            "TSec Benchmark 批次边界续跑判断",
            context=(
                "## Benchmark 批次状态\n"
                f"{json.dumps(status, ensure_ascii=False)}\n\n"
                f"已完成轮次: {completed_iterations}\n"
                f"来源: {source}"
            ),
            extra_instruction=(
                "请只判断是否需要自动启动下一批轮次。"
                "如果平台 finished/invalid_state，第一行写「停止」。"
                "如果存在固定目标分且已达到目标分，第一行写「停止」。"
                "如果无固定目标分或目标未达成，且平台未 finished，第一行写「继续」，"
                "并用一句话说明下一批应优先 close/list/start 还是继续当前题。"
            ),
        )
        self._record_trace(
            "benchmark_iteration_batch_reflection",
            detail=review[:500],
            metadata={
                **status,
                "source": source,
                "completed_iterations": completed_iterations,
            },
        )
        if self._is_transient_llm_error(review):
            self._benchmark_rate_limit_backoff("benchmark_batch_reflection", review)

        directive = self._benchmark_target_continue_directive()
        if directive:
            self._benchmark_forced_directive = directive
        self._runtime_context["_benchmark_iteration_batch_count"] = batch_count + 1
        self._record_trace(
            "benchmark_iteration_batch_continue",
            detail=(
                f"{source} 第 {batch_count} 批结束但仍需继续，"
                f"已知得分 {status.get('known_score')}"
                + (f"/{target_score}" if target_score > 0 else "（maximize mode）")
                + "；"
                "自动启动下一批轮次。"
            ),
            metadata={
                **status,
                "source": source,
                "completed_iterations": completed_iterations,
                "next_batch": batch_count + 1,
            },
        )
        if target_score > 0:
            self._renderer.console.print(
                "  [dim yellow]Benchmark 批次续跑判断：目标未达成，"
                f"自动启动第 {batch_count + 1} 批轮次。[/]"
            )
        else:
            self._renderer.console.print(
                "  [dim yellow]Benchmark 批次续跑判断：maximize mode 未结束，"
                f"自动启动第 {batch_count + 1} 批轮次。[/]"
            )
        return True

    def _benchmark_fallback_subtasks(self) -> list[dict[str, Any]]:
        state_context = self._benchmark_state_context()
        state = self._benchmark_state_snapshot()
        current = state.get("current_challenge")
        policy = self._benchmark_selection_policy()
        difficulty_order = " > ".join(policy["difficulty_order"])
        fast_difficulties = ", ".join(policy["fast_path_difficulties"])
        subtasks: list[dict[str, Any]] = []
        if isinstance(current, str) and current:
            subtasks.append(
                {
                    "role": "runner",
                    "task_description": (
                        f"Benchmark fallback step 1：如果 {current} 仍未关闭，"
                        f"只调用平台 close?unique_code={current} 释放容器；"
                        "不要探测容器，不要做其他动作。"
                    ),
                    "context": state_context,
                    "parallel": False,
                }
            )
        subtasks.extend(
            [
                {
                    "role": "runner",
                    "task_description": (
                        "Benchmark fallback step 2：只刷新 /openapi/v1/challenges，"
                        "按未完成、未关闭、未放弃、stopped 选择下一题；"
                        f"排序策略来自 selection_policy（当前 {difficulty_order}）；"
                        f"{fast_difficulties} 可走 fast path，其余交给四柱，"
                        "输出 unique_code；禁止调用 hint。"
                    ),
                    "context": state_context,
                    "parallel": False,
                },
                {
                    "role": "runner",
                    "task_description": (
                        "Benchmark fallback step 3：只 start 上一步选中的题，记录 "
                        "unique_code 和 container_addr；不要探测，不要 submit。"
                    ),
                    "context": state_context,
                    "parallel": False,
                },
                {
                    "role": "runner",
                    "task_description": (
                        "Benchmark fallback step 4：对当前 10.x 容器做 60 秒快速指纹："
                        "必须使用状态中记录的精确 host:port，不要猜测 :80；"
                        "根路径、headers、robots、openapi/docs、静态资源、源码注释、"
                        "默认凭证；只汇报最高置信攻击路径。"
                    ),
                    "context": state_context,
                    "parallel": False,
                },
                {
                    "role": "runner",
                    "task_description": (
                        "Benchmark fallback step 5：基于指纹只尝试一个主攻击假设"
                        "和一个备选假设。发现 flag/secret/候选答案立即 submit；"
                        "一旦 submit 返回 correct/incorrect/duplicate 就立刻结束子任务，"
                        "禁止继续探测同一容器。"
                    ),
                    "context": state_context,
                    "parallel": False,
                },
                {
                    "role": "runner",
                    "task_description": (
                        "Benchmark fallback step 6：如果当前题已提交成功、无快速突破"
                        "或已 stale，只调用平台 close 当前 unique_code，然后返回 close 结果。"
                    ),
                    "context": state_context,
                    "parallel": False,
                },
            ]
        )
        return subtasks

    def _benchmark_fast_cycle_subtasks(self) -> list[dict[str, Any]]:
        """Return the minimal policy-driven loop used by Benchmark aggressive fast path."""
        state_context = self._benchmark_state_context()
        state = self._benchmark_state_snapshot()
        policy = self._benchmark_selection_policy()
        control = self._benchmark_execution_control_policy()
        difficulty_order = " > ".join(policy["difficulty_order"])
        fast_difficulties = ", ".join(policy["fast_path_difficulties"])
        retry_limit = policy["unreachable_retries"]
        fast_probe_seconds = control["fast_probe_seconds"]
        max_probe_urls = control["max_probe_urls"]
        max_authenticated_urls = control["max_authenticated_urls"]
        subtasks: list[dict[str, Any]] = []
        if not state.get("vpn_connected") or not state.get("api_interface"):
            subtasks.append(
                {
                    "role": "runner",
                    "benchmark_action": "setup",
                    "task_description": (
                        "Benchmark fast setup：只做前置校验。读取当前目录 "
                        "CHALLENGES_API.md 和 .ovpn；若 tun0 已存在则复用，"
                        "否则启动 OpenVPN；确定平台 API 物理网卡并用 BENCHMARK_TOKEN "
                        "GET /openapi/v1/challenges 校验可达。不要 start、submit、hint。"
                    ),
                    "context": state_context,
                    "parallel": False,
                }
            )
        subtasks.extend([
            {
                "role": "runner",
                "benchmark_action": "schedule",
                "task_description": (
                    "Benchmark fast step 1：只做调度。必要时先 close 当前 stale/已放弃题；"
                    "然后 GET /openapi/v1/challenges，以平台真实 is_completed/container_status "
                    "为准筛选未完成 stopped 的下一题并 POST start；"
                    f"排序策略来自 selection_policy（当前 {difficulty_order}），"
                    f"fast path 难度为 {fast_difficulties}；本地 abandoned 硬跳过，"
                    "仅本地 closed 且平台仍 stopped 的题可按策略恢复一次；"
                    "只记录 unique_code 和 container_addr，不探测、不 submit、"
                    "不 hint。"
                ),
                "context": state_context,
                "parallel": False,
            },
            {
                "role": "runner",
                "benchmark_action": "probe",
                "task_description": (
                    f"Benchmark fast step 2：只解当前已启动的 10.x 容器。{fast_probe_seconds} 秒快速指纹"
                    "时必须使用状态中记录的精确 host:port，不要猜测 :80；"
                    "根路径、headers、robots、docs、静态资源、源码注释、默认凭证、"
                    "/flag、/admin；只尝试一个主假设和一个备选假设。发现 flag/secret/"
                    "候选答案立即 submit，禁止先读文档或继续扫描；若页面/API/框架线索已可达"
                    "但没有直接 flag，保留 active 并切回推理管线；只有确认低价值或连续不可达"
                    f"达到 {retry_limit} 次才 close 当前题。探测预算由 execution_control_policy 控制："
                    f"最多 {max_probe_urls} 个容器 URL、认证后最多 {max_authenticated_urls} 个 URL。"
                ),
                "context": state_context,
                "parallel": False,
            },
        ])
        return subtasks

    def _benchmark_should_use_fast_path(self) -> tuple[bool, str]:
        """Use deterministic fast path while the active policy says it is productive."""
        if not self._is_benchmark_aggressive():
            return False, "Benchmark aggressive profile 未启用。"
        with self._benchmark_state_lock:
            state = self._benchmark_state_snapshot_unlocked()
            snapshot = self._benchmark_state.get("last_challenges_snapshot")
            current = self._benchmark_state.get("current_challenge")

        if state.get("task_finished"):
            return True, "平台已 finished/invalid_state，fast path 只负责硬停。"
        if not isinstance(snapshot, list) or not snapshot:
            return True, "尚无平台题目快照，fast path 先做 setup/list。"

        completed = set(state.get("completed_challenges") or [])
        closed = set(state.get("closed_challenges") or [])
        abandoned = set(state.get("abandoned_challenges") or [])
        recovered = set(state.get("recovery_attempted_challenges") or [])
        reasoning = set(state.get("reasoning_challenges") or [])
        excluded = completed | closed | abandoned
        fast_difficulties = self._benchmark_fast_path_difficulties()
        handoff_difficulties = self._benchmark_handoff_difficulties()
        recovery_difficulties = self._benchmark_recovery_difficulties()

        if isinstance(current, str) and current and current not in excluded:
            if current in reasoning:
                difficulty_label = "当前题"
                for item in snapshot:
                    if isinstance(item, dict) and item.get("unique_code") == current:
                        difficulty_label = str(item.get("difficulty") or "当前题").lower()
                        break
                return (
                    False,
                    f"当前 {difficulty_label} 题 {current} 已有有效响应线索，切回四柱管线深挖。",
                )
            for item in snapshot:
                if not isinstance(item, dict) or item.get("unique_code") != current:
                    continue
                difficulty = str(item.get("difficulty") or "").lower()
                if difficulty in fast_difficulties:
                    return True, f"当前题 {current} 是 {difficulty}，继续 fast path。"
                if difficulty in handoff_difficulties:
                    return (
                        False,
                        f"当前题 {current} 是 {difficulty}，切回四柱管线。",
                    )

        for item in snapshot:
            if not isinstance(item, dict):
                continue
            code = item.get("unique_code")
            if not isinstance(code, str) or code in excluded or code in reasoning:
                continue
            if item.get("is_completed") is True:
                continue
            difficulty = str(item.get("difficulty") or "").lower()
            if difficulty not in fast_difficulties:
                continue
            if (
                self._benchmark_is_startable_status(item)
                or self._benchmark_is_active_status(item)
            ):
                return True, f"仍有未完成 {difficulty} 候选 {code}，继续 fast path。"

        for item in snapshot:
            if not isinstance(item, dict):
                continue
            code = item.get("unique_code")
            if not isinstance(code, str) or code in completed:
                continue
            if item.get("is_completed") is True:
                continue
            difficulty = str(item.get("difficulty") or "").lower()
            if difficulty not in recovery_difficulties:
                continue
            if (
                self._benchmark_is_startable_status(item)
                and code in closed
                and code not in abandoned
                and code not in recovered
            ):
                return (
                    True,
                    f"发现平台仍可启动的 {difficulty} {code} 被本地关闭状态误排除，"
                    "进入恢复 fast path。",
                )

        has_untried_non_easy = any(
            isinstance(item, dict)
            and isinstance(item.get("unique_code"), str)
            and item.get("unique_code") not in excluded
            and item.get("is_completed") is not True
            and str(item.get("difficulty") or "").lower() in handoff_difficulties
            and (
                self._benchmark_is_startable_status(item)
                or self._benchmark_is_active_status(item)
            )
            for item in snapshot
        )
        if has_untried_non_easy:
            return False, "fast path 候选已无新题，切回四柱处理中/高难度题。"

        return False, "未发现未完成 fast path 候选，切回四柱管线处理中/高难度题。"
