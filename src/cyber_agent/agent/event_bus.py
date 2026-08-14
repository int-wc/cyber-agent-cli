"""事件钩子系统：让能力/外部模块可以观察或拦截 agent 关键事件。

对标 dsh「事件即扩展点」的最小落地：
- AgentRunner 在产生 TOOL_CALL/TOOL_RESULT/APPROVAL_REQUEST 等事件时，
  除原有 event_handler 回调外，还会发布到 EventBus；
- 任何模块（动态能力、审计、外部集成）可 subscribe 观察事件，
  或注册可改写 payload 的拦截器（先观察后拦截，避免破坏现有语义）。

设计约束：
- 单例总线（global_event_bus）+ 可注入实例，测试用临时实例隔离；
- publish 不抛错：单个监听器异常不影响主流程（与现有 event_handler 语义一致）；
- payload 是结构化 dict（只含 JSON 安全字段），不做序列化校验。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("cyber_agent.event_bus")

# 事件监听器签名：handler(event_type: str, payload: dict | None) -> dict | None
# 返回 None 表示不修改；返回 dict 表示改写 payload（拦截器语义）。
EventListener = Callable[[str, dict | None], dict | None]


class EventBus:
    """线程安全的事件订阅/发布总线。"""

    def __init__(self) -> None:
        self._listeners: dict[str, list[EventListener]] = {}
        self._wildcard: list[EventListener] = []
        self._lock = threading.RLock()

    def subscribe(self, event_type: str, handler: EventListener) -> Callable[[], None]:
        """订阅某类事件（event_type 为空字符串表示订阅全部）。返回退订函数。"""
        if not callable(handler):
            raise TypeError("事件监听器必须是可调用对象")

        def _unsubscribe() -> None:
            with self._lock:
                if event_type:
                    bucket = self._listeners.get(event_type)
                    if bucket and handler in bucket:
                        bucket.remove(handler)
                elif handler in self._wildcard:
                    self._wildcard.remove(handler)

        with self._lock:
            if event_type:
                self._listeners.setdefault(event_type, []).append(handler)
            else:
                self._wildcard.append(handler)
        return _unsubscribe

    def publish(self, event_type: str, payload: dict | None = None) -> dict | None:
        """发布事件：先通配监听器，再精确监听器。

        监听器可改写 payload（拦截语义），返回最终 payload；
        单个监听器异常仅记录日志，不中断主流程。
        """
        final_payload = payload
        handlers: list[EventListener] = []
        with self._lock:
            handlers = list(self._wildcard)
            handlers.extend(list(self._listeners.get(event_type, [])))
        for handler in handlers:
            try:
                result = handler(event_type, final_payload)
                if result is not None:
                    final_payload = result
            except Exception as exc:  # noqa: BLE001 - 监听器异常不影响主流程
                logger.warning("event_bus 监听器异常 [%s]: %s", event_type, exc)
        return final_payload

    def listener_count(self, event_type: str = "") -> int:
        """返回某类事件（空=全部）的监听器数量，供测试/诊断。"""
        with self._lock:
            base = len(self._wildcard) if not event_type else 0
            return base + len(self._listeners.get(event_type, []))


# 全局单例：AgentRunner 发布事件的目标；测试可注入临时实例隔离
global_event_bus = EventBus()
