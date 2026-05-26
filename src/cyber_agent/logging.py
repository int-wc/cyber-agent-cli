"""结构化日志模块，覆盖关键执行路径，支持文件持久化与级别控制。

配置方式（优先级：env > 默认值）：
  CYBER_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR    # 默认 INFO
  CYBER_LOG_DIR=<目录路径>                    # 默认 ~/.cyber-agent-cli-logs/
  CYBER_LOG_ENABLED=true|false                # 默认 true
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_logger: logging.Logger | None = None
_log_lock = threading.Lock()

DEFAULT_LOG_DIR = Path.home() / ".cyber-agent-cli-logs"
LOG_FILENAME_PREFIX = "cyber-agent"


class _StructuredFormatter(logging.Formatter):
    """JSON 行格式，便于后续接入日志聚合与检索。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            payload["exc"] = str(record.exc_info[1])
        if record.args and isinstance(record.args, dict):
            for key in ("elapsed_ms", "tool_name", "model", "service", "char_count", "token_count"):
                value = record.args.get(key)
                if value is not None:
                    payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def _resolve_log_level() -> int:
    """从环境变量解析日志级别，无效值时回退到 INFO。"""
    import os
    raw_level = os.getenv("CYBER_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, raw_level, logging.INFO)


def _resolve_log_dir() -> Path:
    """解析日志落盘目录，默认 ~/.cyber-agent-cli-logs/。"""
    import os
    raw_dir = os.getenv("CYBER_LOG_DIR", "").strip()
    return Path(raw_dir).expanduser() if raw_dir else DEFAULT_LOG_DIR


def _resolve_log_enabled() -> bool:
    """判断是否启用日志落盘。"""
    import os
    raw_enabled = os.getenv("CYBER_LOG_ENABLED", "true").strip().lower()
    return raw_enabled not in {"0", "false", "no", "off", "disable", "disabled"}


def _build_logger() -> logging.Logger:
    """构建日志记录器，同时输出到文件和 stderr。"""
    logger = logging.getLogger("cyber-agent")
    logger.setLevel(_resolve_log_level())
    logger.propagate = False

    if logger.handlers:
        return logger

    if _resolve_log_enabled():
        log_dir = _resolve_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = log_dir / f"{LOG_FILENAME_PREFIX}-{today}.log"
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_StructuredFormatter())
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter(
        "[%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(console_handler)

    return logger


def get_logger() -> logging.Logger:
    """返回全局共享的日志记录器。"""
    global _logger
    if _logger is not None:
        return _logger
    with _log_lock:
        if _logger is not None:
            return _logger
        _logger = _build_logger()
        return _logger


# ── 便捷函数，便于在关键路径直接调用 ──

def log_model_call(
    service: str,
    model: str,
    *,
    char_count: int = 0,
    token_count: int = 0,
    elapsed_ms: float = 0.0,
    success: bool = True,
    error: str = "",
) -> None:
    """记录模型调用关键指标。"""
    logger = get_logger()
    extra: dict[str, Any] = {
        "elapsed_ms": round(elapsed_ms, 1),
        "service": service,
        "model": model,
        "char_count": char_count,
        "token_count": token_count,
    }
    if success:
        logger.info("模型调用成功", extra)
    else:
        extra["exc"] = error
        logger.error(f"模型调用失败: {error}", extra)


def log_tool_execution(
    tool_name: str,
    *,
    elapsed_ms: float = 0.0,
    success: bool = True,
    error: str = "",
    result_len: int = 0,
) -> None:
    """记录工具执行情况。"""
    logger = get_logger()
    extra: dict[str, Any] = {
        "tool_name": tool_name,
        "elapsed_ms": round(elapsed_ms, 1),
        "result_len": result_len,
    }
    if success:
        logger.info("工具执行成功", extra)
    else:
        extra["exc"] = error
        logger.error(f"工具执行失败 [{tool_name}]: {error}", extra)


def log_context_compression(
    *,
    before_chars: int = 0,
    after_chars: int = 0,
    compressed_count: int = 0,
    method: str = "model",
) -> None:
    """记录上下文压缩操作。"""
    get_logger().info(
        "上下文压缩完成",
        {
            "before_chars": before_chars,
            "after_chars": after_chars,
            "compressed_count": compressed_count,
            "method": method,
        },
    )


def log_capability_operation(
    operation: str,
    name: str,
    *,
    kind: str = "",
    audit_score: int = 0,
    elapsed_ms: float = 0.0,
    success: bool = True,
    error: str = "",
) -> None:
    """记录动态 capability 生命周期操作。"""
    logger = get_logger()
    extra: dict[str, Any] = {
        "operation": operation,
        "name": name,
        "kind": kind,
        "audit_score": audit_score,
        "elapsed_ms": round(elapsed_ms, 1),
    }
    if success:
        logger.info(f"capability {operation}", extra)
    else:
        extra["exc"] = error
        logger.error(f"capability {operation} 失败 [{name}]: {error}", extra)


def log_error(module: str, message: str, exc: Exception | None = None) -> None:
    """通用错误记录。"""
    extra: dict[str, Any] = {"module": module}
    if exc is not None:
        extra["exc"] = str(exc)
    get_logger().error(message, extra)


def log_warning(module: str, message: str) -> None:
    """通用警告记录。"""
    get_logger().warning(message, {"module": module})


def log_info(module: str, message: str) -> None:
    """通用信息记录。"""
    get_logger().info(message, {"module": module})
