#!/usr/bin/env python3
"""原语工作流管线单轮挖洞驱动脚本（真实模型 + 真实工具）。

用于在非交互环境验证 PrimitiveWorkflowPipeline 迁移成果：
- 复用 build_runtime_context / create_runner（与 CLI 同一套上下文构建）
- 强制 pipeline_mode=primitive、multi_agent_enabled=True、auto_decision=True
- 限 max_iterations 控制执行闭环轮数（默认 2，避免首次验证失控烧 token）
- 完整 stdout/日志写入 scripts/logs/primitive_hunt_<时间戳>.log

用法:
  python3 scripts/run_primitive_hunt.py "任务描述" [--max-iterations 2]

示例:
  python3 scripts/run_primitive_hunt.py "对补天厂商理想汽车 api-app.lixiang.com 做漏洞挖掘"
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from datetime import datetime
from pathlib import Path

# 允许以源码方式直接运行（未 pip install -e）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description="原语工作流单轮挖洞")
    ap.add_argument("task", nargs="?", default="", help="挖洞任务描述（含厂商与目标）")
    ap.add_argument("--task-file", help="从文件读取任务描述（避免 shell 转义问题）")
    ap.add_argument("--max-iterations", type=int, default=2, help="执行闭环最大轮数，默认 2")
    ap.add_argument("--no-save", action="store_true", help="不保存日志，直接打印")
    args = ap.parse_args()

    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8").strip()
    else:
        task = args.task
    if not task:
        ap.error("必须提供任务描述（task 或 --task-file）")

    from cyber_agent.agent.mode import AgentMode
    from cyber_agent.agent.approval import ApprovalPolicy
    from cyber_agent.cli.interactive import InteractionUiMode
    from cyber_agent.cli.app import build_runtime_context, create_runner
    from cyber_agent.cli.app_multi_agent import PIPELINE_MODE_KEY
    from cyber_agent.cli.app_multi_agent import _run_multi_agent_turn

    # 限制执行闭环轮数，避免首次验证失控（pydantic 模型需 object.__setattr__）
    from cyber_agent.config import settings
    object.__setattr__(
        settings,
        "pipeline_max_iterations",
        max(1, min(args.max_iterations, 20)),
    )

    # 代理可选（与当前 claude code 一致，默认直连）；
    # 旧版 require_proxy_url 强制 opencode 必须配代理，现已在 config 中放宽为 resolve_proxy_url。
    # 如需走代理，在 .env 中配置 OPENCODE_PROXY_URL / MODEL_PROXY_URL 即可，无需在此绕行。

    runtime_context = build_runtime_context(
        mode=AgentMode.AUTHORIZED,
        allow_paths=[],
        tool_specs=[],
        approval_policy=ApprovalPolicy.AUTO,
        ui_mode=InteractionUiMode.CLI,
        auto_decision=True,
    )
    runtime_context["multi_agent_enabled"] = True
    runtime_context[PIPELINE_MODE_KEY] = "primitive"

    runner = create_runner(runtime_context)

    # 日志落盘：捕获运行输出（管线 console 输出 + 任务头），
    # 完整结构化轨迹由管线自行写入 ~/.cyber-agent-cli-traces/<sid>.trace.json
    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"primitive_hunt_{ts}.log"

    print(f"🧬 原语工作流挖洞开始: {task[:120]}...")
    print(f"   max_iterations={args.max_iterations}, 日志={log_file}")
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            _run_multi_agent_turn(task, runner, runtime_context)
    except KeyboardInterrupt:
        print("\n⏹ 用户中断")
        captured.write("\n[用户中断]\n")
    finally:
        if not args.no_save:
            log_file.write_text(
                f"# 任务: {task}\n\n"
                f"# 运行时间: {datetime.now().isoformat(timespec='seconds')}\n"
                f"# 完整结构化轨迹: ~/.cyber-agent-cli-traces/<session_id>.trace.json\n"
                f"{'-' * 60}\n\n"
                f"{captured.getvalue()}",
                encoding="utf-8",
            )
    print(f"\n✅ 完成。日志: {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
