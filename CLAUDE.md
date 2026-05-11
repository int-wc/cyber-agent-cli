# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 构建与测试

```bash
# 可编辑安装
pip install -e .

# 运行全部测试
pytest -q

# 运行单个测试文件
pytest tests/test_agent_runner.py -q

# 标准库 unittest 方式
python -m unittest discover tests -v

# search_web 工具需要浏览器依赖
python -m playwright install chromium
```

## 桌面 IDE（`desktop/`)

Tauri v2 + React + TypeScript + Tailwind CSS 实现的跨平台桌面 IDE。

```bash
# 前端开发（Vite HMR，端口 1420）
cd desktop && npm install && npm run dev

# 仅启动后端 API 服务器（不启动桌面应用）
cyber-agent ide-server --port 9876

# 启动完整桌面 IDE（需要 Rust 工具链）
cyber-agent ide

# 构建桌面应用（AppImage / MSI）
cd desktop && ./scripts/build.sh
```

架构：`cyber-agent ide` 启动 FastAPI 后端（`cli/ide_server.py`），然后 Tauri 应用通过 WebSocket 连接 `ws://127.0.0.1:<port>/ws/chat`。
前端状态管理使用 Zustand，与后端通信通过 WebSocket（AI 对话流式传输）和 REST API（文件操作、Git、终端命令）。

源码目录：`src/cyber_agent/`，测试目录：`tests/`，桌面应用目录：`desktop/`。pyproject.toml 中已配置 `pythonpath = ["src"]` 和 `asyncio_default_fixture_loop_scope = "function"`。

## 架构

### Agent 执行循环（`agent/runner.py`、`agent/core.py`）

`AgentRunner` 是核心执行单元。它封装了 LangGraph 的 `StateGraph`，节点为 `agent -> tools -> agent`。当 `langgraph` 不可用时，`agent/core.py` 中的 `_FallbackCompiledGraph` 提供最小执行器，模拟同样的循环，上限 9999 轮。

`AgentRunner.run()` 是主入口：
1. 将 `HumanMessage` 追加到历史记录
2. 通过 `_stream_model_response()` 流式获取模型响应——遇到瞬时流错误会重试一次
3. 如果模型返回 `tool_calls`，通过 `_invoke_tool()` 逐一执行，追加 `ToolMessage` 结果，然后循环
4. 通过 `detect_tool_call_loop()` 检测重复/循环工具调用——连续 3 轮完全相同或 2 个以上子模式重复即停止
5. 返回最终文本，或在超过 `MAX_TOOL_ITERATIONS`（9999）后抛出异常

### 双界面：CLI + TUI（`cli/app.py`、`cli/tui.py`、`cli/interactive.py`）

界面模式由 `--ui auto|tui|cli` 决定。`auto`（默认）会优先尝试 Textual TUI，失败时回退到基于 Rich 的 CLI。TUI 实现在 `cli/tui.py`（Textual 框架）。CLI 使用 `prompt_toolkit` 做带自动补全的输入（`cli/prompting.py`），不可用时降级到 `typer.prompt()`。

`render_agent_event()` 将 `AgentRunner` 事件（turn_start、response_token、tool_call、approval_request 等）映射为 Rich 控制台输出。

`/stop` 支持：CLI 模式下，agent 在守护线程中运行，主线程以非阻塞方式轮询 stdin 读取 `/stop`。`ExecutionController` 管理取消状态并终止已注册的子进程树。

### 模式系统（`agent/mode.py`）

通过 `AgentMode` 枚举定义两种模式：
- `STANDARD`——保守模式，工具访问范围以当前工作目录为主
- `AUTHORIZED`——必须显式声明 `--allow-path` 目录和 `--tool name=绝对路径` 注册的外部工具

通过 `/mode` 或 `AgentRunner.switch_mode()` 切换模式会重建系统提示词、工具集和访问范围，然后清空会话上下文。

### 工具风险与审批管道（`tools/metadata.py`、`agent/approval.py`）

每个工具都有 `metadata["risk"]` 标签：`"read"`、`"write"` 或 `"execute"`。标记为 `write`/`execute` 的工具会触发审批管道。三种 `ApprovalPolicy`：`prompt`（交互确认）、`auto`（自动批准）、`never`（全部拒绝）。

审批处理器根据上下文不同而不同：
- CLI 交互模式：直接 `typer.confirm()` 内联确认
- CLI 后台线程模式：通过 `Queue` 移交主线程展示和确认
- Webhook 模式：非交互（prompt 策略下拒绝所有高风险工具）

### 上下文压缩（`agent/runner.py`）

每次模型调用前都会运行 `_ensure_context_window()`。当字符数超过 `MAX_CONTEXT_CHARS`（默认 14000）或 token 估算超过 `MAX_CONTEXT_TOKENS`（默认 100万）时，通过模型调用将较早消息压缩为 `compressed_summary`，以 `SystemMessage` 形式前置插入。如果压缩素材本身也超出预算，则使用 `_summarize_messages_locally_for_context()` 进行确定性的首尾截断兜底。

`_shrink_model_messages_to_token_budget()` 是第二道防线：总预算仍然超限时，逐条压缩最长的单条消息。

### 动态 capability 系统（`capability_registry.py`）

模型可以在运行时通过 `create_generated_capability` 工具生成新的 skill 或 tool。流程如下：
1. 模型调用 `create_generated_capability(name, kind, description)`——`CapabilityRegistry._generate_capability_spec()` 让模型输出包含 Python 代码的结构化 JSON
2. `_materialize_capability_artifacts()` 将真实 `.py` 文件写入 `.cyber-agent-cli-capabilities/<名称>/capability.py`，同时生成启动脚本
3. 运行烟雾测试执行生成的代码，审计模型打分（0-100）
4. 如果分数达到 `CAPABILITY_AUDIT_MIN_SCORE`（默认 75），状态变为 `awaiting_user_feedback`，否则为 `needs_feedback`
5. capability 作为 LangChain tool（如果 `register_as_tool=True`）或追加的系统提示文本（如果 `kind=skill`）注入 agent
6. 修订走 `revise_generated_capability`，最终由 `mark_generated_capability_satisfied` 标记完成

生成的代码必须定义 `handle_request(request, context) -> str`（工具）或 `build_skill_prompt() -> str`（技能），只能使用 Python 标准库。

### 懒加载策略

重依赖（LangChain、OpenAI SDK、Playwright、Textual、prompt_toolkit）通过 `cli/app.py` 及其他模块中的 `_load_*_support()` 函数按需导入，确保 `--help` 和 `--version` 快速响应。

### OpenAI 兼容层（`openai_compat.py`）

项目通过统一的模型网关 `GATEWAY_BASE_URL`（默认 `http://127.0.0.1:8317/v1`）访问模型。切换服务商（`/service openai|deepseek|claude|mimo`）只改变 `extra_body` 中的 `provider` 字段，基址始终指向网关入口。

`ensure_deepseek_reasoning_content_compat()` 对 langchain-openai 内部做 monkey-patch，在消息转换过程中透传 `reasoning_content`，因为上游 LangChain 未处理此字段。

### Webhook 网关（`cli/webhook.py`、`cli/feishu_long_connection.py`）

`webhook serve` 启动 HTTP 服务，提供飞书、钉钉、企微、邮件四个平台的路由。每个平台路由处理器的流程：
1. 校验签名/验证 token
2. 从平台特定的请求体中提取用户消息
3. 创建 `AgentRunner`（或按会话键复用已有实例），通过 `runner.run()` 处理消息
4. 通过 reply webhook URL 或平台 SDK 发送回复

飞书长连接模式使用飞书 SDK 的 WebSocket 客户端，无需公网回调地址。

### 会话持久化（`session_store.py`）

每轮对话后自动保存会话到 `.cyber-agent-cli-sessions/<id>.json`。消息通过 LangChain 的 `messages_to_dict`/`messages_from_dict` 序列化。搜索（`/history search <关键词>`）对所有会话做全文匹配，并在命中位置生成上下文摘要片段。

### 执行控制（`execution_control.py`）

`ExecutionController` 通过线程事件管理取消状态。`begin_run()`/`finish_run()` 包裹每次 agent 执行。`ensure_not_cancelled()` 在关键边界（模型调用前、每个工具执行前）检查取消状态。通过 `register_process()` 登记的子进程会被 `terminate_process_tree()` 终止（SIGTERM + 回退 SIGKILL，或 Windows 下的 `taskkill /T`）。
