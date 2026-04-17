# Cyber Agent CLI

一个面向本地终端的网络安全与代码辅助智能体原型。

它基于 `Typer + Rich/Textual + LangChain/OpenAI` 构建，支持在命令行中与模型对话，并通过受限工具集完成目录浏览、文本文件读写、补丁应用、Shell 命令执行、端口探测等任务。

> 当前版本更适合作为实验性原型、课程项目、个人工具链样例或后续扩展的基础骨架，而不是直接面向生产环境的安全自动化平台。

## 功能特性

- 支持 `CLI` 与 `TUI` 双界面，默认在真实终端中优先使用 `TUI`。
- 提供启动页字符画动画与状态欢迎面板，适合终端交互展示。
- 支持 `standard` 与 `authorized` 两种运行模式。
- 提供 `prompt`、`auto`、`never` 三种高风险工具审批策略。
- 内置文件系统、补丁、端口扫描、Shell 执行等默认工具。
- 支持在授权模式下注册外部工具，并限制为显式声明的绝对路径。
- 支持工作目录级本地配置文件 `.cyber-agent-cli.json` 持久化允许访问目录。
- 提供 `/help`、`/tools`、`/status`、`/mode`、`/approval` 等交互命令。

## 适用场景

- 在受控目录内辅助阅读、修改和补丁化本地代码或文本文件。
- 在本地实验环境中执行简单的网络安全相关操作，例如端口可达性探测。
- 作为 AI Agent CLI、终端交互、审批链路、工具调用边界的教学或实验项目。
- 作为后续扩展更多安全工具、工作流和策略层的基础仓库。

## 技术栈

- Python 3.11+
- Typer
- Rich
- Textual
- prompt_toolkit
- LangChain
- OpenAI API

## 快速开始

### 1. 安装

```bash
git clone <your-repo-url>
cd cyber-agent-cli
python -m venv .venv
pip install -e .
```

### 2. 配置环境变量

可参考仓库中的 `.env.example`，在项目根目录创建 `.env`：

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5.4
OPENAI_BASE_URL=
SERVICE_NAME=openai
```

说明：

- `OPENAI_API_KEY`：必填。
- `OPENAI_MODEL`：可选，默认是 `gpt-5.4`。
- `OPENAI_BASE_URL`：可选，适用于兼容 OpenAI API 的代理服务；当 `SERVICE_NAME=deepseek` 时，会优先使用内置的 `https://api.deepseek.com/v1`，如需覆盖请在 CLI 中使用 `/service deepseek <基址>` 或在代码侧显式传入基址。
- `SERVICE_NAME`：可选，默认是 `openai`，当前支持 `openai`、`deepseek` 以及其他手动指定兼容基址的 OpenAI 兼容服务。

### 3. 启动

```bash
cyber-agent
```

默认行为：

- 直接进入交互式会话。
- 默认模式为 `standard`。
- 默认审批策略为 `prompt`。
- 默认界面模式为 `auto`，会优先尝试 `TUI`。

## 使用示例

### 交互式启动

```bash
cyber-agent --ui cli
```

```bash
cyber-agent --ui tui
```

### 单轮运行

```bash
cyber-agent run "扫描 127.0.0.1 的 80 端口"
```

### 授权模式

```bash
cyber-agent --mode authorized --allow-path ./labs --approval-policy auto
```

### 注册外部工具

```bash
cyber-agent --mode authorized --tool python=/absolute/path/to/python tools
```

## 运行模式

### `standard`

- 默认模式。
- 工具访问范围以当前工作目录为主。
- 适合保守执行、查看状态、进行小范围文件操作或快速试验。

### `authorized`

- 在显式声明的允许路径和已注册外部工具范围内更主动地调用工具。
- 可通过 `--allow-path` 扩展允许访问目录。
- 可通过 `--tool name=absolute_path` 注册外部工具。
- 命令执行继承当前 CLI 进程权限，不会自动提权。

## 审批策略

### `prompt`

- 默认策略。
- 高风险工具调用前进行交互确认。

### `auto`

- 自动批准高风险工具调用。
- 适合受控环境下的快速实验或自动化流程。

### `never`

- 拒绝所有高风险工具调用。
- 适合只读检查、状态查看或演示场景。

高风险工具通常包括：

- 文件写入
- 文本替换
- 补丁应用
- Shell 命令执行
- 外部工具执行

## 默认工具

当前默认启用的工具包括：

| 工具名 | 说明 |
| --- | --- |
| `scan_port` | 扫描指定目标 IP 或域名的指定端口是否开放 |
| `list_directory` | 列出允许访问范围内指定目录的文件和子目录 |
| `read_text_file` | 读取允许访问范围内的文本文件内容 |
| `write_text_file` | 写入文本文件内容 |
| `replace_in_file` | 在文本文件中替换指定片段 |
| `apply_unified_patch` | 应用 unified diff 补丁到允许访问范围内的文本文件 |
| `run_shell_command` | 在受限工作目录内执行 shell 命令 |
| `run_registered_tool` | 执行已注册的外部工具，仅在授权模式且存在工具注册时可用 |

## CLI 命令

顶层命令：

- `cyber-agent`：直接进入交互模式。
- `cyber-agent chat`：进入交互模式，或通过 `--message/-m` 执行单轮对话。
- `cyber-agent run <message>`：执行单轮对话。
- `cyber-agent tools`：查看当前默认启用的工具列表。
- `cyber-agent doctor`：检查运行依赖、配置、允许路径和工具注册状态。

常用选项：

- `--mode standard|authorized`
- `--allow-path <path>`
- `--tool name=absolute_path`
- `--approval-policy prompt|auto|never`
- `--ui auto|tui|cli`

## 交互命令

进入交互界面后，可使用以下内建命令：

- `/help`
- `/tools`
- `/status`
- `/mode`
- `/mode standard`
- `/mode authorized`
- `/config`
- `/config allow-path`
- `/config allow-path add <目录>`
- `/service`
- `/service <服务商>`
- `/service <服务商> <基址>`
- `/model`
- `/model <模型名>`
- `/allow-path`
- `/allow-path add <目录>`
- `/approval`
- `/approval prompt`
- `/approval auto`
- `/approval never`
- `/clear`
- `/exit`

其中：

- `/allow-path add <目录>` 只对当前会话生效。
- `/config allow-path add <目录>` 会把目录写入工作目录下的 `.cyber-agent-cli.json`，供后续会话复用。
- `/service <服务商>` 和 `/model <模型名>` 只对当前会话生效，不会改写 `.env`。

## 本地配置

项目支持工作目录级本地配置文件：

```json
{
  "allow_paths": [
    "/absolute/path/to/labs",
    "/absolute/path/to/data"
  ]
}
```

配置文件名固定为：

```text
.cyber-agent-cli.json
```

它用于持久化授权目录，避免每次启动都重复传入 `--allow-path`。

## 安全边界

这个项目强调“在边界内使用工具”，而不是无约束地执行系统操作。

- 文件系统工具仅能访问允许范围内的目录。
- `run_shell_command` 只能在允许的工作目录中执行命令。
- 外部工具必须显式注册为绝对路径后才能调用。
- 外部工具与 Shell 命令均继承当前 CLI 进程权限，不会自动提权。
- 当前实现不提供提权、绕过鉴权、突破沙箱或未授权访问能力。
- 请仅在合法、受控、已授权的环境中使用。

## 项目结构

```text
src/cyber_agent/
|- agent/        # Agent 运行器、模式、审批与状态管理
|- cli/          # CLI/TUI 界面、渲染、提示词与启动页
|- tools/        # 文件系统、补丁、系统命令、安全探测等工具
|- config.py     # 模型与环境变量配置
`- local_config.py

tests/
|- test_cli_commands.py
|- test_cli_chat_e2e.py
|- test_cli_interactive.py
`- ...
```

## 开发与测试

运行测试：

```bash
python -m unittest discover tests -v
```

查看工具列表：

```bash
cyber-agent tools
```

查看运行诊断：

```bash
cyber-agent doctor
```
