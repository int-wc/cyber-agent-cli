import json
from collections.abc import Callable
from pathlib import Path

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from ..config import settings
from ..execution_control import ExecutionController, ExecutionInterruptedError
from ..tools import get_default_tools, resolve_allowed_roots, resolve_command_registry
from .approval import ApprovalDecision
from .mode import AgentMode, get_mode_system_prompt

AgentEventHandler = Callable[[str, object], None]
ApprovalHandler = Callable[[BaseTool, dict], ApprovalDecision]
MAX_TOOL_ITERATIONS = 12


def extract_text_content(content: str | list[str | dict]) -> str:
    """从 LangChain 消息内容结构中提取纯文本。"""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "".join(parts)


def normalize_tool_args(tool_call: dict) -> dict:
    """将模型返回的工具参数规范化为字典，兼容字符串形式的 JSON 参数。"""
    raw_args = tool_call.get("args", {})
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        stripped_args = raw_args.strip()
        if not stripped_args:
            return {}
        try:
            parsed_args = json.loads(stripped_args)
        except json.JSONDecodeError as exc:
            raise ValueError(f"工具参数不是合法 JSON：{exc}") from exc
        if not isinstance(parsed_args, dict):
            raise ValueError("工具参数必须是 JSON 对象。")
        return parsed_args
    raise ValueError("工具参数格式无效，必须为对象或 JSON 字符串。")


def iter_stream_characters(content: str) -> list[str]:
    """将流式文本统一拆成单字符事件，避免不同模型分片粒度影响终端逐字输出。"""
    return list(content)


class AgentRunner:
    def __init__(
        self,
        tools,
        system_prompt: str | None = None,
        mode: AgentMode = AgentMode.STANDARD,
        allowed_roots: list[Path] | None = None,
        command_registry: dict[str, Path] | None = None,
        extra_allowed_paths: list[Path] | None = None,
        configured_registry: dict[str, Path] | None = None,
        execution_controller: ExecutionController | None = None,
    ):
        self.service = settings.get_service()
        self.llm = ChatOpenAI(**settings.get_chat_openai_kwargs(self.service))
        self.mode = mode
        self.extra_allowed_paths = extra_allowed_paths or []
        self.configured_registry = configured_registry or {}
        self.execution_controller = execution_controller or ExecutionController()
        self.allowed_roots = allowed_roots or resolve_allowed_roots(
            mode,
            self.extra_allowed_paths,
        )
        self.command_registry = command_registry or resolve_command_registry(
            mode,
            self.configured_registry,
        )
        self.tools = tools
        self.system_prompt = system_prompt or get_mode_system_prompt(self.mode)
        self.base_messages: list[BaseMessage] = []
        self.history: list[BaseMessage] = []
        self.reset()

    def _refresh_runtime_scope(self) -> None:
        """按当前模式与授权配置重建可用工具和访问范围。"""
        self.allowed_roots = resolve_allowed_roots(self.mode, self.extra_allowed_paths)
        self.command_registry = resolve_command_registry(
            self.mode,
            self.configured_registry,
        )
        self.tools = get_default_tools(
            self.mode,
            self.extra_allowed_paths,
            self.configured_registry,
            self.execution_controller,
        )

    def reset(self) -> None:
        """重置会话上下文，便于开始一轮新的交互。"""
        self.base_messages = [SystemMessage(content=self.system_prompt)]
        self.history = list(self.base_messages)

    def get_turn_count(self) -> int:
        """返回当前会话中用户已发起的轮次。"""
        return sum(isinstance(message, HumanMessage) for message in self.history)

    def get_history_snapshot(self) -> list[BaseMessage]:
        """返回当前会话消息副本，供上下文查看和持久化复用。"""
        return list(self.history)

    def restore_history(self, messages: list[BaseMessage]) -> None:
        """使用已保存历史恢复当前会话上下文。"""
        if not messages:
            raise ValueError("恢复历史时消息列表不能为空。")
        if not isinstance(messages[0], SystemMessage):
            raise ValueError("恢复历史时首条消息必须是系统消息。")

        self.base_messages = [messages[0]]
        self.system_prompt = extract_text_content(messages[0].content)
        self.history = list(messages)

    def switch_mode(self, mode: AgentMode) -> None:
        """
        切换运行模式时同步刷新系统提示词。
        同时重建工具范围与权限范围，切换后直接清空会话上下文。
        """
        self.mode = mode
        self._refresh_runtime_scope()
        self.system_prompt = get_mode_system_prompt(mode)
        self.reset()

    def add_allowed_path(self, path: Path | str) -> tuple[Path, bool]:
        """
        在授权模式下为当前会话增加允许访问的目录根路径。
        返回规范化后的目录路径，以及本次是否新增成功。
        """
        if self.mode is not AgentMode.AUTHORIZED:
            raise ValueError("仅授权模式支持添加允许访问目录。")

        return self.register_allowed_path(path)

    def register_allowed_path(self, path: Path | str) -> tuple[Path, bool]:
        """
        记录额外允许访问目录，并立即刷新运行时工具范围。
        该能力既用于当前会话内动态授权，也用于从本地配置恢复持久化目录。
        """
        target_path = Path(path).expanduser().resolve()
        if not target_path.exists():
            raise ValueError(f"目录不存在：{target_path}")
        if not target_path.is_dir():
            raise ValueError(f"目标路径不是目录：{target_path}")
        if target_path in self.extra_allowed_paths or target_path in self.allowed_roots:
            return target_path, False

        self.extra_allowed_paths.append(target_path)
        self._refresh_runtime_scope()
        return target_path, True

    def _build_tool_registry(self) -> dict[str, BaseTool]:
        return {tool.name: tool for tool in self.tools}

    def _stream_model_response(
        self,
        messages: list[BaseMessage],
        event_handler: AgentEventHandler | None,
    ) -> AIMessage:
        self.execution_controller.ensure_not_cancelled()
        llm_with_tools = self.llm.bind_tools(self.tools, parallel_tool_calls=False)
        accumulated_chunk: AIMessageChunk | None = None

        if event_handler is not None:
            event_handler("response_begin", None)

        for chunk in llm_with_tools.stream(messages):
            self.execution_controller.ensure_not_cancelled()
            accumulated_chunk = chunk if accumulated_chunk is None else accumulated_chunk + chunk
            token_text = extract_text_content(chunk.content)
            if token_text and event_handler is not None:
                for character in iter_stream_characters(token_text):
                    event_handler("response_token", character)

        if accumulated_chunk is None:
            raise RuntimeError("模型未返回任何响应分片。")

        accumulated_text = extract_text_content(accumulated_chunk.content)
        has_tool_calls = bool(accumulated_chunk.tool_calls)
        if event_handler is not None:
            event_handler(
                "response_end",
                {"content": accumulated_text, "has_tool_calls": has_tool_calls},
            )

        return AIMessage(
            content=accumulated_chunk.content,
            additional_kwargs=accumulated_chunk.additional_kwargs,
            response_metadata=accumulated_chunk.response_metadata,
            tool_calls=list(accumulated_chunk.tool_calls),
            invalid_tool_calls=list(accumulated_chunk.invalid_tool_calls),
        )

    def _tool_requires_approval(self, tool: BaseTool) -> bool:
        tool_risk = self._get_tool_risk(tool)
        return tool_risk in {"write", "execute"}

    def _get_tool_risk(self, tool: BaseTool) -> str:
        """读取工具风险级别，兼容当前工具元数据结构。"""
        tool_metadata = tool.metadata or {}
        return str(tool_metadata.get("risk", "read"))

    def _invoke_tool(
        self,
        tool_call: dict,
        tool_registry: dict[str, BaseTool],
        approval_handler: ApprovalHandler | None,
        event_handler: AgentEventHandler | None,
    ) -> ToolMessage:
        tool_name = str(tool_call.get("name", ""))
        tool = tool_registry.get(tool_name)
        if tool is None:
            return ToolMessage(
                content=f"❌ 未知工具：{tool_name}",
                name=tool_name or "unknown",
                tool_call_id=str(tool_call.get("id", "")),
            )

        self.execution_controller.ensure_not_cancelled()

        if self._tool_requires_approval(tool):
            if event_handler is not None:
                event_handler(
                    "approval_request",
                    {
                        "tool_name": tool_name,
                        "tool_call": tool_call,
                        "risk": self._get_tool_risk(tool),
                    },
                )

            if approval_handler is None:
                decision = ApprovalDecision(
                    approved=False,
                    reason="未配置审批处理器，已拒绝高风险工具调用。",
                )
            else:
                decision = approval_handler(tool, tool_call)

            if event_handler is not None:
                event_handler(
                    "approval_result",
                    {
                        "tool_name": tool_name,
                        "approved": decision.approved,
                        "reason": decision.reason,
                    },
                )

            if not decision.approved:
                return ToolMessage(
                    content=f"❌ 审批未通过：{decision.reason}",
                    name=tool_name,
                    tool_call_id=str(tool_call.get("id", "")),
                )

        try:
            self.execution_controller.ensure_not_cancelled()
            tool_result = tool.invoke(normalize_tool_args(tool_call))
            normalized_result = str(tool_result)
        except ExecutionInterruptedError:
            raise
        except ValueError as exc:
            normalized_result = f"❌ 工具参数错误：{exc}"
        except Exception as exc:
            normalized_result = f"❌ 工具执行异常：{exc}"

        if event_handler is not None:
            event_handler(
                "tool_result",
                {
                    "tool_name": tool_name,
                    "content": normalized_result,
                },
            )

        return ToolMessage(
            content=normalized_result,
            name=tool_name,
            tool_call_id=str(tool_call.get("id", "")),
        )

    def run(
        self,
        user_input: str,
        verbose: bool = True,
        event_handler: AgentEventHandler | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> str:
        """运行一次对话，并在会话内保留上下文。"""
        if not user_input.strip():
            return ""

        self.execution_controller.begin_run()
        try:
            if event_handler is not None:
                event_handler("turn_start", {"input": user_input})
            elif verbose:
                print("开始处理用户输入...")

            self.history.append(HumanMessage(content=user_input))
            tool_registry = self._build_tool_registry()

            for _ in range(MAX_TOOL_ITERATIONS):
                self.execution_controller.ensure_not_cancelled()
                ai_message = self._stream_model_response(self.history, event_handler)
                self.history.append(ai_message)

                if ai_message.tool_calls:
                    if event_handler is not None:
                        event_handler("tool_call", ai_message.tool_calls)

                    for tool_call in ai_message.tool_calls:
                        self.execution_controller.ensure_not_cancelled()
                        tool_message = self._invoke_tool(
                            tool_call,
                            tool_registry,
                            approval_handler,
                            event_handler,
                        )
                        self.history.append(tool_message)
                    continue

                final_response = extract_text_content(ai_message.content)
                if event_handler is None and verbose:
                    print(f"最终回复: {final_response}")
                return final_response

            raise RuntimeError("智能体在多轮工具调用后仍未收敛，请检查提示词或工具结果。")
        finally:
            self.execution_controller.finish_run()
