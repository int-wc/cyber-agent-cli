import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

try:
    from langchain_openai import ChatOpenAI

    LANGCHAIN_OPENAI_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - 是否安装依赖由运行环境决定
    ChatOpenAI = None
    LANGCHAIN_OPENAI_IMPORT_ERROR = exc

from ..capability_registry import CapabilityRegistry
from ..config import settings
from ..execution_control import ExecutionController, ExecutionInterruptedError
from ..openai_compat import (
    ensure_deepseek_reasoning_content_compat,
    prepare_messages_for_openai_compatible_service,
)
from ..tools import get_default_tools, resolve_allowed_roots, resolve_command_registry
from .approval import ApprovalDecision
from .mode import AgentMode, get_mode_system_prompt

AgentEventHandler = Callable[[str, object], None]
ApprovalHandler = Callable[[BaseTool, dict], ApprovalDecision]
MAX_TOOL_ITERATIONS = 48
MAX_IDENTICAL_TOOL_ROUNDS = 3
MAX_CYCLIC_TOOL_PATTERN_LENGTH = 4
MAX_TOOL_RESULT_SIGNATURE_CHARS = 400
MAX_MODEL_STREAM_START_ATTEMPTS = 2
MODEL_STREAM_START_RETRY_DELAY_SECONDS = 0.3
MAX_EMPTY_FINAL_RESPONSE_RETRIES = 1
EMPTY_FINAL_RESPONSE_ERROR = (
    "模型返回空最终回复：本轮没有新的工具调用，也没有生成可发送文本。"
)
TRANSIENT_MODEL_STREAM_ERROR_PATTERNS = (
    "empty_stream",
    "upstream stream closed before first payload",
    "internal_server_error",
)


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


def format_message_for_context_summary(message: BaseMessage) -> str:
    """将消息压缩为可用于上下文摘要和调试的文本。"""
    role_label = "system"
    if isinstance(message, HumanMessage):
        role_label = "user"
    elif isinstance(message, AIMessage):
        role_label = "assistant"
    elif isinstance(message, ToolMessage):
        role_label = f"tool:{message.name or 'unknown'}"

    content = extract_text_content(message.content).strip()
    if isinstance(message, AIMessage) and message.tool_calls and not content:
        content = f"工具调用: {json.dumps(message.tool_calls, ensure_ascii=False)}"
    if not content:
        content = "（空内容）"
    return f"{role_label}: {content}"


def normalize_tool_signature_text(text: str) -> str:
    """压缩工具结果文本，避免循环检测被长输出干扰。"""
    normalized_text = " ".join(text.split())
    if len(normalized_text) <= MAX_TOOL_RESULT_SIGNATURE_CHARS:
        return normalized_text
    return f"{normalized_text[:MAX_TOOL_RESULT_SIGNATURE_CHARS]}..."


def serialize_tool_args_for_signature(tool_call: dict) -> str:
    """稳定序列化工具参数，便于识别重复调用。"""
    try:
        normalized_args: object = normalize_tool_args(tool_call)
    except ValueError:
        normalized_args = tool_call.get("args", {})

    try:
        return json.dumps(normalized_args, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(normalized_args)


def should_retry_model_stream_start_error(error: Exception) -> bool:
    """判断模型是否命中了可安全重试的首包前断流错误。"""
    normalized_error = str(error).strip().lower()
    if not normalized_error:
        return False
    return any(
        pattern in normalized_error
        for pattern in TRANSIENT_MODEL_STREAM_ERROR_PATTERNS
    )


def build_tool_round_signature(
    tool_calls: list[dict],
    tool_messages: list[ToolMessage],
) -> str:
    """构建单轮工具调用与结果的签名，用于异常循环检测。"""
    round_payload = []
    for tool_call, tool_message in zip(tool_calls, tool_messages):
        round_payload.append(
            {
                "name": str(tool_call.get("name", "")),
                "args": serialize_tool_args_for_signature(tool_call),
                "result": normalize_tool_signature_text(
                    extract_text_content(tool_message.content)
                ),
            }
        )
    return json.dumps(round_payload, ensure_ascii=False, sort_keys=True)


def detect_tool_call_loop(round_signatures: list[str]) -> str | None:
    """检测重复工具调用或重复链路，避免异常循环。"""
    if len(round_signatures) >= MAX_IDENTICAL_TOOL_ROUNDS:
        recent_signatures = round_signatures[-MAX_IDENTICAL_TOOL_ROUNDS:]
        if len(set(recent_signatures)) == 1:
            return (
                "检测到重复工具调用循环：最近 3 轮工具调用与工具结果完全相同，"
                "已主动停止当前轮次。"
            )

    max_pattern_length = min(
        MAX_CYCLIC_TOOL_PATTERN_LENGTH,
        len(round_signatures) // 2,
    )
    for pattern_length in range(2, max_pattern_length + 1):
        recent_signatures = round_signatures[-pattern_length * 2:]
        if recent_signatures[:pattern_length] == recent_signatures[pattern_length:]:
            return (
                "检测到重复工具链循环：最近 "
                f"{pattern_length} 轮工具调用链完整重复了 2 次，已主动停止当前轮次。"
            )
    return None


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
        capability_registry: CapabilityRegistry | None = None,
        service_name: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_context_chars: int | None = None,
        context_keep_recent_messages: int | None = None,
        context_summary_max_chars: int | None = None,
    ):
        self.service = settings.normalize_service_name(service_name)
        self.model_name = settings.get_model_name(model_name, service_name=self.service)
        self.api_key = settings.get_api_key(self.service, api_key=api_key)
        self.base_url = settings.resolve_base_url(self.service, base_url=base_url)
        self.llm: Any | None = None
        self.mode = mode
        self.extra_allowed_paths = extra_allowed_paths or []
        self.configured_registry = configured_registry or {}
        self.execution_controller = execution_controller or ExecutionController()
        self.capability_registry = capability_registry
        self.max_context_chars = max_context_chars or settings.max_context_chars
        self.context_keep_recent_messages = (
            context_keep_recent_messages or settings.context_keep_recent_messages
        )
        self.context_summary_max_chars = (
            context_summary_max_chars or settings.context_summary_max_chars
        )
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
        self.compressed_summary = ""
        self.compressed_message_count = 0
        self.reset()

    def _build_llm(self) -> Any:
        """按当前运行时服务商与模型配置重建模型实例。"""
        if ChatOpenAI is None:
            raise ModuleNotFoundError(
                "缺少 `langchain_openai` 依赖，当前环境无法创建模型客户端。"
            ) from LANGCHAIN_OPENAI_IMPORT_ERROR
        if self.service == "deepseek":
            ensure_deepseek_reasoning_content_compat()
        return ChatOpenAI(
            **settings.get_chat_openai_kwargs(
                self.service,
                model_name=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        )

    def _get_llm(self) -> Any:
        """按需构建模型客户端，避免非模型命令在缺依赖环境下提前失败。"""
        if self.llm is None:
            self.llm = self._build_llm()
        return self.llm

    def update_llm_config(
        self,
        *,
        service_name: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """更新当前会话使用的服务商、模型或基址，并立即重建模型实例。"""
        if service_name is not None:
            self.service = settings.normalize_service_name(service_name)
        if model_name is not None:
            self.model_name = settings.get_model_name(
                model_name,
                service_name=self.service,
            )
        elif service_name is not None:
            self.model_name = settings.get_model_name(service_name=self.service)
        if service_name is not None:
            self.api_key = settings.get_api_key(self.service)
        if service_name is not None or base_url is not None:
            self.base_url = settings.resolve_base_url(
                self.service,
                base_url=base_url,
            )

        self.llm = None
        if self.capability_registry is not None:
            self.capability_registry.update_llm_config(
                service_name=self.service,
                model_name=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
            )

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
            self.capability_registry,
        )

    def reset(self) -> None:
        """重置会话上下文，便于开始一轮新的交互。"""
        self.base_messages = [SystemMessage(content=self.system_prompt)]
        self.history = list(self.base_messages)
        self.compressed_summary = ""
        self.compressed_message_count = 0

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
        self.compressed_summary = ""
        self.compressed_message_count = 0

    def get_model_context_snapshot(self) -> list[BaseMessage]:
        """返回当前模型真正会读取到的上下文快照。"""
        return list(self._build_model_messages())

    def get_context_diagnostics(self) -> dict[str, object]:
        """返回当前完整历史、压缩摘要和模型上下文的调试信息。"""
        model_messages = self._build_model_messages()
        return {
            "history_message_count": len(self.history),
            "model_message_count": len(model_messages),
            "compressed_message_count": self.compressed_message_count,
            "compressed_summary": self.compressed_summary,
            "history_preview": [format_message_for_context_summary(message) for message in self.history],
            "model_preview": [
                format_message_for_context_summary(message) for message in model_messages
            ],
        }

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

    def _compose_system_prompt(self) -> str:
        """按当前模式和已激活 skill 生成模型实际使用的系统提示。"""
        prompt_parts = [self.system_prompt]
        if self.capability_registry is not None:
            skill_prompt = self.capability_registry.build_skill_prompt().strip()
            if skill_prompt:
                prompt_parts.append(skill_prompt)
        return "\n\n".join(part for part in prompt_parts if part.strip())

    def _estimate_context_char_count(self, messages: list[BaseMessage]) -> int:
        """估算当前上下文占用字符数，用于触发压缩。"""
        return sum(len(format_message_for_context_summary(message)) for message in messages)

    def _summarize_messages_for_context(
        self,
        previous_summary: str,
        messages_to_summarize: list[BaseMessage],
    ) -> str:
        """将较早消息增量压缩为新的上下文摘要。"""
        serialized_messages = "\n".join(
            format_message_for_context_summary(message)
            for message in messages_to_summarize
        )
        summary_prompt = """
你是会话上下文压缩器。请将已有摘要与新增消息压缩为一个新的中文摘要。
必须保留：
1. 用户的真实目标、约束和偏好。
2. 已完成的重要动作、搜索结果、文件变更、工具输出和失败原因。
3. 当前仍未解决的问题、待确认项和后续建议。
4. 已创建的 capability、历史会话、路径、命令或关键标识。
输出只要纯文本摘要，不要加标题，不要虚构内容。
""".strip()
        response = self._get_llm().invoke(
            [
                SystemMessage(content=summary_prompt),
                HumanMessage(
                    content=(
                        f"已有摘要:\n{previous_summary or '无'}\n\n"
                        f"新增消息:\n{serialized_messages}"
                    )
                ),
            ]
        )
        summary_text = extract_text_content(response.content).strip()
        if len(summary_text) > self.context_summary_max_chars:
            return f"{summary_text[:self.context_summary_max_chars]}..."
        return summary_text

    def _adjust_compression_boundary(
        self,
        non_system_messages: list[BaseMessage],
        compression_boundary: int,
    ) -> int:
        """避免压缩边界把工具回合切成半截，留下孤立的 ToolMessage。"""
        resolved_boundary = compression_boundary
        while (
            resolved_boundary < len(non_system_messages)
            and isinstance(non_system_messages[resolved_boundary], ToolMessage)
        ):
            resolved_boundary += 1
        return resolved_boundary

    def _ensure_context_window(self) -> None:
        """在模型调用前按阈值压缩较早消息，避免上下文持续无限增长。"""
        non_system_messages = self.history[1:]
        if not non_system_messages:
            return
        if self._estimate_context_char_count(self.history) <= self.max_context_chars:
            return

        if len(non_system_messages) <= 1:
            return

        keep_recent_messages = max(1, self.context_keep_recent_messages)
        compression_boundary = max(1, len(non_system_messages) - keep_recent_messages)
        compression_boundary = self._adjust_compression_boundary(
            non_system_messages,
            compression_boundary,
        )
        if compression_boundary <= self.compressed_message_count:
            return

        messages_to_summarize = non_system_messages[
            self.compressed_message_count:compression_boundary
        ]
        if not messages_to_summarize:
            return

        self.execution_controller.ensure_not_cancelled()
        self.compressed_summary = self._summarize_messages_for_context(
            self.compressed_summary,
            messages_to_summarize,
        )
        self.compressed_message_count = compression_boundary

    def _build_model_messages(self) -> list[BaseMessage]:
        """构建模型实际读取的消息列表，必要时插入压缩摘要。"""
        self._ensure_context_window()
        model_messages: list[BaseMessage] = [
            SystemMessage(content=self._compose_system_prompt())
        ]
        if self.compressed_summary:
            model_messages.append(
                SystemMessage(
                    content=(
                        "以下是更早对话的压缩摘要，请基于它继续保持上下文一致性：\n"
                        f"{self.compressed_summary}"
                    )
                )
            )
        model_messages.extend(self.history[1 + self.compressed_message_count :])
        return prepare_messages_for_openai_compatible_service(
            model_messages,
            self.service,
            deepseek_thinking_enabled=(
                self.service == "deepseek" and settings.is_deepseek_thinking_enabled()
            ),
        )

    def _stream_model_response(
        self,
        event_handler: AgentEventHandler | None,
    ) -> AIMessage:
        self.execution_controller.ensure_not_cancelled()
        messages = self._build_model_messages()
        llm_with_tools = self._get_llm().bind_tools(self.tools, parallel_tool_calls=False)

        if event_handler is not None:
            event_handler("response_begin", None)

        accumulated_chunk: AIMessageChunk | None = None
        for attempt_index in range(1, MAX_MODEL_STREAM_START_ATTEMPTS + 1):
            accumulated_chunk = None
            try:
                for chunk in llm_with_tools.stream(messages):
                    self.execution_controller.ensure_not_cancelled()
                    accumulated_chunk = chunk if accumulated_chunk is None else accumulated_chunk + chunk
                    token_text = extract_text_content(chunk.content)
                    if token_text and event_handler is not None:
                        for character in iter_stream_characters(token_text):
                            event_handler("response_token", character)
            except ExecutionInterruptedError:
                raise
            except Exception as exc:
                can_retry = (
                    accumulated_chunk is None
                    and attempt_index < MAX_MODEL_STREAM_START_ATTEMPTS
                    and should_retry_model_stream_start_error(exc)
                )
                if not can_retry:
                    raise
                time.sleep(MODEL_STREAM_START_RETRY_DELAY_SECONDS)
                continue

            if accumulated_chunk is not None:
                break
            if attempt_index < MAX_MODEL_STREAM_START_ATTEMPTS:
                time.sleep(MODEL_STREAM_START_RETRY_DELAY_SECONDS)
                continue

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
            tool_round_signatures: list[str] = []
            empty_final_response_retries = 0

            for _ in range(MAX_TOOL_ITERATIONS):
                self.execution_controller.ensure_not_cancelled()
                ai_message = self._stream_model_response(event_handler)
                self.history.append(ai_message)

                if ai_message.tool_calls:
                    if event_handler is not None:
                        event_handler("tool_call", ai_message.tool_calls)

                    tool_registry = self._build_tool_registry()
                    round_tool_messages: list[ToolMessage] = []
                    for tool_call in ai_message.tool_calls:
                        self.execution_controller.ensure_not_cancelled()
                        tool_message = self._invoke_tool(
                            tool_call,
                            tool_registry,
                            approval_handler,
                            event_handler,
                        )
                        self.history.append(tool_message)
                        round_tool_messages.append(tool_message)
                    round_signature = build_tool_round_signature(
                        ai_message.tool_calls,
                        round_tool_messages,
                    )
                    loop_error = detect_tool_call_loop(
                        [*tool_round_signatures, round_signature]
                    )
                    tool_round_signatures.append(round_signature)
                    if loop_error is not None:
                        raise RuntimeError(loop_error)
                    continue

                final_response = extract_text_content(ai_message.content)
                if not final_response.strip():
                    self.history.pop()
                    if empty_final_response_retries < MAX_EMPTY_FINAL_RESPONSE_RETRIES:
                        empty_final_response_retries += 1
                        if event_handler is not None:
                            event_handler(
                                "response_retry",
                                {
                                    "reason": EMPTY_FINAL_RESPONSE_ERROR,
                                    "attempt": empty_final_response_retries,
                                },
                            )
                        continue
                    raise RuntimeError(EMPTY_FINAL_RESPONSE_ERROR)
                if event_handler is None and verbose:
                    print(f"最终回复: {final_response}")
                return final_response

            raise RuntimeError(
                "工具调用轮数已达到安全上限 "
                f"({MAX_TOOL_ITERATIONS} 轮)。最近调用仍在变化，为避免无限执行，"
                "当前轮次已被停止。"
            )
        finally:
            self.execution_controller.finish_run()
