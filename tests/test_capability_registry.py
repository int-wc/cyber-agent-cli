import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cyber_agent.agent.runner import AgentRunner
from cyber_agent.capability_registry import CapabilityRegistry
from cyber_agent.execution_control import ExecutionController
from cyber_agent.tools import get_default_tools


class NoopChatOpenAI:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def bind_tools(self, tools, **kwargs):
        return self

    def stream(self, messages):
        raise AssertionError("当前用例不应触发真实对话流。")


def build_tool_capability_spec(name: str, *, register_as_tool: bool = True) -> dict[str, object]:
    return {
        "name": name,
        "kind": "tool",
        "register_as_tool": register_as_tool,
        "tool_description": "根据请求返回结构化结果。",
        "usage_hint": "当用户需要稳定、可重复的文本处理结果时使用。",
        "quality_checklist": ["结构稳定", "错误可诊断", "便于继续扩展"],
        "smoke_requests": ["示例请求"],
        "tool_python_code": """
def handle_request(request: str, context: str) -> str:
    cleaned_request = request.strip()
    cleaned_context = context.strip() or "无"
    return f"TOOL:{cleaned_request}|{cleaned_context}"
""".strip(),
        "skill_python_code": "",
    }


def build_skill_capability_spec(name: str) -> dict[str, object]:
    return {
        "name": name,
        "kind": "skill",
        "register_as_tool": False,
        "tool_description": "查看该技能的注入摘要。",
        "usage_hint": "当用户希望长期约束输出风格时使用。",
        "quality_checklist": ["提示清晰", "边界明确", "可持续复用"],
        "smoke_requests": [],
        "tool_python_code": "",
        "skill_python_code": """
def build_skill_prompt() -> str:
    return "回答前先给结论，再给步骤，最后补充风险说明。"
""".strip(),
    }


class CapabilityRegistryTestCase(unittest.TestCase):
    def test_registry_can_create_real_code_files_and_expose_runtime_tool(self) -> None:
        """
        测试：生成 capability 后会写出真实代码文件，并作为当前 agent 的工具暴露。
        """
        with TemporaryDirectory() as temp_dir:
            registry = CapabilityRegistry(base_dir=Path(temp_dir))

            with patch.object(
                registry,
                "_generate_capability_spec",
                return_value=build_tool_capability_spec("doc_helper"),
            ), patch.object(
                registry,
                "_audit_capability",
                return_value={
                    "score": 91,
                    "summary": "质量达标",
                    "issues": [],
                    "recommendations": ["可继续收集真实反馈"],
                },
            ):
                capability = registry.create_or_update_capability(
                    name="doc_helper",
                    kind="tool",
                    description="根据用户输入生成标准文档。",
                    register_as_tool=True,
                )
                runtime_tool = next(
                    tool for tool in registry.get_dynamic_tools() if tool.name == "doc_helper"
                )
                tool_result = runtime_tool.invoke(
                    {"request": "生成示例", "context": "额外上下文"}
                )

            self.assertEqual(capability.status, "awaiting_user_feedback")
            self.assertEqual(tool_result, "TOOL:生成示例|额外上下文")
            self.assertTrue(Path(capability.entrypoint_path).exists())
            self.assertTrue(Path(capability.tool_launcher_path).exists())

            capability_file = (
                Path(temp_dir) / ".cyber-agent-cli-capabilities" / "doc_helper.json"
            )
            self.assertTrue(capability_file.exists())
            stored_data = json.loads(capability_file.read_text(encoding="utf-8"))
            self.assertEqual(stored_data["name"], "doc_helper")
            self.assertIn("handle_request", stored_data["source_code"])

    def test_registry_can_create_skill_file_and_inject_skill_prompt(self) -> None:
        """
        测试：skill 类型 capability 会写出真实代码文件，并把提示词注入当前会话。
        """
        with TemporaryDirectory() as temp_dir:
            registry = CapabilityRegistry(base_dir=Path(temp_dir))

            with patch.object(
                registry,
                "_generate_capability_spec",
                return_value=build_skill_capability_spec("review_style"),
            ), patch.object(
                registry,
                "_audit_capability",
                return_value={
                    "score": 93,
                    "summary": "可直接注入",
                    "issues": [],
                    "recommendations": [],
                },
            ):
                capability = registry.create_or_update_capability(
                    name="review_style",
                    kind="skill",
                    description="约束回答结构和风险表达。",
                    register_as_tool=False,
                )

            self.assertEqual(capability.status, "awaiting_user_feedback")
            self.assertIn("先给结论", capability.system_prompt)
            self.assertIn("review_style", registry.build_skill_prompt())
            self.assertTrue(Path(capability.entrypoint_path).exists())
            self.assertTrue(Path(capability.skill_launcher_path).exists())

    def test_registry_tracks_feedback_revision_and_satisfaction(self) -> None:
        """
        测试：能力生成后可根据用户反馈修订真实代码文件，直到显式标记为满意。
        """
        with TemporaryDirectory() as temp_dir:
            registry = CapabilityRegistry(base_dir=Path(temp_dir))
            revised_spec = build_tool_capability_spec("audit_helper")
            revised_spec["tool_python_code"] = """
def handle_request(request: str, context: str) -> str:
    return f"REVISED:{request.strip()}|{context.strip() or '无'}"
""".strip()

            with patch.object(
                registry,
                "_generate_capability_spec",
                side_effect=[
                    build_tool_capability_spec("audit_helper"),
                    revised_spec,
                ],
            ), patch.object(
                registry,
                "_audit_capability",
                side_effect=[
                    {
                        "score": 60,
                        "summary": "边界不够明确",
                        "issues": ["输出格式不稳定"],
                        "recommendations": ["补充边界条件"],
                    },
                    {
                        "score": 95,
                        "summary": "质量达标",
                        "issues": [],
                        "recommendations": ["等待真实使用反馈"],
                    },
                ],
            ):
                first_capability = registry.create_or_update_capability(
                    name="audit_helper",
                    kind="tool",
                    description="帮助审查输出质量。",
                    register_as_tool=True,
                )
                updated_capability = registry.create_or_update_capability(
                    name="audit_helper",
                    kind="tool",
                    description="帮助审查输出质量。",
                    register_as_tool=True,
                    feedback="请补充异常场景和失败处理。",
                )
                updated_status = updated_capability.status
                updated_revision = updated_capability.revision
                updated_feedback_history = list(updated_capability.feedback_history)
                updated_code = Path(updated_capability.entrypoint_path).read_text(
                    encoding="utf-8"
                )
                satisfied_capability = registry.mark_capability_satisfied(
                    "audit_helper",
                    notes="用户确认可以继续使用。",
                )

        self.assertEqual(first_capability.status, "needs_feedback")
        self.assertEqual(updated_status, "awaiting_user_feedback")
        self.assertEqual(updated_revision, 2)
        self.assertIn("请补充异常场景和失败处理。", updated_feedback_history)
        self.assertIn("REVISED:", updated_code)
        self.assertEqual(satisfied_capability.status, "satisfied")
        self.assertIn("用户确认可以继续使用。", satisfied_capability.feedback_history)

    def test_capability_summary_contains_review_and_satisfaction_workflow(self) -> None:
        """
        测试：生成后的摘要会明确引导查看详情、继续修订和满意标记的闭环。
        """
        with TemporaryDirectory() as temp_dir:
            registry = CapabilityRegistry(base_dir=Path(temp_dir))

            with patch.object(
                registry,
                "_generate_capability_spec",
                return_value=build_tool_capability_spec("workflow_helper"),
            ), patch.object(
                registry,
                "_audit_capability",
                return_value={
                    "score": 90,
                    "summary": "质量达标",
                    "issues": [],
                    "recommendations": ["继续收集用户反馈"],
                },
            ):
                capability = registry.create_or_update_capability(
                    name="workflow_helper",
                    kind="tool",
                    description="用于测试生成后工作流。",
                    register_as_tool=True,
                )
                summary = registry._render_capability_summary(capability)

        self.assertEqual(capability.status, "awaiting_user_feedback")
        self.assertIn("show_generated_capability", summary)
        self.assertIn("revise_generated_capability", summary)
        self.assertIn("mark_generated_capability_satisfied", summary)

    def test_registry_refresh_callback_injects_generated_tool_into_current_runner(self) -> None:
        """
        测试：生成 capability 后会刷新当前 runner 的工具列表，实现即时注入。
        """
        with TemporaryDirectory() as temp_dir:
            execution_controller = ExecutionController()
            registry = CapabilityRegistry(
                base_dir=Path(temp_dir),
                execution_controller=execution_controller,
            )

            with patch("cyber_agent.agent.runner.ChatOpenAI", NoopChatOpenAI):
                runner = AgentRunner(
                    get_default_tools(
                        execution_controller=execution_controller,
                        capability_registry=registry,
                    ),
                    execution_controller=execution_controller,
                    capability_registry=registry,
                )

            registry.register_refresh_callback(runner._refresh_runtime_scope)

            with patch.object(
                registry,
                "_generate_capability_spec",
                return_value=build_tool_capability_spec("inject_helper"),
            ), patch.object(
                registry,
                "_audit_capability",
                return_value={
                    "score": 92,
                    "summary": "可直接使用",
                    "issues": [],
                    "recommendations": [],
                },
            ):
                registry.create_or_update_capability(
                    name="inject_helper",
                    kind="tool",
                    description="在当前会话中即时注入一个工具。",
                    register_as_tool=True,
                )

        self.assertIn("inject_helper", [tool.name for tool in runner.tools])


if __name__ == "__main__":
    unittest.main()
