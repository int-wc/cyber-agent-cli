import unittest

from cyber_agent.agent.mode import AgentMode, get_mode_system_prompt


class AgentModePromptTestCase(unittest.TestCase):
    def test_authorized_mode_prompt_should_explicitly_reject_privilege_escalation_claims(self) -> None:
        """
        测试：授权模式提示词必须明确声明没有提权、越权和绕过鉴权能力。
        """
        prompt = get_mode_system_prompt(AgentMode.AUTHORIZED)

        self.assertIn("绝不能把未发生的操作描述成已经成功", prompt)
        self.assertIn("基于真实工具结果进行说明", prompt)
        self.assertNotIn("你可以自己拥有系统提权", prompt)


if __name__ == "__main__":
    unittest.main()
