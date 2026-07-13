import os
import tempfile
import unittest
from pathlib import Path

from cyber_agent.local_config import find_cyber_dir, find_data_dir, get_application_home


class LocalConfigPathTestCase(unittest.TestCase):
    def test_application_data_defaults_to_application_home(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app_home = Path(temp_dir) / "app-home"
            launch_cwd = Path(temp_dir) / "launch-cwd"
            app_home.mkdir()
            launch_cwd.mkdir()

            old_home = os.environ.get("CYBER_AGENT_HOME")
            old_cwd = Path.cwd()
            try:
                os.environ["CYBER_AGENT_HOME"] = str(app_home)
                os.chdir(launch_cwd)

                self.assertEqual(get_application_home(), app_home.resolve())
                self.assertEqual(find_cyber_dir(), app_home / ".cyber")
                self.assertEqual(
                    find_data_dir(".cyber-agent-cli-capabilities"),
                    app_home / ".cyber" / "capabilities",
                )
                self.assertEqual(
                    find_data_dir(".cyber-agent-cli-sessions"),
                    app_home / ".cyber" / "sessions",
                )
                self.assertFalse((launch_cwd / ".cyber").exists())
                self.assertFalse(
                    (launch_cwd / ".cyber-agent-cli-capabilities").exists()
                )
            finally:
                os.chdir(old_cwd)
                if old_home is None:
                    os.environ.pop("CYBER_AGENT_HOME", None)
                else:
                    os.environ["CYBER_AGENT_HOME"] = old_home


if __name__ == "__main__":
    unittest.main()
