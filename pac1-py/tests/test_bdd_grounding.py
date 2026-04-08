import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from pac1_agent.grounding import auto_command
from pac1_agent.models import Req_Read


class GroundingBddTests(unittest.TestCase):
    def test_given_agent_file_with_startup_reads_when_auto_command_reads_it_then_followup_reads_are_triggered(self) -> None:
        session = SimpleNamespace(
            grounded_agent_paths=set(),
            attempted_agent_paths=set(),
        )
        runtime = MagicMock()
        runtime.execute.return_value = (
            "AGENTS.md\n"
            "# Instructions\n\n"
            "- Always read `/docs/process.md`\n"
            "- Startup read `/README.md`\n"
        )
        appended: list[tuple[str, str]] = []
        startup_reads: list[list[str]] = []

        def append_tool_result(_session, tool_name: str, text: str) -> None:
            appended.append((tool_name, text))

        def run_startup_reads(_runtime, _session, paths: list[str]) -> None:
            startup_reads.append(paths)

        text = auto_command(
            runtime,
            session,
            Req_Read(tool="read", path="/AGENTS.md"),
            append_tool_result=append_tool_result,
            run_startup_reads=run_startup_reads,
            cli_green="",
            cli_yellow="",
            cli_clr="",
        )

        self.assertIn("/AGENTS.md", session.grounded_agent_paths)
        self.assertEqual(text, runtime.execute.return_value)
        self.assertEqual(appended[0][0], "Req_Read")
        self.assertEqual(startup_reads, [["/docs/process.md", "/README.md"]])


if __name__ == "__main__":
    unittest.main()
