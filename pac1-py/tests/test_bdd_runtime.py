import unittest
from unittest.mock import MagicMock

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from pac1_agent.models import ReportTaskCompletion
from pac1_agent.runtime import PcmRuntimeAdapter


class RuntimeBddTests(unittest.TestCase):
    def test_given_transient_bad_gateway_when_executing_answer_then_retry_succeeds(self) -> None:
        adapter = PcmRuntimeAdapter("http://example.invalid")
        adapter.client = MagicMock()
        adapter.retry_attempts = 2
        adapter.retry_delay_seconds = 0
        adapter.client.answer.side_effect = [RuntimeError("Bad Gateway"), MagicMock()]

        payload = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=["Resolved the lookup"],
            message="answer@example.com",
            grounding_refs=["/accounts/acct_009.json"],
            outcome="OUTCOME_OK",
        )

        result = adapter.execute(payload)

        self.assertEqual(result, "{}")
        self.assertEqual(adapter.client.answer.call_count, 2)

    def test_given_transient_deadline_exceeded_when_executing_answer_then_retry_succeeds(self) -> None:
        adapter = PcmRuntimeAdapter("http://example.invalid")
        adapter.client = MagicMock()
        adapter.retry_attempts = 4
        adapter.retry_delay_seconds = 0
        adapter.client.answer.side_effect = [
            ConnectError(Code.DEADLINE_EXCEEDED, "Request timed out"),
            ConnectError(Code.DEADLINE_EXCEEDED, "Request timed out"),
            MagicMock(),
        ]

        payload = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=["Resolved the lookup"],
            message="answer@example.com",
            grounding_refs=["/accounts/acct_009.json"],
            outcome="OUTCOME_OK",
        )

        result = adapter.execute(payload)

        self.assertEqual(result, "{}")
        self.assertEqual(adapter.client.answer.call_count, 3)

    def test_given_non_transient_invalid_argument_when_executing_then_do_not_retry(self) -> None:
        adapter = PcmRuntimeAdapter("http://example.invalid")
        adapter.client = MagicMock()
        adapter.retry_attempts = 2
        adapter.retry_delay_seconds = 0
        adapter.client.answer.side_effect = ConnectError(Code.INVALID_ARGUMENT, "bad answer payload")

        payload = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=["Resolved the lookup"],
            message="answer@example.com",
            grounding_refs=["/accounts/acct_009.json"],
            outcome="OUTCOME_OK",
        )

        with self.assertRaises(ConnectError):
            adapter.execute(payload)

        self.assertEqual(adapter.client.answer.call_count, 1)


if __name__ == "__main__":
    unittest.main()
