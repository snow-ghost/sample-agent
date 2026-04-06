import unittest

from pac1_agent.telemetry import AgentRunTelemetry, TokenUsage


class TelemetryBddTests(unittest.TestCase):
    def test_given_openai_usage_dict_when_extracting_tokens_then_fields_are_normalized(self) -> None:
        usage = TokenUsage.from_response_usage(
            {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            }
        )

        self.assertEqual(usage.prompt_tokens, 120)
        self.assertEqual(usage.completion_tokens, 30)
        self.assertEqual(usage.total_tokens, 150)

    def test_given_multiple_llm_calls_when_recording_telemetry_then_time_and_tokens_accumulate(self) -> None:
        telemetry = AgentRunTelemetry()

        telemetry.record_llm_call(2100, TokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140))
        telemetry.record_llm_call(900, TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30))

        self.assertEqual(telemetry.llm_calls, 2)
        self.assertEqual(telemetry.llm_time_ms, 3000)
        self.assertEqual(telemetry.prompt_tokens, 120)
        self.assertEqual(telemetry.completion_tokens, 50)
        self.assertEqual(telemetry.total_tokens, 170)

    def test_given_deterministic_task_when_no_llm_calls_are_recorded_then_telemetry_stays_zeroed(self) -> None:
        telemetry = AgentRunTelemetry()

        self.assertEqual(telemetry.llm_calls, 0)
        self.assertEqual(telemetry.llm_time_ms, 0)
        self.assertEqual(telemetry.total_tokens, 0)


if __name__ == "__main__":
    unittest.main()
