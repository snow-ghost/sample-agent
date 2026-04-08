import unittest
from unittest.mock import patch

from main import _build_totals, _full_run_log_path, _render_summary_table
from pac1_agent.config import AgentConfig, _should_use_gbnf
from pac1_agent.llm import (
    GBNF_NEXT_STEP,
    GBNF_TASK_FRAME,
    JsonChatClient,
    LocalFlatNextStep,
    _coerce_local_next_step,
    _grammar_for_model,
    _normalize_local_next_step_payload,
)
from pac1_agent.models import NextStep, TaskFrame


class OutputAndConfigBddTests(unittest.TestCase):
    def test_given_localhost_base_url_when_detecting_gbnf_then_it_is_enabled(self) -> None:
        self.assertTrue(_should_use_gbnf("http://localhost:8090/v1"))
        self.assertTrue(_should_use_gbnf("http://127.0.0.1:8080/v1"))

    def test_given_remote_base_url_when_detecting_gbnf_then_it_stays_disabled(self) -> None:
        self.assertFalse(_should_use_gbnf("https://foundation-models.api.cloud.ru/v1"))
        self.assertFalse(_should_use_gbnf(None))

    def test_given_default_env_when_building_config_then_fastpath_mode_is_framed(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            config = AgentConfig.from_env("test-model")

        self.assertEqual(config.fastpath_mode, "framed")

    def test_given_task_rows_when_rendering_summary_then_table_contains_metrics_columns(self) -> None:
        table = _render_summary_table(
            [
                {
                    "task_id": "t06",
                    "score": "1.00",
                    "wall_time_ms": 1502,
                    "llm_calls": 0,
                    "llm_time_ms": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                {
                    "task_id": "t17",
                    "score": "1.00",
                    "wall_time_ms": 32866,
                    "llm_calls": 1,
                    "llm_time_ms": 15275,
                    "prompt_tokens": 3670,
                    "completion_tokens": 115,
                    "total_tokens": 3785,
                },
            ]
        )

        self.assertIn("Task", table)
        self.assertIn("Tokens", table)
        self.assertIn("LLM", table)
        self.assertIn("Wall ms", table)
        self.assertIn("LLM ms", table)
        self.assertIn("t06", table)
        self.assertIn("3785", table)

    def test_given_task_rows_when_building_totals_then_aggregate_and_average_metrics_are_returned(self) -> None:
        totals = _build_totals(
            [
                {
                    "task_id": "t06",
                    "score": "1.00",
                    "wall_time_ms": 1500,
                    "llm_calls": 0,
                    "llm_time_ms": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                {
                    "task_id": "t17",
                    "score": "1.00",
                    "wall_time_ms": 3500,
                    "llm_calls": 2,
                    "llm_time_ms": 1200,
                    "prompt_tokens": 3000,
                    "completion_tokens": 100,
                    "total_tokens": 3100,
                },
            ],
            100.0,
        )

        self.assertEqual(totals["tasks_run"], 2)
        self.assertEqual(totals["llm_tasks_run"], 1)
        self.assertEqual(totals["wall_time_ms"], 5000)
        self.assertEqual(totals["llm_calls"], 2)
        self.assertEqual(totals["total_tokens"], 3100)
        self.assertEqual(totals["avg_wall_time_ms"], 2500.0)
        self.assertEqual(totals["avg_llm_calls_per_task"], 1.0)
        self.assertEqual(totals["avg_tokens_per_task"], 1550.0)
        self.assertEqual(totals["avg_llm_time_ms_when_used"], 1200.0)
        self.assertEqual(totals["avg_tokens_when_used"], 3100.0)

    def test_given_full_run_when_resolving_log_path_then_latest_full_run_file_is_used(self) -> None:
        path = _full_run_log_path([])

        self.assertIsNotNone(path)
        self.assertEqual(path.name, "latest_full_run.txt")

    def test_given_partial_run_when_resolving_log_path_then_no_full_log_file_is_written(self) -> None:
        self.assertIsNone(_full_run_log_path(["t06"]))

    def test_given_local_gbnf_config_when_building_llm_request_then_grammar_is_attached(self) -> None:
        client = JsonChatClient(
            AgentConfig(
                model="local-model",
                openai_api_key="local",
                openai_base_url="http://localhost:8090/v1",
                use_gbnf_grammar=True,
            )
        )

        kwargs = client._request_kwargs([{"role": "user", "content": "Return JSON"}], TaskFrame)

        self.assertEqual(kwargs["extra_body"]["grammar"], GBNF_TASK_FRAME)

    def test_given_task_models_when_selecting_grammar_then_schema_specific_gbnf_is_used(self) -> None:
        self.assertEqual(_grammar_for_model(TaskFrame), GBNF_TASK_FRAME)
        self.assertEqual(_grammar_for_model(NextStep), GBNF_NEXT_STEP)

    def test_given_local_flat_report_completion_when_coercing_then_standard_next_step_is_restored(self) -> None:
        flat = LocalFlatNextStep(
            current_state="inbox triaged",
            plan_step="report completion",
            task_completed=True,
            tool="report_completion",
            arg1="Processed inbox",
            arg2="Done",
            arg3="/inbox",
            outcome="OUTCOME_OK",
        )

        step = _coerce_local_next_step(flat)

        self.assertEqual(step.current_state, "inbox triaged")
        self.assertEqual(step.plan_remaining_steps_brief, ["report completion"])
        self.assertTrue(step.task_completed)
        self.assertEqual(step.function.tool, "report_completion")
        self.assertEqual(step.function.completed_steps_laconic, ["Processed inbox"])
        self.assertEqual(step.function.grounding_refs, ["/inbox"])

    def test_given_list_plan_step_and_local_tool_when_normalizing_local_payload_then_values_are_compacted(self) -> None:
        payload = _normalize_local_next_step_payload(
            {
                "current_state": "cleanup",
                "plan_step": ["relevant_roots"],
                "task_completed": False,
                "tool": "local",
            }
        )

        self.assertEqual(payload["plan_step"], "relevant_roots")
        self.assertEqual(payload["tool"], "list")

    def test_given_schema_echo_payload_when_normalizing_local_payload_then_safe_list_fallback_is_used(self) -> None:
        payload = _normalize_local_next_step_payload({"json": "return a valid JSON object"})

        self.assertEqual(payload["tool"], "list")
        self.assertEqual(payload["arg1"], "/")
        self.assertFalse(payload["task_completed"])

    def test_given_previous_flat_read_payload_when_normalizing_local_payload_then_generic_args_are_filled(self) -> None:
        payload = _normalize_local_next_step_payload(
            {
                "current_state": "inspect file",
                "plan_step": "read target",
                "task_completed": False,
                "tool": "read",
                "path": "/docs/README.md",
                "number": True,
                "start_line": 5,
                "end_line": 15,
            }
        )

        self.assertEqual(payload["tool"], "read")
        self.assertEqual(payload["arg1"], "/docs/README.md")
        self.assertTrue(payload["flag1"])
        self.assertEqual(payload["num1"], 5)
        self.assertEqual(payload["num2"], 15)


if __name__ == "__main__":
    unittest.main()
