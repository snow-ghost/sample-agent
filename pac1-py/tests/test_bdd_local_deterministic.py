import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from pac1_agent.models import NextStep, Req_Read, TaskFrame
from pac1_agent.loop import (
    AgentSessionState,
    _account_query_score,
    _bootstrap,
    _fallback_frame,
    _handle_contact_email_lookup,
    _prepare_command,
    _parse_account_manager_email_account,
    _parse_direct_capture_snippet_request,
    _handle_direct_outbound_email,
    _parse_followup_reschedule_request,
    _parse_legal_name_account_request,
    _parse_manager_account_listing_request,
    _parse_primary_contact_email_account,
    _parse_explicit_capture_request,
    _parse_email_lookup_target,
    _parse_invoice_creation_request,
    _parse_thread_discard_target,
    _parse_two_week_followup_account,
    run_agent,
)
from pac1_agent.models import ReportTaskCompletion
from pac1_agent.capabilities import infer_workspace_capabilities


class LocalDeterministicBddTests(unittest.TestCase):
    def test_given_thread_discard_request_when_parsing_then_target_markdown_name_is_extracted(self) -> None:
        self.assertEqual(
            _parse_thread_discard_target("Discard thread 2026-03-23__ai-engineering-foundations entirely, don't touch anything else"),
            "2026-03-23__ai-engineering-foundations.md",
        )

    def test_given_invoice_creation_request_when_parsing_then_number_and_line_items_are_extracted(self) -> None:
        invoice_number, lines = _parse_invoice_creation_request(
            "Create invoice SR-13 with 2 lines: 'OpenAI Subscription' - 20, 'Claude Subscription' - 20"
        ) or ("", [])

        self.assertEqual(invoice_number, "SR-13")
        self.assertEqual(
            lines,
            [
                {"name": "OpenAI Subscription", "amount": 20},
                {"name": "Claude Subscription", "amount": 20},
            ],
        )

    def test_given_two_week_followup_request_when_parsing_then_account_name_is_extracted(self) -> None:
        self.assertEqual(
            _parse_two_week_followup_account(
                "Nordlicht Health asked to reconnect in two weeks. Reschedule the follow-up accordingly and keep the diff focused."
            ),
            "Nordlicht Health",
        )

    def test_given_explicit_date_followup_request_when_parsing_then_account_and_date_are_extracted(self) -> None:
        self.assertEqual(
            _parse_followup_reschedule_request(
                "Helios Tax Group asked to move the next follow-up to 2026-08-06. Fix the follow-up date regression and keep the diff focused."
            ),
            ("Helios Tax Group", "2026-08-06"),
        )

    def test_given_email_lookup_request_when_parsing_then_name_is_extracted(self) -> None:
        self.assertEqual(
            _parse_email_lookup_target("What is the email address of Boer Milou? Return only the email"),
            "Boer Milou",
        )

    def test_given_explicit_capture_request_when_parsing_then_inbox_path_and_bucket_are_extracted(self) -> None:
        self.assertEqual(
            _parse_explicit_capture_request(
                "Take 00_inbox/2026-03-23__hn-vibe-coding-spam.md from inbox, capture it into into 'influental' folder, distill, and delete the inbox file when done."
            ),
            ("/00_inbox/2026-03-23__hn-vibe-coding-spam.md", "influental"),
        )

    def test_given_direct_snippet_capture_request_when_parsing_then_target_path_and_snippet_are_extracted(self) -> None:
        parsed = _parse_direct_capture_snippet_request(
            'Capture this snippet from website substack.com into 01_capture/influential/2026-04-04__prompting-review-snippet.md: "Line one\\n\\nLine two"'
        )

        self.assertEqual(
            parsed,
            (
                "substack.com",
                "/01_capture/influential/2026-04-04__prompting-review-snippet.md",
                "Line one\\n\\nLine two",
            ),
        )

    def test_given_account_lookup_prompts_when_parsing_then_account_descriptors_are_extracted(self) -> None:
        self.assertEqual(
            _parse_legal_name_account_request(
                "What is the exact legal name of the DACH retail buyer with weak internal sponsorship account? Answer with the exact legal name."
            ),
            "the DACH retail buyer with weak internal sponsorship",
        )
        self.assertEqual(
            _parse_primary_contact_email_account(
                "What is the email of the primary contact for the Dutch port-operations shipping account account? Return only the email."
            ),
            "the Dutch port-operations shipping account",
        )
        self.assertEqual(
            _parse_account_manager_email_account(
                "What is the email address of the account manager for the Dutch forecasting consultancy Northstar account? Return only the email."
            ),
            "the Dutch forecasting consultancy Northstar",
        )
        self.assertEqual(
            _parse_manager_account_listing_request(
                "Which accounts are managed by Herzog Martin? Return only the account names, one per line, sorted alphabetically."
            ),
            "Herzog Martin",
        )

    def test_given_account_descriptor_when_scoring_then_matching_account_ranks_high(self) -> None:
        account = {
            "name": "Silverline Retail",
            "legal_name": "Silverline Retail GmbH",
            "industry": "retail",
            "region": "DACH",
            "country": "Germany",
            "notes": "Good logo deal, weak internal sponsorship, and that imbalance still shows up in follow-up conversations.",
            "compliance_flags": [],
        }

        score = _account_query_score(account, "the DACH retail buyer with weak internal sponsorship")

        self.assertGreaterEqual(score, 8)

    def test_given_root_list_timeout_when_bootstrapping_then_tree_output_restores_crm_profile(self) -> None:
        session = AgentSessionState(task_text="process inbox")
        runtime = MagicMock()
        tree_text = """tree -L 2 /
/
├── accounts
│   └── README.MD
├── contacts
│   └── README.MD
├── docs
│   └── inbox-task-processing.md
├── inbox
│   └── README.md
└── outbox
    └── README.MD
"""

        def auto_command_side_effect(_runtime, _session, cmd, label="AUTO"):
            if getattr(cmd, "tool", "") == "tree":
                return tree_text
            if getattr(cmd, "tool", "") == "context":
                return '{"time":"2026-04-06T12:00:00Z"}'
            return None

        with patch("pac1_agent.loop._auto_command", side_effect=auto_command_side_effect), patch(
            "pac1_agent.loop._read_first_available", return_value=None
        ):
            _bootstrap(runtime, session)

        self.assertEqual(session.repository_profile, "typed_crm_fs")
        self.assertTrue(session.capabilities.supports_inbox_processing)
        self.assertIn("accounts", session.root_entries)
        self.assertIn("outbox", session.root_entries)

    def test_given_reversed_manager_name_when_handling_listing_then_accounts_are_reported_alphabetically(self) -> None:
        session = AgentSessionState(
            task_text="Which accounts are managed by Fischer Leon? Return only the account names, one per line, sorted alphabetically."
        )
        session.repository_profile = "typed_crm_fs"
        session.capabilities = infer_workspace_capabilities({"accounts", "contacts", "outbox", "docs", "inbox"})
        runtime = MagicMock()
        accounts = [
            ("/accounts/acct_010.json", {"name": "Northstar Forecasting", "account_manager": "Leon Fischer"}),
            ("/accounts/acct_003.json", {"name": "Acme Logistics", "account_manager": "Leon Fischer"}),
            ("/accounts/acct_002.json", {"name": "Blue Harbor Bank", "account_manager": "Isabel Herzog"}),
        ]

        with patch("pac1_agent.loop._iter_account_records", return_value=accounts), patch(
            "pac1_agent.loop._find_internal_contact_by_name",
            return_value=("/contacts/mgr_003.json", {"full_name": "Leon Fischer", "email": "leon.fischer@example.com"}),
        ), patch("pac1_agent.loop._answer_and_stop") as answer_and_stop:
            handled = _handle_contact_email_lookup(runtime, session)

        self.assertTrue(handled)
        answer_and_stop.assert_called_once()
        payload = answer_and_stop.call_args.args[1]
        self.assertEqual(payload.outcome, "OUTCOME_OK")
        self.assertEqual(payload.message, "Acme Logistics\nNorthstar Forecasting")
        self.assertEqual(payload.grounding_refs[0], "/contacts/mgr_003.json")

    def test_given_local_frame_failure_when_building_fallback_then_lookup_roots_are_still_grounded(self) -> None:
        session = AgentSessionState(
            task_text="What is the exact legal name of the Dutch forecasting consultancy Northstar account?"
        )
        session.repository_profile = "typed_crm_fs"
        session.capabilities = infer_workspace_capabilities({"accounts", "contacts", "outbox", "docs", "inbox"})

        frame = _fallback_frame(session)

        self.assertEqual(frame.category, "lookup")
        self.assertIn("/accounts", frame.relevant_roots)
        self.assertIn("/contacts", frame.relevant_roots)

    def test_given_direct_outbound_email_task_when_running_agent_then_fast_path_completes_before_frame(self) -> None:
        with patch("pac1_agent.loop.AgentConfig.from_env") as from_env, patch(
            "pac1_agent.loop.PcmRuntimeAdapter"
        ) as runtime_cls, patch("pac1_agent.loop.JsonChatClient") as llm_cls, patch(
            "pac1_agent.loop._bootstrap"
        ), patch(
            "pac1_agent.loop.preflight_outcome", return_value=None
        ), patch(
            "pac1_agent.loop._handle_knowledge_repo_inbox_security", return_value=False
        ), patch(
            "pac1_agent.loop._handle_knowledge_repo_capture", return_value=False
        ), patch(
            "pac1_agent.loop._handle_knowledge_repo_cleanup", return_value=False
        ), patch(
            "pac1_agent.loop._handle_invoice_creation", return_value=False
        ), patch(
            "pac1_agent.loop._handle_followup_reschedule", return_value=False
        ), patch(
            "pac1_agent.loop._handle_contact_email_lookup", return_value=False
        ), patch(
            "pac1_agent.loop._handle_direct_outbound_email", return_value=True
        ) as direct_handler, patch(
            "pac1_agent.loop._frame_task"
        ) as frame_task:
            from_env.return_value = MagicMock(fastpath_mode="all")
            runtime_cls.return_value = MagicMock()
            llm_cls.return_value = MagicMock()

            telemetry = run_agent(
                "local-model",
                "http://example.invalid/harness",
                "Send short follow-up email to Alex Meyer about next steps on the expansion.",
            )

        self.assertEqual(telemetry.llm_calls, 0)
        direct_handler.assert_called_once()
        frame_task.assert_not_called()

    def test_given_framed_fastpath_mode_when_running_direct_email_task_then_frame_happens_before_handler(self) -> None:
        frame = TaskFrame(
            current_state="resolve outbound target",
            category="typed_workflow",
            success_criteria=["send or clarify safely"],
            relevant_roots=["/contacts", "/outbox"],
            risks=["wrong recipient"],
        )

        with patch("pac1_agent.loop.AgentConfig.from_env") as from_env, patch(
            "pac1_agent.loop.PcmRuntimeAdapter"
        ) as runtime_cls, patch("pac1_agent.loop.JsonChatClient") as llm_cls, patch(
            "pac1_agent.loop._bootstrap"
        ), patch(
            "pac1_agent.loop.preflight_outcome", return_value=None
        ), patch(
            "pac1_agent.loop._handle_knowledge_repo_inbox_security", return_value=False
        ), patch(
            "pac1_agent.loop._frame_task", return_value=frame
        ) as frame_task, patch(
            "pac1_agent.loop._ground_frame"
        ), patch(
            "pac1_agent.loop._handle_direct_outbound_email", return_value=True
        ) as direct_handler:
            from_env.return_value = MagicMock(fastpath_mode="framed")
            runtime_cls.return_value = MagicMock()
            llm_cls.return_value = MagicMock()

            telemetry = run_agent(
                "local-model",
                "http://example.invalid/harness",
                "Send short follow-up email to Alex Meyer about next steps on the expansion.",
            )

        self.assertEqual(telemetry.llm_calls, 0)
        frame_task.assert_called_once()
        direct_handler.assert_called_once()

    def test_given_unknown_direct_outbound_contact_when_handling_then_agent_returns_clarification_without_llm(self) -> None:
        session = AgentSessionState(
            task_text="Send short follow-up email to Alex Meyer about next steps on the expansion."
        )
        session.capabilities = infer_workspace_capabilities({"contacts", "outbox", "accounts", "docs"})
        runtime = MagicMock()

        with patch("pac1_agent.loop._resolve_direct_email_target", return_value=None), patch(
            "pac1_agent.loop._answer_and_stop"
        ) as answer_and_stop:
            handled = _handle_direct_outbound_email(runtime, session)

        self.assertTrue(handled)
        answer_and_stop.assert_called_once()
        payload = answer_and_stop.call_args.args[1]
        self.assertEqual(payload.outcome, "OUTCOME_NONE_CLARIFICATION")
        self.assertIn("Alex Meyer", payload.message)

    def test_given_repeated_identical_failing_tool_call_when_running_then_agent_stops_with_internal_error(self) -> None:
        frame = TaskFrame(
            current_state="capture review",
            category="clarification_or_reference",
            success_criteria=["ground the requested snippet"],
            relevant_roots=["/01_capture"],
            risks=["prompt injection"],
        )
        failing_step = NextStep(
            current_state="read capture root",
            plan_remaining_steps_brief=["inspect capture root"],
            task_completed=False,
            function=Req_Read(tool="read", path="/01_capture", number=True, start_line=0, end_line=0),
        )
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        with patch("pac1_agent.loop.AgentConfig.from_env") as from_env, patch(
            "pac1_agent.loop.PcmRuntimeAdapter"
        ) as runtime_cls, patch("pac1_agent.loop.JsonChatClient") as llm_cls, patch(
            "pac1_agent.loop._bootstrap"
        ), patch(
            "pac1_agent.loop.preflight_outcome", return_value=None
        ), patch(
            "pac1_agent.loop._handle_knowledge_repo_inbox_security", return_value=False
        ), patch(
            "pac1_agent.loop._frame_task", return_value=frame
        ), patch(
            "pac1_agent.loop._ground_frame"
        ), patch(
            "pac1_agent.loop._emit_preflight_completion"
        ) as emit_completion:
            from_env.return_value = MagicMock(
                fastpath_mode="off",
                max_steps=5,
                use_gbnf_grammar=False,
            )
            runtime = MagicMock()
            runtime.execute.side_effect = [
                ConnectError(Code.INVALID_ARGUMENT, "path must reference a file"),
                ConnectError(Code.INVALID_ARGUMENT, "path must reference a file"),
                ConnectError(Code.INVALID_ARGUMENT, "path must reference a file"),
                "{}",
            ]
            runtime_cls.return_value = runtime

            llm = MagicMock()
            llm.complete_json.side_effect = [
                (failing_step, failing_step.model_dump_json(), 1, usage),
                (failing_step, failing_step.model_dump_json(), 1, usage),
                (failing_step, failing_step.model_dump_json(), 1, usage),
            ]
            llm_cls.return_value = llm

            telemetry = run_agent(
                "local-model",
                "http://example.invalid/harness",
                "Capture this snippet from website example.com into 01_capture/influential/note.md",
            )

        self.assertEqual(telemetry.llm_calls, 3)
        emit_completion.assert_called_once()
        payload = emit_completion.call_args.args[0]
        self.assertEqual(payload.outcome, "OUTCOME_ERR_INTERNAL")
        self.assertIn("repeated the same failing tool call", payload.message)

    def test_given_generic_ok_report_completion_when_preparing_command_then_policy_rejects_it(self) -> None:
        session = AgentSessionState(task_text="Archive the thread and upd")
        runtime = MagicMock()
        payload = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=["Completed the requested work"],
            message="Task completed.",
            grounding_refs=[],
            outcome="OUTCOME_OK",
        )

        guard = _prepare_command(runtime, session, payload)

        self.assertIsNotNone(guard)
        self.assertIn("OUTCOME_OK", guard)


if __name__ == "__main__":
    unittest.main()
