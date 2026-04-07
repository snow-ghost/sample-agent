import unittest

from pac1_agent.capabilities import infer_workspace_capabilities
from pac1_agent.framing import derive_fallback_frame, derive_high_confidence_frame


class FramingBddTests(unittest.TestCase):
    def test_given_explicit_knowledge_capture_request_when_deriving_frame_then_typed_workflow_frame_is_returned(self) -> None:
        frame = derive_high_confidence_frame(
            "Take 00_inbox/2026-03-23__hn-agent-kernel-stateful-agents.md from inbox, capture it into into 'influental' folder, distill, and delete the inbox file when done.",
            "knowledge_repo",
            infer_workspace_capabilities(profile="knowledge_repo"),
        )

        self.assertIsNotNone(frame)
        self.assertEqual(frame.category, "typed_workflow")
        self.assertIn("/00_inbox", frame.relevant_roots)
        self.assertIn("/02_distill", frame.relevant_roots)

    def test_given_invoice_creation_request_when_deriving_frame_then_invoice_surface_is_selected_without_llm(self) -> None:
        frame = derive_high_confidence_frame(
            "Create invoice SR-13 with 2 lines: 'OpenAI Subscription' - 20, 'Claude Subscription' - 20",
            "typed_crm_fs",
            infer_workspace_capabilities(profile="typed_crm_fs"),
        )

        self.assertIsNotNone(frame)
        self.assertEqual(frame.category, "typed_workflow")
        self.assertEqual(frame.relevant_roots, ["/my-invoices"])

    def test_given_follow_up_reschedule_request_when_deriving_frame_then_account_and_reminder_roots_are_selected(self) -> None:
        frame = derive_high_confidence_frame(
            "Nordlicht Health asked to reconnect in two weeks. Reschedule the follow-up accordingly and keep the diff focused.",
            "typed_crm_fs",
            infer_workspace_capabilities(profile="typed_crm_fs"),
        )

        self.assertIsNotNone(frame)
        self.assertEqual(frame.category, "typed_workflow")
        self.assertIn("/accounts", frame.relevant_roots)
        self.assertIn("/reminders", frame.relevant_roots)

    def test_given_direct_outbound_email_request_when_deriving_frame_then_outbox_workflow_is_selected(self) -> None:
        frame = derive_high_confidence_frame(
            'Send email to Blue Harbor Bank with subject "Security review follow-up" and body "Checking in."',
            "typed_crm_fs",
            infer_workspace_capabilities(profile="typed_crm_fs"),
        )

        self.assertIsNotNone(frame)
        self.assertEqual(frame.category, "typed_workflow")
        self.assertIn("/outbox", frame.relevant_roots)
        self.assertIn("/contacts", frame.relevant_roots)

    def test_given_manager_account_listing_request_when_deriving_frame_then_lookup_frame_is_selected(self) -> None:
        frame = derive_high_confidence_frame(
            "Which accounts are managed by Koch Lea? Return only the account names, one per line, sorted alphabetically.",
            "typed_crm_fs",
            infer_workspace_capabilities(profile="typed_crm_fs"),
        )

        self.assertIsNotNone(frame)
        self.assertEqual(frame.category, "lookup")
        self.assertIn("/accounts", frame.relevant_roots)
        self.assertIn("/contacts", frame.relevant_roots)

    def test_given_generic_process_inbox_request_when_deriving_frame_then_llm_frame_is_still_required(self) -> None:
        frame = derive_high_confidence_frame(
            "process inbox",
            "typed_crm_fs",
            infer_workspace_capabilities(profile="typed_crm_fs"),
        )

        self.assertIsNone(frame)

    def test_given_local_frame_failure_for_crm_lookup_when_deriving_fallback_then_lookup_roots_are_still_grounded(self) -> None:
        frame = derive_fallback_frame(
            "What is the exact legal name of the Dutch forecasting consultancy Northstar account?",
            "typed_crm_fs",
            infer_workspace_capabilities(profile="typed_crm_fs"),
        )

        self.assertEqual(frame.category, "lookup")
        self.assertIn("/accounts", frame.relevant_roots)
        self.assertIn("/contacts", frame.relevant_roots)

    def test_given_local_frame_failure_for_knowledge_inbox_when_deriving_fallback_then_security_sensitive_roots_are_selected(self) -> None:
        frame = derive_fallback_frame(
            "Process the oldest inbox item safely.",
            "knowledge_repo",
            infer_workspace_capabilities(profile="knowledge_repo"),
        )

        self.assertEqual(frame.category, "security_sensitive")
        self.assertIn("/00_inbox", frame.relevant_roots)
        self.assertIn("/99_process", frame.relevant_roots)


if __name__ == "__main__":
    unittest.main()
