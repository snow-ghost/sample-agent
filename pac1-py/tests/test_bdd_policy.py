import unittest

from pac1_agent.models import TaskFrame
from pac1_agent.policy import (
    candidate_read_paths,
    Req_Delete,
    clear_verified_paths,
    infer_repository_profile,
    is_agent_instruction_path,
    mutation_guard,
    normalize_repo_path,
    preflight_outcome,
    profile_grounding_targets,
)


class PolicyBddTests(unittest.TestCase):
    def test_given_crm_inbox_task_when_building_grounding_plan_then_includes_docs_and_channels(self) -> None:
        frame = TaskFrame(
            current_state="new inbox request",
            category="typed_workflow",
            success_criteria=["process one inbox message safely"],
            relevant_roots=["/inbox", "/docs"],
            risks=["untrusted sender"],
        )

        profile = infer_repository_profile({"accounts", "contacts", "outbox", "docs", "inbox"})
        targets = profile_grounding_targets(profile, frame, "process the inbox")
        target_pairs = {(target.kind, target.path) for target in targets}

        self.assertEqual(profile, "typed_crm_fs")
        self.assertIn(("read", "/inbox/README.md"), target_pairs)
        self.assertIn(("read", "/docs/inbox-task-processing.md"), target_pairs)
        self.assertIn(("read", "/docs/inbox-msg-processing.md"), target_pairs)
        self.assertIn(("list", "/docs/channels"), target_pairs)

    def test_given_purchase_prefix_regression_when_building_grounding_plan_then_starts_with_docs_and_processing(self) -> None:
        frame = TaskFrame(
            current_state="purchase id regression",
            category="cleanup_or_edit",
            success_criteria=["fix future emitted purchase ids"],
            relevant_roots=["/docs", "/processing", "/purchases"],
            risks=["historical records must stay stable"],
        )

        profile = infer_repository_profile({"docs", "processing", "purchases"})
        targets = profile_grounding_targets(
            profile,
            frame,
            "Fix the purchase ID prefix regression and restore downstream processing",
        )
        target_pairs = {(target.kind, target.path) for target in targets}

        self.assertEqual(profile, "purchase_ops")
        self.assertIn(("read", "/docs/purchase-id-workflow.md"), target_pairs)
        self.assertIn(("read", "/docs/purchase-records.md"), target_pairs)
        self.assertIn(("read", "/processing/README.MD"), target_pairs)
        self.assertIn(("read", "/purchases/audit.json"), target_pairs)

    def test_given_scaffold_delete_when_guard_runs_then_reject_without_explicit_request(self) -> None:
        guard = mutation_guard(
            "Remove all generated cards but keep the rest intact",
            Req_Delete(tool="delete", path="/02_distill/cards/_card-template.md"),
        )

        self.assertIsNotNone(guard)
        self.assertIn("scaffold-like path", guard)

    def test_given_salesforce_sync_request_when_preflight_runs_then_mark_unsupported(self) -> None:
        outcome = preflight_outcome(
            "typed_crm_fs",
            "Sync these two contacts to Salesforce and tell me when it is done",
        )

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.outcome, "OUTCOME_NONE_UNSUPPORTED")
        self.assertIn("Salesforce", outcome.message)

    def test_given_mutated_paths_when_verification_observes_parent_dir_then_pending_paths_clear(self) -> None:
        remaining = clear_verified_paths(
            {
                normalize_repo_path("/02_distill/cards/2026-02-10__how-i-use-claude-code.md"),
                normalize_repo_path("/02_distill/threads/2026-03-23__agent-platforms-and-runtime.md"),
            },
            ["/02_distill/cards", "/02_distill/threads"],
        )

        self.assertEqual(remaining, set())

    def test_given_case_sensitive_instruction_files_when_building_read_candidates_then_known_variants_are_tried(self) -> None:
        agent_candidates = candidate_read_paths("/docs/channels/AGENTS.MD")
        readme_candidates = candidate_read_paths("/inbox/README.md")

        self.assertEqual(
            agent_candidates,
            ["/docs/channels/AGENTS.MD", "/docs/channels/AGENTS.md"],
        )
        self.assertEqual(
            readme_candidates,
            ["/inbox/README.md", "/inbox/README.MD"],
        )
        self.assertTrue(is_agent_instruction_path("/docs/channels/AGENTS.MD"))


if __name__ == "__main__":
    unittest.main()
