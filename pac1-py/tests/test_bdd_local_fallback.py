import unittest

from pac1_agent.capabilities import infer_workspace_capabilities
from pac1_agent.loop import AgentSessionState, _local_fallback_command


class LocalFallbackBddTests(unittest.TestCase):
    def test_given_knowledge_repo_cleanup_task_when_local_fallback_runs_then_cleanup_docs_and_distill_surfaces_are_grounded(self) -> None:
        session = AgentSessionState(
            task_text="Remove all captured cards and threads. Do not touch anything else.",
            repository_profile="knowledge_repo",
            capabilities=infer_workspace_capabilities(profile="knowledge_repo"),
        )

        first = _local_fallback_command(session)
        second = _local_fallback_command(session)

        self.assertEqual(first.tool, "read")
        self.assertEqual(first.path, "/99_process/document_cleanup.md")
        self.assertEqual(second.tool, "read")
        self.assertEqual(second.path, "/02_distill/AGENTS.md")

    def test_given_crm_email_lookup_task_when_local_fallback_runs_then_contacts_surface_is_grounded(self) -> None:
        session = AgentSessionState(
            task_text="What is the email address of Boer Milou? Return only the email",
            repository_profile="typed_crm_fs",
            capabilities=infer_workspace_capabilities(profile="typed_crm_fs"),
        )

        first = _local_fallback_command(session)
        second = _local_fallback_command(session)

        self.assertEqual(first.tool, "list")
        self.assertEqual(first.path, "/contacts")
        self.assertEqual(second.tool, "read")
        self.assertEqual(second.path, "/contacts/README.MD")


if __name__ == "__main__":
    unittest.main()
