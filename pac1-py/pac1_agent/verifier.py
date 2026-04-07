from __future__ import annotations

from .models import ReportTaskCompletion, ToolRequest
from .pathing import normalize_repo_path
from .policy import (
    clear_verified_paths,
    command_paths,
    is_mutating_command,
    is_verification_command,
    mutation_guard,
)


def generic_completion_guard(payload: ReportTaskCompletion) -> str | None:
    if payload.outcome != "OUTCOME_OK":
        return None

    generic_steps = {
        "completed the requested work",
        "completed the task",
        "task completed",
        "finished the task",
    }
    generic_messages = {
        "task completed.",
        "completed the requested work.",
        "task completed",
        "completed the requested work",
    }
    normalized_steps = {step.strip().lower() for step in payload.completed_steps_laconic if step.strip()}
    normalized_message = payload.message.strip().lower()

    if not payload.grounding_refs:
        return (
            "Do not report OUTCOME_OK without concrete grounding refs. "
            "Ground the result in observed files or return clarification."
        )
    if not normalized_steps or normalized_steps <= generic_steps:
        return (
            "Do not report OUTCOME_OK with generic completed steps. "
            "List the concrete work that was verified."
        )
    if normalized_message in generic_messages:
        return (
            "Do not report OUTCOME_OK with a generic completion message. "
            "Summarize the concrete verified result."
        )
    return None


def prepare_command(
    task_text: str,
    pending_verification_paths: set[str],
    cmd: ToolRequest,
) -> str | None:
    guard = mutation_guard(task_text, cmd)
    if guard:
        return guard

    if isinstance(cmd, ReportTaskCompletion):
        completion_guard = generic_completion_guard(cmd)
        if completion_guard:
            return completion_guard

    if isinstance(cmd, ReportTaskCompletion) and pending_verification_paths:
        pending = ", ".join(sorted(pending_verification_paths))
        return f"Verification required before report_completion. Confirm final state for: {pending}"

    return None


def next_pending_verification_paths(
    pending_paths: set[str],
    cmd: ToolRequest,
) -> set[str]:
    updated = set(pending_paths)
    paths = command_paths(cmd)
    if is_mutating_command(cmd):
        for path in paths:
            updated.add(normalize_repo_path(path))
        return updated
    if is_verification_command(cmd):
        return clear_verified_paths(updated, paths)
    return updated
