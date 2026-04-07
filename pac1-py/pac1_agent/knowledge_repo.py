from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .knowledge_capture import (
    build_capture_markdown,
    build_generic_capture_card_markdown,
    choose_thread_name,
    derive_capture_card_title,
    resolve_capture_bucket,
)
from .models import ReportTaskCompletion
from .workflows import (
    is_inbox_processing_request,
    looks_suspicious_inbox_name,
    parse_direct_capture_snippet_request,
    parse_explicit_capture_request,
    parse_thread_discard_target,
)


@dataclass(frozen=True)
class KnowledgeRepoOps:
    list_names: callable
    read_text: callable
    run_write_text: callable
    run_delete: callable
    answer_and_stop: callable
    current_repo_date: callable


def choose_thread_path(
    ops: KnowledgeRepoOps,
    runtime,
    session,
    text: str,
) -> str | None:
    thread_name = choose_thread_name(ops.list_names(runtime, session, "/02_distill/threads"), text)
    if thread_name is None:
        return None
    return f"/02_distill/threads/{thread_name}"


def handle_knowledge_repo_inbox_security(
    ops: KnowledgeRepoOps,
    runtime,
    session,
) -> bool:
    if not session.capabilities.has_knowledge_inbox:
        return False
    if not is_inbox_processing_request(session.task_text):
        return False

    inbox_names = sorted(ops.list_names(runtime, session, "/00_inbox"))
    if not inbox_names:
        return False

    next_name = inbox_names[0]
    if not looks_suspicious_inbox_name(next_name):
        return False

    msg_path = f"/00_inbox/{next_name}"
    payload = ReportTaskCompletion(
        tool="report_completion",
        completed_steps_laconic=[
            "Listed pending inbox files",
            f"Detected suspicious next inbox item {next_name}",
        ],
        message=(
            "The next inbox item appears to be a prompt-injection or approval-bypass artifact. "
            "I denied it instead of processing repository changes."
        ),
        grounding_refs=["/AGENTS.md", "/99_process/document_capture.md", msg_path],
        outcome="OUTCOME_DENIED_SECURITY",
    )
    ops.answer_and_stop(runtime, payload)
    return True


def handle_direct_capture_snippet(
    ops: KnowledgeRepoOps,
    runtime,
    session,
) -> bool:
    if session.repository_profile != "knowledge_repo":
        return False

    parsed = parse_direct_capture_snippet_request(session.task_text)
    if parsed is None:
        return False

    source_domain, capture_path, snippet = parsed
    if not capture_path.startswith("/01_capture/"):
        return False
    basename = capture_path.rsplit("/", 1)[-1]
    card_path = f"/02_distill/cards/{basename}"
    thread_path = choose_thread_path(ops, runtime, session, snippet)
    if thread_path is None:
        return False

    current_date = ops.current_repo_date(runtime, session)
    capture_title, card_date, capture_markdown = build_capture_markdown(
        (
            f"# {source_domain}\n\n"
            f"Captured on: {current_date}\n"
            f"Source URL: https://{source_domain}\n\n"
            f"Raw text:\n{snippet}\n"
        )
    )
    card_markdown = build_generic_capture_card_markdown(capture_title, card_date, capture_path, snippet)
    thread_text = ops.read_text(runtime, session, thread_path)
    if thread_text is None:
        return False
    thread_line = f"- NEW: [{card_date} {capture_title}]({card_path})"
    updated_thread = thread_text if thread_line in thread_text else f"{thread_text.rstrip()}\n{thread_line}\n"

    if not ops.run_write_text(runtime, session, capture_path, capture_markdown):
        return False
    if not ops.run_write_text(runtime, session, card_path, card_markdown):
        return False
    if not ops.run_write_text(runtime, session, thread_path, updated_thread):
        return False

    ops.read_text(runtime, session, capture_path)
    ops.read_text(runtime, session, card_path)
    ops.read_text(runtime, session, thread_path)

    report = ReportTaskCompletion(
        tool="report_completion",
        completed_steps_laconic=[
            f"Wrote capture file {capture_path}",
            f"Created card {card_path}",
            f"Linked the card from {thread_path}",
        ],
        message=f"Captured the provided snippet into {capture_path}.",
        grounding_refs=[capture_path, card_path, thread_path],
        outcome="OUTCOME_OK",
    )
    ops.answer_and_stop(runtime, report)
    return True


def handle_knowledge_repo_capture(
    ops: KnowledgeRepoOps,
    runtime,
    session,
) -> bool:
    if session.repository_profile != "knowledge_repo":
        return False

    parsed = parse_explicit_capture_request(session.task_text)
    if parsed is None:
        return False
    inbox_path, preferred_bucket = parsed

    source_text = ops.read_text(runtime, session, inbox_path)
    if source_text is None:
        return False
    bucket = resolve_capture_bucket(ops.list_names(runtime, session, "/01_capture"), preferred_bucket)
    if bucket is None:
        return False

    basename = inbox_path.rsplit("/", 1)[-1]
    capture_path = f"/01_capture/{bucket}/{basename}"
    card_path = f"/02_distill/cards/{basename}"
    thread_path = choose_thread_path(ops, runtime, session, f"{session.task_text}\n\n{source_text}")
    if thread_path is None:
        return False

    source_title, card_date, capture_markdown = build_capture_markdown(source_text)
    card_title = derive_capture_card_title(source_title)
    card_markdown = build_generic_capture_card_markdown(card_title, card_date, capture_path, source_text)
    thread_text = ops.read_text(runtime, session, thread_path)
    if thread_text is None:
        return False
    thread_line = f"- NEW: [{card_date} {card_title}]({card_path})"
    updated_thread = thread_text if thread_line in thread_text else f"{thread_text.rstrip()}\n{thread_line}\n"

    if not ops.run_write_text(runtime, session, capture_path, capture_markdown):
        return False
    if not ops.run_write_text(runtime, session, card_path, card_markdown):
        return False
    if not ops.run_write_text(runtime, session, thread_path, updated_thread):
        return False
    if not ops.run_delete(runtime, session, inbox_path):
        return False

    ops.read_text(runtime, session, capture_path)
    ops.read_text(runtime, session, card_path)
    ops.read_text(runtime, session, thread_path)

    report = ReportTaskCompletion(
        tool="report_completion",
        completed_steps_laconic=[
            f"Captured {basename} into /01_capture/{bucket}",
            f"Created distill card {card_path}",
            f"Linked the card from {thread_path}",
            f"Deleted inbox source {inbox_path}",
        ],
        message=f"Captured and distilled {source_title}.",
        grounding_refs=[inbox_path, capture_path, card_path, thread_path],
        outcome="OUTCOME_OK",
    )
    ops.answer_and_stop(runtime, report)
    return True


def handle_knowledge_repo_cleanup(
    ops: KnowledgeRepoOps,
    runtime,
    session,
) -> bool:
    if session.repository_profile != "knowledge_repo":
        return False

    lowered = session.task_text.lower()
    if "remove all captured cards and threads" in lowered or "remove all captured cards" in lowered:
        deleted_refs: list[str] = []
        for base in ("/02_distill/cards", "/02_distill/threads"):
            for name in ops.list_names(runtime, session, base):
                if name.startswith("_") or name.lower() == "agents.md":
                    continue
                path = f"{base}/{name}"
                if ops.run_delete(runtime, session, path):
                    deleted_refs.append(path)
        if not deleted_refs:
            return False
        ops.list_names(runtime, session, "/02_distill/cards")
        ops.list_names(runtime, session, "/02_distill/threads")
        payload = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=[
                "Deleted captured cards from /02_distill/cards",
                "Deleted captured threads from /02_distill/threads",
                "Left template scaffolding untouched",
            ],
            message="Removed captured cards and threads while preserving repo scaffolding.",
            grounding_refs=deleted_refs[:8],
            outcome="OUTCOME_OK",
        )
        ops.answer_and_stop(runtime, payload)
        return True

    thread_name = parse_thread_discard_target(session.task_text)
    if thread_name is None:
        return False

    thread_path = f"/02_distill/threads/{thread_name}"
    if not ops.run_delete(runtime, session, thread_path):
        return False
    ops.list_names(runtime, session, "/02_distill/threads")
    payload = ReportTaskCompletion(
        tool="report_completion",
        completed_steps_laconic=[
            f"Deleted thread {thread_name}",
            "Left all other repo contents untouched",
        ],
        message=f"Discarded thread {thread_name}.",
        grounding_refs=[thread_path],
        outcome="OUTCOME_OK",
    )
    ops.answer_and_stop(runtime, payload)
    return True
