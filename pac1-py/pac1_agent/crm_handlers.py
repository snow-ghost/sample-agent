from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .models import ReportTaskCompletion
from .pathing import normalize_repo_path
from .workflows import (
    CrmLookupRequest,
    count_channel_status,
    names_match,
    parse_crm_lookup_request,
    parse_direct_outbound_request,
    parse_explicit_email_instruction,
)


ResolveAccountByDescriptorFn = Callable[[Any, Any, str], tuple[str, dict] | None]
FindInternalContactByNameFn = Callable[[Any, Any, str], tuple[str, dict] | None]
IterAccountRecordsFn = Callable[[Any, Any], list[tuple[str, dict]]]
FindNamedContactFn = Callable[[Any, Any, str], tuple[str, dict] | None]
ResolveDirectEmailTargetFn = Callable[[Any, Any, str], tuple[str, list[str]] | None]
WriteOutboundEmailFn = Callable[[Any, Any, str, str, str, list[str] | None], str | None]
ParseChannelStatusRequestFn = Callable[[Any, Any, str], Any]
ReadNamedChannelStatusTextFn = Callable[[Any, Any, str], tuple[str | None, str | None]]
ReadJsonFn = Callable[[Any, Any, str], dict | None]
ReadTextFn = Callable[[Any, Any, str], str | None]
AnswerAndStopFn = Callable[[Any, ReportTaskCompletion], None]


@dataclass(frozen=True)
class CrmHandlerOps:
    resolve_account_by_descriptor: ResolveAccountByDescriptorFn
    find_internal_contact_by_name: FindInternalContactByNameFn
    iter_account_records: IterAccountRecordsFn
    find_named_contact: FindNamedContactFn
    resolve_direct_email_target: ResolveDirectEmailTargetFn
    write_outbound_email: WriteOutboundEmailFn
    parse_channel_status_request: ParseChannelStatusRequestFn
    read_named_channel_status_text: ReadNamedChannelStatusTextFn
    read_json: ReadJsonFn
    read_text: ReadTextFn
    answer_and_stop: AnswerAndStopFn


def handle_contact_email_lookup(
    ops: CrmHandlerOps,
    runtime: Any,
    session: Any,
) -> bool:
    if session.repository_profile != "typed_crm_fs":
        return False

    request = parse_crm_lookup_request(session.task_text)
    if request is None:
        return False

    if request.kind == "legal_name":
        account_match = ops.resolve_account_by_descriptor(runtime, session, request.target)
        if account_match is None:
            return False
        account_path, account = account_match
        legal_name = str(account.get("legal_name", "")).strip()
        if not legal_name:
            return False
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=[f"Resolved account {account.get('name', '')}"],
                message=legal_name,
                grounding_refs=[account_path],
                outcome="OUTCOME_OK",
            ),
        )
        return True

    if request.kind == "primary_contact_email":
        account_match = ops.resolve_account_by_descriptor(runtime, session, request.target)
        if account_match is None:
            return False
        account_path, account = account_match
        contact_path = f"/contacts/{str(account.get('primary_contact_id', '')).strip()}.json"
        contact = ops.read_json(runtime, session, contact_path)
        if not contact:
            return False
        email = str(contact.get("email", "")).strip()
        if not email:
            return False
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=[f"Resolved primary contact for {account.get('name', '')}"],
                message=email,
                grounding_refs=[account_path, contact_path],
                outcome="OUTCOME_OK",
            ),
        )
        return True

    if request.kind == "manager_email":
        account_match = ops.resolve_account_by_descriptor(runtime, session, request.target)
        if account_match is None:
            return False
        account_path, account = account_match
        manager_match = ops.find_internal_contact_by_name(runtime, session, str(account.get("account_manager", "")))
        if manager_match is None:
            return False
        manager_path, manager = manager_match
        email = str(manager.get("email", "")).strip()
        if not email:
            return False
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=[f"Resolved account manager for {account.get('name', '')}"],
                message=email,
                grounding_refs=[account_path, manager_path],
                outcome="OUTCOME_OK",
            ),
        )
        return True

    if request.kind == "managed_accounts":
        matched_accounts = [
            (path, account)
            for path, account in ops.iter_account_records(runtime, session)
            if names_match(str(account.get("account_manager", "")), request.target)
        ]
        if not matched_accounts:
            return False
        matched_accounts.sort(key=lambda item: str(item[1].get("name", "")))
        manager_match = ops.find_internal_contact_by_name(runtime, session, request.target)
        refs = [path for path, _ in matched_accounts[:8]]
        if manager_match is not None:
            refs.insert(0, manager_match[0])
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=[f"Resolved accounts managed by {request.target}"],
                message="\n".join(str(account.get("name", "")) for _, account in matched_accounts),
                grounding_refs=refs,
                outcome="OUTCOME_OK",
            ),
        )
        return True

    if request.kind != "contact_email":
        return False

    match = ops.find_named_contact(runtime, session, request.target)
    if match is None:
        return False
    contact_path, contact = match
    email = str(contact.get("email", "")).strip()
    if not email:
        return False
    ops.answer_and_stop(
        runtime,
        ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=[f"Matched contact {contact.get('full_name', '')}"],
            message=email,
            grounding_refs=[contact_path],
            outcome="OUTCOME_OK",
        ),
    )
    return True


def handle_direct_outbound_email(
    ops: CrmHandlerOps,
    runtime: Any,
    session: Any,
) -> bool:
    if not session.capabilities.supports_outbound_email or not session.capabilities.has_contacts:
        return False
    if "inbox" in session.task_text.lower():
        return False

    parsed = parse_direct_outbound_request(session.task_text)
    if parsed is None:
        parsed = parse_explicit_email_instruction(session.task_text)
    if parsed is None:
        return False

    target, subject, body = parsed
    resolved = ops.resolve_direct_email_target(runtime, session, target)
    if resolved is None:
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=[
                    f"Parsed outbound email request for {target}",
                    f"Found no matching contact for {target}",
                ],
                message=(
                    f"No contact record found for {target} in /contacts. "
                    "Cannot send email without a valid email address. "
                    "Please confirm the correct contact or provide their email."
                ),
                grounding_refs=["/contacts/README.MD", "/outbox/README.MD"],
                outcome="OUTCOME_NONE_CLARIFICATION",
            ),
        )
        return True

    to_email, refs = resolved
    outbox_path = ops.write_outbound_email(runtime, session, to_email, subject, body, None)
    if outbox_path is None:
        return False

    ops.read_text(runtime, session, outbox_path)
    ops.read_text(runtime, session, "/outbox/seq.json")

    ops.answer_and_stop(
        runtime,
        ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=[
                f"Resolved outbound email target {target}",
                f"Wrote outbound email {normalize_repo_path(outbox_path)}",
            ],
            message="Processed the outbound email request.",
            grounding_refs=[*refs, outbox_path, "/outbox/seq.json"],
            outcome="OUTCOME_OK",
        ),
    )
    return True


def handle_channel_status_lookup(
    ops: CrmHandlerOps,
    runtime: Any,
    session: Any,
) -> bool:
    if not session.capabilities.has_channel_docs:
        return False

    request = ops.parse_channel_status_request(runtime, session, session.task_text)
    if request is None:
        return False

    channel_path, channel_text = ops.read_named_channel_status_text(runtime, session, request.channel_name)
    if channel_text is None or channel_path is None:
        return False

    total = count_channel_status(channel_text, request.status)
    ops.answer_and_stop(
        runtime,
        ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=[
                f"Read {request.channel_name} channel status file",
                f"Counted {total} {request.status} {request.channel_name} accounts",
            ],
            message=str(total),
            grounding_refs=[channel_path],
            outcome="OUTCOME_OK",
        ),
    )
    return True
