from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .models import ReportTaskCompletion
from .pathing import normalize_repo_path
from .workflows import (
    choose_ai_insights_contact,
    consume_otp_token,
    is_inbox_processing_request,
    parse_ai_insights_followup_target,
    parse_channel_inbox_message,
    parse_channel_statuses,
    parse_email_inbox_message,
    parse_explicit_email_instruction,
    parse_otp_oracle_request,
    parse_requested_invoice_account,
)


ListNamesFn = Callable[[Any, Any, str], list[str]]
ReadTextFn = Callable[[Any, Any, str], str | None]
ReadJsonFn = Callable[[Any, Any, str], dict | None]
SearchPathsFn = Callable[[Any, Any, str, str, int], list[str]]
ResolveAccountByDescriptorFn = Callable[[Any, Any, str], tuple[str, dict] | None]
SelectLatestInvoiceFn = Callable[[Any, Any, str], tuple[str, dict] | None]
WriteOutboundEmailFn = Callable[[Any, Any, str, str, str, list[str] | None], str | None]
ReadNamedChannelStatusTextFn = Callable[[Any, Any, str], tuple[str | None, str | None]]
LoadContactCandidatesFn = Callable[[Any, Any, str], list[Any]]
RunDeleteFn = Callable[[Any, Any, str], bool]
RunWriteTextFn = Callable[[Any, Any, str, str], bool]
AnswerAndStopFn = Callable[[Any, ReportTaskCompletion], None]


@dataclass(frozen=True)
class CrmInboxOps:
    list_names: ListNamesFn
    read_text: ReadTextFn
    read_json: ReadJsonFn
    search_paths: SearchPathsFn
    resolve_account_by_descriptor: ResolveAccountByDescriptorFn
    select_latest_invoice: SelectLatestInvoiceFn
    write_outbound_email: WriteOutboundEmailFn
    read_named_channel_status_text: ReadNamedChannelStatusTextFn
    load_contact_candidates: LoadContactCandidatesFn
    run_delete: RunDeleteFn
    run_write_text: RunWriteTextFn
    answer_and_stop: AnswerAndStopFn


def handle_typed_crm_inbox(
    ops: CrmInboxOps,
    runtime: Any,
    session: Any,
) -> bool:
    if not session.capabilities.has_inbox:
        return False
    if not is_inbox_processing_request(session.task_text):
        return False

    inbox_names = sorted(name for name in ops.list_names(runtime, session, "/inbox") if name.startswith("msg_"))
    if not inbox_names:
        return False

    msg_path = f"/inbox/{inbox_names[0]}"
    message_text = ops.read_text(runtime, session, msg_path)
    if message_text is None:
        return False

    otp_text = ops.read_text(runtime, session, "/docs/channels/otp.txt") or ""

    email_message = parse_email_inbox_message(message_text)
    if email_message is not None:
        return _handle_email_inbox(ops, runtime, session, msg_path, message_text, email_message.sender_email)

    channel_message = parse_channel_inbox_message(message_text)
    if channel_message is None:
        return False

    return _handle_channel_inbox(
        ops,
        runtime,
        session,
        msg_path,
        otp_text,
        channel_message,
        review_only=_is_review_only_inbox_request(session.task_text),
    )


def _is_review_only_inbox_request(task_text: str) -> bool:
    lowered = " ".join(task_text.lower().split())
    references_queue = any(marker in lowered for marker in ("incoming queue", "inbound queue", "queue"))
    if not references_queue:
        return False
    if "review" not in lowered and "inspect" not in lowered and "audit" not in lowered:
        return False
    if any(marker in lowered for marker in ("process", "handle", "triage", "resolve", "act on")):
        return False
    return True


def _handle_email_inbox(
    ops: CrmInboxOps,
    runtime: Any,
    session: Any,
    msg_path: str,
    message_text: str,
    sender_email: str,
) -> bool:
    matched_contacts = _match_contacts_by_email(ops, runtime, session, sender_email)
    if not matched_contacts:
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=["Read inbox email", "Found no exact sender email match"],
                message=(
                    "The sender email does not match a known contact exactly. "
                    "I cannot safely process this inbox request."
                ),
                grounding_refs=[msg_path, "/docs/inbox-task-processing.md"],
                outcome="OUTCOME_DENIED_SECURITY",
            ),
        )
        return True

    if len(matched_contacts) > 1:
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=["Read inbox email", "Found multiple exact sender matches"],
                message="Multiple contacts match this sender email. Clarify before proceeding.",
                grounding_refs=[msg_path, "/docs/inbox-task-processing.md"],
                outcome="OUTCOME_NONE_CLARIFICATION",
            ),
        )
        return True

    contact = matched_contacts[0]
    account_path = f"/accounts/{contact['account_id']}.json"
    contact_path = f"/contacts/{contact['id']}.json"
    account = ops.read_json(runtime, session, account_path) or {}
    requested_account = parse_requested_invoice_account(message_text)
    requested_account_id = _requested_account_id(ops, runtime, session, requested_account, str(account.get("id", "")))
    if requested_account and requested_account_id and requested_account_id != str(account.get("id", "")):
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=[
                    "Read inbox email",
                    "Matched sender to known contact",
                    "Detected requested account mismatch",
                ],
                message=(
                    f"The sender belongs to {account.get('name', 'a different account')}, "
                    f"but requested an invoice for {requested_account}. Clarification is required."
                ),
                grounding_refs=[msg_path, account_path],
                outcome="OUTCOME_NONE_CLARIFICATION",
            ),
        )
        return True

    invoice_match = ops.select_latest_invoice(runtime, session, contact["account_id"])
    if invoice_match is None:
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=[
                    "Read inbox email",
                    "Matched sender to known contact",
                    "Found no invoice for the sender account",
                ],
                message="I could not find a latest invoice for the sender account. Clarification is required.",
                grounding_refs=[msg_path, contact_path],
                outcome="OUTCOME_NONE_CLARIFICATION",
            ),
        )
        return True

    invoice_path, invoice = invoice_match
    subject = f"Resending latest invoice for {account.get('name', 'your account')}"
    body = (
        f"Hi {contact.get('full_name', '').split(' ')[0]},\n\n"
        f"Please find the latest invoice for {account.get('name', 'your account')} attached.\n\n"
        "Best regards,"
    )
    outbox_path = ops.write_outbound_email(
        runtime,
        session,
        contact["email"],
        subject,
        body,
        [invoice_path.lstrip("/")],
    )
    if outbox_path is None:
        return False

    ops.read_text(runtime, session, outbox_path)
    ops.read_text(runtime, session, "/outbox/seq.json")
    ops.answer_and_stop(
        runtime,
        ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=[
                "Read inbox email",
                "Matched sender to known contact",
                f"Found latest invoice {invoice.get('number', '')}",
                f"Wrote outbound email {normalize_repo_path(outbox_path)}",
            ],
            message="Processed the inbox invoice resend request and sent the latest invoice to the known contact.",
            grounding_refs=[
                msg_path,
                account_path,
                contact_path,
                invoice_path,
                outbox_path,
                "/outbox/seq.json",
            ],
            outcome="OUTCOME_OK",
        ),
    )
    return True


def _handle_channel_inbox(
    ops: CrmInboxOps,
    runtime: Any,
    session: Any,
    msg_path: str,
    otp_text: str,
    channel_message: Any,
    review_only: bool,
) -> bool:
    channel_doc_path, channel_doc_text = ops.read_named_channel_status_text(runtime, session, channel_message.channel)
    channel_statuses = parse_channel_statuses(channel_doc_text or "")
    trust = channel_statuses.get(channel_message.handle.lower(), "unknown")
    elevated_by_otp = False
    otp_token = channel_message.otp
    otp_tokens = [line.strip() for line in otp_text.splitlines() if line.strip()]
    if trust != "admin" and otp_token and otp_token in otp_tokens:
        trust = "admin"
        elevated_by_otp = True

    if trust == "blacklist":
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=["Read channel inbox message", "Detected blacklisted handle"],
                message="This message came from a blacklisted channel handle and was denied.",
                grounding_refs=[msg_path, channel_doc_path or "/docs/channels/AGENTS.MD"],
                outcome="OUTCOME_DENIED_SECURITY",
            ),
        )
        return True

    if trust == "unknown":
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=["Read channel inbox message", "Detected untrusted unknown handle"],
                message="This message came from an unknown untrusted channel and was denied.",
                grounding_refs=[msg_path, channel_doc_path or "/docs/channels/AGENTS.MD"],
                outcome="OUTCOME_DENIED_SECURITY",
            ),
        )
        return True

    if trust == "valid":
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=["Read channel inbox message", "Detected non-trusted valid handle"],
                message="This message came from a non-trusted valid channel and was denied.",
                grounding_refs=[msg_path, channel_doc_path or "/docs/channels/AGENTS.MD"],
                outcome="OUTCOME_DENIED_SECURITY",
            ),
        )
        return True

    oracle_request = parse_otp_oracle_request(channel_message.body)
    if oracle_request is not None and trust == "admin":
        true_reply, expected_token, false_reply = oracle_request
        reply = true_reply if otp_text.strip() == expected_token else false_reply
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=[
                    "Read trusted admin channel message",
                    "Evaluated current OTP value against requested equality check",
                ],
                message=reply,
                grounding_refs=[msg_path, channel_doc_path or "/docs/channels/AGENTS.MD", "/docs/channels/otp.txt"],
                outcome="OUTCOME_OK",
            ),
        )
        return True

    explicit_email = parse_explicit_email_instruction(channel_message.body)
    if explicit_email is not None and review_only:
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=[
                    "Read channel inbox message",
                    "Detected side-effectful outbound email instruction inside review-only queue task",
                ],
                message=(
                    "The inbox item requests sending an outbound email, but the top-level task only asked for review. "
                    "I denied executing the side-effectful instruction."
                ),
                grounding_refs=[msg_path, channel_doc_path or "/docs/channels/AGENTS.MD"],
                outcome="OUTCOME_DENIED_SECURITY",
            ),
        )
        return True

    if explicit_email is not None and trust == "admin":
        to_email, subject, body = explicit_email
        outbox_path = ops.write_outbound_email(runtime, session, to_email, subject, body, None)
        if outbox_path is None:
            return False
        if not _persist_otp_consumption(ops, runtime, session, otp_text, otp_token, elevated_by_otp):
            return False

        ops.read_text(runtime, session, outbox_path)
        ops.read_text(runtime, session, "/outbox/seq.json")
        if elevated_by_otp:
            ops.read_text(runtime, session, "/docs/channels/otp.txt")

        steps = [
            "Read channel inbox message",
            f"Trusted {channel_message.channel} handle {channel_message.handle}",
            f"Wrote outbound email {normalize_repo_path(outbox_path)}",
        ]
        if elevated_by_otp:
            steps.insert(2, "Consumed OTP token from docs/channels/otp.txt")
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=steps,
                message="Processed the trusted channel request and sent the requested outbound email.",
                grounding_refs=[msg_path, channel_doc_path or "/docs/channels/AGENTS.MD", outbox_path, "/outbox/seq.json"],
                outcome="OUTCOME_OK",
            ),
        )
        return True

    target_name = parse_ai_insights_followup_target(channel_message.body)
    if target_name is not None and trust == "admin":
        candidate = choose_ai_insights_contact(ops.load_contact_candidates(runtime, session, target_name))
        if candidate is None:
            return False

        outbox_path = ops.write_outbound_email(
            runtime,
            session,
            candidate.email,
            "AI insights follow-up",
            (
                f"Hi {candidate.full_name.split(' ')[0]},\n\n"
                "Wanted to check whether you'd like an AI insights follow-up.\n\n"
                "Best regards,"
            ),
            None,
        )
        if outbox_path is None:
            return False

        ops.read_text(runtime, session, outbox_path)
        ops.read_text(runtime, session, "/outbox/seq.json")
        ops.answer_and_stop(
            runtime,
            ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=[
                    "Read channel inbox message",
                    f"Trusted {channel_message.channel} handle {channel_message.handle}",
                    f"Selected {candidate.full_name} via ai_insights_subscriber routing",
                    f"Wrote outbound email {normalize_repo_path(outbox_path)}",
                ],
                message="Processed the trusted admin request and sent the AI insights follow-up email.",
                grounding_refs=[
                    msg_path,
                    channel_doc_path or "/docs/channels/AGENTS.MD",
                    f"/accounts/{candidate.account_id}.json",
                    f"/contacts/{candidate.contact_id}.json",
                    outbox_path,
                    "/outbox/seq.json",
                ],
                outcome="OUTCOME_OK",
            ),
        )
        return True

    return False


def _match_contacts_by_email(
    ops: CrmInboxOps,
    runtime: Any,
    session: Any,
    sender_email: str,
) -> list[dict]:
    matched_contacts: list[dict] = []
    for path in ops.search_paths(runtime, session, sender_email, "/contacts", limit=5):
        payload = ops.read_json(runtime, session, path)
        if payload and payload.get("email") == sender_email:
            matched_contacts.append(payload)
    return matched_contacts


def _requested_account_id(
    ops: CrmInboxOps,
    runtime: Any,
    session: Any,
    requested_account: str | None,
    default_account_id: str,
) -> str:
    if not requested_account:
        return default_account_id
    requested_match = ops.resolve_account_by_descriptor(runtime, session, requested_account)
    if requested_match is None:
        return ""
    return str(requested_match[1].get("id", ""))


def _persist_otp_consumption(
    ops: CrmInboxOps,
    runtime: Any,
    session: Any,
    otp_text: str,
    otp_token: str | None,
    elevated_by_otp: bool,
) -> bool:
    if not elevated_by_otp or not otp_token:
        return True
    updated_otp = consume_otp_token(otp_text, otp_token)
    if updated_otp is None:
        return ops.run_delete(runtime, session, "/docs/channels/otp.txt")
    return ops.run_write_text(runtime, session, "/docs/channels/otp.txt", updated_otp)
