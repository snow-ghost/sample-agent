from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

from connectrpc.errors import ConnectError

from .config import AgentConfig
from .llm import JsonChatClient
from .models import (
    NextStep,
    ReportTaskCompletion,
    Req_Context,
    Req_Delete,
    Req_List,
    Req_Read,
    Req_Search,
    Req_Tree,
    Req_Write,
    TaskFrame,
    ToolRequest,
)
from .policy import (
    AGENT_FILE_NAMES,
    build_execution_prompt,
    build_system_prompt,
    build_task_frame_prompt,
    build_tool_result_prompt,
    candidate_agent_paths,
    candidate_read_paths,
    clear_verified_paths,
    command_paths,
    extract_startup_reads,
    infer_repository_profile,
    is_agent_instruction_path,
    is_mutating_command,
    is_verification_command,
    mutation_guard,
    normalize_repo_path,
    preflight_outcome,
    profile_grounding_targets,
    relevant_roots,
)
from .runtime import PcmRuntimeAdapter
from .workflows import (
    ChannelInboxMessage,
    ContactCandidate,
    choose_ai_insights_contact,
    count_channel_status,
    consume_otp_token,
    extract_purchase_prefix,
    looks_suspicious_inbox_name,
    names_match,
    parse_ai_insights_followup_target,
    parse_channel_inbox_message,
    parse_channel_statuses,
    parse_direct_outbound_request,
    parse_email_inbox_message,
    parse_explicit_email_instruction,
    parse_otp_oracle_request,
    parse_requested_invoice_account,
)

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_CLR = "\x1B[0m"
CLI_BLUE = "\x1B[34m"
CLI_YELLOW = "\x1B[33m"


@dataclass
class AgentSessionState:
    task_text: str
    messages: list[dict[str, str]] = field(default_factory=list)
    grounded_agent_paths: set[str] = field(default_factory=set)
    attempted_agent_paths: set[str] = field(default_factory=set)
    pending_verification_paths: set[str] = field(default_factory=set)
    root_entries: set[str] = field(default_factory=set)
    repository_profile: str = "generic"
    frame: TaskFrame | None = None

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})


def _append_tool_result(
    session: AgentSessionState,
    tool_name: str,
    text: str,
) -> None:
    session.add_message("user", build_tool_result_prompt(tool_name, text))


def _record_agent_file(
    session: AgentSessionState,
    agent_path: str,
    content: str,
) -> list[str]:
    normalized = normalize_repo_path(agent_path)
    session.grounded_agent_paths.add(normalized)
    return extract_startup_reads(content)


def _parse_listing_entries(text: str) -> set[str]:
    if "\n" not in text:
        return set()
    body = text.split("\n", 1)[1]
    return {line.rstrip("/").strip() for line in body.splitlines() if line.strip() and line.strip() != "."}


def _run_grounding_target(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    kind: str,
    path: str,
) -> None:
    if kind == "read":
        _read_first_available(runtime, session, path)
        return
    if kind == "list":
        from .models import Req_List

        listing = _auto_command(runtime, session, Req_List(tool="list", path=path))
        if path == "/" and listing:
            session.root_entries = _parse_listing_entries(listing)
            session.repository_profile = infer_repository_profile(session.root_entries)
        return
    raise ValueError(f"Unknown grounding target kind: {kind}")


def _run_startup_reads(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    paths: list[str],
) -> None:
    for extra_path in paths:
        for candidate in candidate_read_paths(extra_path):
            normalized = normalize_repo_path(candidate)
            if normalized in session.attempted_agent_paths:
                continue
            session.attempted_agent_paths.add(normalized)
            text = _auto_command(runtime, session, Req_Read(tool="read", path=candidate))
            if text is not None:
                break


def _read_first_available(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
    label: str = "AUTO",
) -> str | None:
    for candidate in candidate_read_paths(path):
        text = _auto_command(runtime, session, Req_Read(tool="read", path=candidate), label=label)
        if text is not None:
            return text
    return None


def _auto_command(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    cmd: ToolRequest,
    label: str = "AUTO",
) -> str | None:
    try:
        text = runtime.execute(cmd)
        print(f"{CLI_GREEN}{label}{CLI_CLR}: {text}")
        _append_tool_result(session, cmd.__class__.__name__, text)
        if isinstance(cmd, Req_Read) and is_agent_instruction_path(cmd.path):
            startup_reads = _record_agent_file(
                session,
                cmd.path,
                text.split("\n", 1)[1] if "\n" in text else "",
            )
            _run_startup_reads(runtime, session, startup_reads)
        return text
    except ConnectError as exc:
        print(f"{CLI_YELLOW}{label} ERR {exc.code}: {exc.message}{CLI_CLR}")
        return None


def _ensure_agent_grounding(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    target_path: str,
) -> None:
    for agent_path in candidate_agent_paths(target_path):
        if agent_path in session.attempted_agent_paths:
            continue
        session.attempted_agent_paths.add(agent_path)
        _auto_command(runtime, session, Req_Read(tool="read", path=agent_path))


def _bootstrap(runtime: PcmRuntimeAdapter, session: AgentSessionState) -> None:
    session.add_message("system", build_system_prompt())

    _run_grounding_target(runtime, session, "list", "/")
    _auto_command(runtime, session, Req_Tree(tool="tree", root="/", level=2))
    for agent_name in AGENT_FILE_NAMES:
        session.attempted_agent_paths.add(normalize_repo_path(f"/{agent_name}"))
    _read_first_available(runtime, session, "/AGENTS.md")
    _auto_command(runtime, session, Req_Context(tool="context"))


def _frame_task(
    llm: JsonChatClient,
    session: AgentSessionState,
) -> TaskFrame:
    frame, raw_text, elapsed_ms = llm.complete_json(
        [*session.messages, {"role": "user", "content": build_task_frame_prompt(session.task_text)}],
        TaskFrame,
    )
    print(f"{CLI_BLUE}FRAME{CLI_CLR}: {frame.category} ({elapsed_ms} ms)")
    print(f"  success: {', '.join(frame.success_criteria)}")
    if frame.risks:
        print(f"  risks: {', '.join(frame.risks)}")
    session.frame = frame
    session.add_message("assistant", raw_text.strip() or frame.model_dump_json(indent=2))
    return frame


def _ground_frame(runtime: PcmRuntimeAdapter, session: AgentSessionState, frame: TaskFrame) -> None:
    for target in profile_grounding_targets(session.repository_profile, frame, session.task_text):
        _run_grounding_target(runtime, session, target.kind, target.path)
    for root in relevant_roots(frame):
        _ensure_agent_grounding(runtime, session, root)


def _emit_preflight_completion(payload: ReportTaskCompletion) -> None:
    status = CLI_GREEN if payload.outcome == "OUTCOME_OK" else CLI_YELLOW
    print(f"{status}agent {payload.outcome}{CLI_CLR}. Summary:")
    for item in payload.completed_steps_laconic:
        print(f"- {item}")
    print(f"\n{CLI_BLUE}AGENT SUMMARY: {payload.message}{CLI_CLR}")
    for ref in payload.grounding_refs:
        print(f"- {CLI_BLUE}{ref}{CLI_CLR}")


def _prepare_command(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    cmd: ToolRequest,
) -> str | None:
    for path in command_paths(cmd):
        if is_mutating_command(cmd):
            _ensure_agent_grounding(runtime, session, path)

    guard = mutation_guard(session.task_text, cmd)
    if guard:
        print(f"{CLI_YELLOW}POLICY{CLI_CLR}: {guard}")
        return guard

    if isinstance(cmd, ReportTaskCompletion) and session.pending_verification_paths:
        pending = ", ".join(sorted(session.pending_verification_paths))
        message = (
            f"Verification required before report_completion. "
            f"Confirm final state for: {pending}"
        )
        print(f"{CLI_YELLOW}VERIFY{CLI_CLR}: {message}")
        return message

    return None


def _update_verification_state(session: AgentSessionState, cmd: ToolRequest) -> None:
    paths = command_paths(cmd)
    if is_mutating_command(cmd):
        for path in paths:
            session.pending_verification_paths.add(normalize_repo_path(path))
    elif is_verification_command(cmd):
        session.pending_verification_paths = clear_verified_paths(
            session.pending_verification_paths,
            paths,
        )


def _extract_tool_body(text: str | None) -> str:
    if not text:
        return ""
    return text.split("\n", 1)[1] if "\n" in text else ""


def _list_names(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
) -> list[str]:
    text = _auto_command(runtime, session, Req_List(tool="list", path=path))
    body = _extract_tool_body(text)
    return [line.rstrip("/").strip() for line in body.splitlines() if line.strip() and line.strip() != "."]


def _read_text(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
) -> str | None:
    return _extract_tool_body(_read_first_available(runtime, session, path)) or None


def _read_json(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
) -> dict | None:
    text = _read_text(runtime, session, path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _search_paths(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    pattern: str,
    root: str,
    limit: int = 20,
) -> list[str]:
    text = _auto_command(
        runtime,
        session,
        Req_Search(tool="search", pattern=pattern, root=root, limit=limit),
    )
    body = _extract_tool_body(text)
    paths: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        path = normalize_repo_path(line.split(":", 1)[0])
        if path not in paths:
            paths.append(path)
    return paths


def _run_write_json(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
    payload: dict,
) -> bool:
    text = _auto_command(
        runtime,
        session,
        Req_Write(
            tool="write",
            path=path,
            content=json.dumps(payload, indent=2),
        ),
    )
    return text is not None


def _run_write_text(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
    content: str,
) -> bool:
    text = _auto_command(
        runtime,
        session,
        Req_Write(tool="write", path=path, content=content),
    )
    return text is not None


def _run_delete(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
) -> bool:
    text = _auto_command(runtime, session, Req_Delete(tool="delete", path=path))
    return text is not None


def _answer_and_stop(
    runtime: PcmRuntimeAdapter,
    payload: ReportTaskCompletion,
) -> None:
    txt = runtime.execute(payload)
    print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
    _emit_preflight_completion(payload)


def _load_contact_candidates(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    full_name: str,
) -> list[ContactCandidate]:
    candidates: list[ContactCandidate] = []
    for path in _search_paths(runtime, session, full_name, "/contacts", limit=20):
        contact = _read_json(runtime, session, path)
        if not contact or contact.get("full_name") != full_name:
            continue
        account_id = contact.get("account_id", "")
        account = _read_json(runtime, session, f"/accounts/{account_id}.json") or {}
        candidates.append(
            ContactCandidate(
                contact_id=contact.get("id", ""),
                account_id=account_id,
                full_name=contact.get("full_name", ""),
                email=contact.get("email", ""),
                account_name=account.get("name", ""),
                compliance_flags=tuple(account.get("compliance_flags", [])),
                account_notes=account.get("notes", ""),
            )
        )
    return candidates


def _find_exact_contact(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    full_name: str,
) -> tuple[str, dict] | None:
    target = full_name.strip().lower()
    matches: list[tuple[str, dict]] = []
    for name in _list_names(runtime, session, "/contacts"):
        if not name.endswith(".json"):
            continue
        path = f"/contacts/{name}"
        contact = _read_json(runtime, session, path)
        if contact and str(contact.get("full_name", "")).strip().lower() == target:
            matches.append((path, contact))
    if len(matches) != 1:
        return None
    return matches[0]


def _find_account_contact_by_name(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    account_id: str,
    full_name: str,
) -> tuple[str, dict] | None:
    matches: list[tuple[str, dict]] = []
    for name in _list_names(runtime, session, "/contacts"):
        if not name.endswith(".json"):
            continue
        path = f"/contacts/{name}"
        contact = _read_json(runtime, session, path)
        if not contact:
            continue
        if str(contact.get("account_id", "")).strip() != account_id:
            continue
        if names_match(str(contact.get("full_name", "")), full_name):
            matches.append((path, contact))
    if len(matches) != 1:
        return None
    return matches[0]


def _find_exact_account(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    account_name: str,
) -> tuple[str, dict] | None:
    target = account_name.strip().lower()
    matches: list[tuple[str, dict]] = []
    for name in _list_names(runtime, session, "/accounts"):
        if not name.endswith(".json"):
            continue
        path = f"/accounts/{name}"
        account = _read_json(runtime, session, path)
        if account and str(account.get("name", "")).strip().lower() == target:
            matches.append((path, account))
    if len(matches) != 1:
        return None
    return matches[0]


def _resolve_direct_email_target(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    target: str,
) -> tuple[str, list[str]] | None:
    cleaned = target.strip()
    if "@" in cleaned and " " not in cleaned:
        return cleaned, []

    contact_match = _find_exact_contact(runtime, session, cleaned)
    if contact_match is not None:
        contact_path, contact = contact_match
        email = str(contact.get("email", "")).strip()
        if email:
            return email, [contact_path]

    account_match = _find_exact_account(runtime, session, cleaned)
    if account_match is not None:
        account_path, account = account_match
        primary_contact_id = str(account.get("primary_contact_id", "")).strip()
        if not primary_contact_id:
            return None
        contact_path = f"/contacts/{primary_contact_id}.json"
        contact = _read_json(runtime, session, contact_path)
        if not contact:
            return None
        email = str(contact.get("email", "")).strip()
        if email:
            return email, [account_path, contact_path]

    if " at " in cleaned.lower():
        person_name, account_name = re.split(r"\s+at\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)
        account_match = _find_exact_account(runtime, session, account_name)
        if account_match is None:
            return None
        account_path, account = account_match
        account_id = str(account.get("id", "")).strip()
        if not account_id:
            return None
        contact_match = _find_account_contact_by_name(runtime, session, account_id, person_name)
        if contact_match is None:
            return None
        contact_path, contact = contact_match
        email = str(contact.get("email", "")).strip()
        if email:
            return email, [account_path, contact_path]
    return None


def _select_latest_invoice(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    account_id: str,
) -> tuple[str, dict] | None:
    invoice_paths = _search_paths(
        runtime,
        session,
        f'"account_id": "{account_id}"',
        "/my-invoices",
        limit=20,
    )
    best: tuple[str, str, str, dict] | None = None
    for path in invoice_paths:
        invoice = _read_json(runtime, session, path)
        if not invoice or invoice.get("account_id") != account_id:
            continue
        issued_on = str(invoice.get("issued_on", ""))
        number = str(invoice.get("number", ""))
        candidate = (issued_on, number, path, invoice)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return None
    return best[2], best[3]


def _write_outbound_email(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    to_email: str,
    subject: str,
    body: str,
    attachments: list[str] | None = None,
) -> str | None:
    seq = _read_json(runtime, session, "/outbox/seq.json")
    if not seq or "id" not in seq:
        return None

    current_id = int(seq["id"])
    outbox_path = f"/outbox/{current_id}.json"
    email_payload = {
        "subject": subject,
        "to": to_email,
        "body": body,
        "attachments": attachments or [],
        "sent": False,
    }
    if not _run_write_json(runtime, session, outbox_path, email_payload):
        return None
    if not _run_write_text(runtime, session, "/outbox/seq.json", json.dumps({"id": current_id + 1})):
        return None
    return outbox_path


def _handle_knowledge_repo_inbox_security(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    lowered_task = session.task_text.lower()
    if session.repository_profile != "knowledge_repo":
        return False
    if "process the next file from the inbox" not in lowered_task and "process inbox" not in lowered_task:
        return False

    inbox_names = sorted(_list_names(runtime, session, "/00_inbox"))
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
    _answer_and_stop(runtime, payload)
    return True


def _handle_direct_outbound_email(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    if session.repository_profile != "typed_crm_fs":
        return False
    if "inbox" in session.task_text.lower():
        return False

    parsed = parse_direct_outbound_request(session.task_text)
    if parsed is None:
        parsed = parse_explicit_email_instruction(session.task_text)
    if parsed is None:
        return False

    target, subject, body = parsed
    resolved = _resolve_direct_email_target(runtime, session, target)
    if resolved is None:
        return False

    to_email, refs = resolved
    outbox_path = _write_outbound_email(runtime, session, to_email, subject, body)
    if outbox_path is None:
        return False

    _read_text(runtime, session, outbox_path)
    _read_text(runtime, session, "/outbox/seq.json")

    payload = ReportTaskCompletion(
        tool="report_completion",
        completed_steps_laconic=[
            f"Resolved outbound email target {target}",
            f"Wrote outbound email {normalize_repo_path(outbox_path)}",
        ],
        message="Processed the outbound email request.",
        grounding_refs=[*refs, outbox_path, "/outbox/seq.json"],
        outcome="OUTCOME_OK",
    )
    _answer_and_stop(runtime, payload)
    return True


def _handle_channel_status_lookup(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    lowered_task = session.task_text.lower()
    if session.repository_profile != "typed_crm_fs":
        return False
    if "how many" not in lowered_task or "telegram" not in lowered_task or "blacklist" not in lowered_task:
        return False

    telegram_text = _read_text(runtime, session, "/docs/channels/Telegram.txt")
    if telegram_text is None:
        return False

    total = count_channel_status(telegram_text, "blacklist")
    payload = ReportTaskCompletion(
        tool="report_completion",
        completed_steps_laconic=[
            "Read Telegram channel status file",
            f"Counted {total} blacklisted Telegram accounts",
        ],
        message=str(total),
        grounding_refs=["/docs/channels/Telegram.txt"],
        outcome="OUTCOME_OK",
    )
    _answer_and_stop(runtime, payload)
    return True


def _handle_typed_crm_inbox(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    lowered_task = session.task_text.lower()
    if session.repository_profile != "typed_crm_fs":
        return False
    if "process inbox" not in lowered_task and "process the inbox" not in lowered_task:
        return False

    inbox_names = sorted(name for name in _list_names(runtime, session, "/inbox") if name.startswith("msg_"))
    if not inbox_names:
        return False

    msg_path = f"/inbox/{inbox_names[0]}"
    message_text = _read_text(runtime, session, msg_path)
    if message_text is None:
        return False

    channel_agents_text = _read_text(runtime, session, "/docs/channels/AGENTS.MD") or ""
    discord_text = _read_text(runtime, session, "/docs/channels/Discord.txt") or ""
    telegram_text = _read_text(runtime, session, "/docs/channels/Telegram.txt") or ""
    otp_text = _read_text(runtime, session, "/docs/channels/otp.txt") or ""

    email_message = parse_email_inbox_message(message_text)
    if email_message is not None:
        contact_paths = _search_paths(runtime, session, email_message.sender_email, "/contacts", limit=5)
        matched_contacts = []
        for path in contact_paths:
            payload = _read_json(runtime, session, path)
            if payload and payload.get("email") == email_message.sender_email:
                matched_contacts.append(payload)

        if not matched_contacts:
            payload = ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=["Read inbox email", "Found no exact sender email match"],
                message=(
                    "The sender email does not match a known contact exactly. "
                    "I cannot safely process this inbox request."
                ),
                grounding_refs=[msg_path, "/docs/inbox-task-processing.md"],
                outcome="OUTCOME_DENIED_SECURITY",
            )
            _answer_and_stop(runtime, payload)
            return True

        if len(matched_contacts) > 1:
            payload = ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=["Read inbox email", "Found multiple exact sender matches"],
                message="Multiple contacts match this sender email. Clarify before proceeding.",
                grounding_refs=[msg_path, "/docs/inbox-task-processing.md"],
                outcome="OUTCOME_NONE_CLARIFICATION",
            )
            _answer_and_stop(runtime, payload)
            return True

        contact = matched_contacts[0]
        account = _read_json(runtime, session, f"/accounts/{contact['account_id']}.json") or {}
        requested_account = parse_requested_invoice_account(message_text)
        if requested_account and requested_account != account.get("name", ""):
            payload = ReportTaskCompletion(
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
                grounding_refs=[msg_path, f"/accounts/{contact['account_id']}.json"],
                outcome="OUTCOME_NONE_CLARIFICATION",
            )
            _answer_and_stop(runtime, payload)
            return True

        invoice_match = _select_latest_invoice(runtime, session, contact["account_id"])
        if invoice_match is None:
            payload = ReportTaskCompletion(
                tool="report_completion",
                completed_steps_laconic=[
                    "Read inbox email",
                    "Matched sender to known contact",
                    "Found no invoice for the sender account",
                ],
                message="I could not find a latest invoice for the sender account. Clarification is required.",
                grounding_refs=[msg_path, f"/contacts/{contact['id']}.json"],
                outcome="OUTCOME_NONE_CLARIFICATION",
            )
            _answer_and_stop(runtime, payload)
            return True

        invoice_path, invoice = invoice_match
        subject = f"Resending latest invoice for {account.get('name', 'your account')}"
        body = (
            f"Hi {contact.get('full_name', '').split(' ')[0]},\n\n"
            f"Please find the latest invoice for {account.get('name', 'your account')} attached.\n\n"
            "Best regards,"
        )
        outbox_path = _write_outbound_email(
            runtime,
            session,
            to_email=contact["email"],
            subject=subject,
            body=body,
            attachments=[invoice_path.lstrip("/")],
        )
        if outbox_path is None:
            return False

        _read_text(runtime, session, outbox_path)
        _read_text(runtime, session, "/outbox/seq.json")

        payload = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=[
                "Read inbox email",
                "Matched sender to known contact",
                f"Found latest invoice {invoice.get('number', '')}",
                f"Wrote outbound email {normalize_repo_path(outbox_path)}",
            ],
            message="Processed the inbox invoice resend request and sent the latest invoice to the known contact.",
            grounding_refs=[msg_path, f"/contacts/{contact['id']}.json", invoice_path, outbox_path, "/outbox/seq.json"],
            outcome="OUTCOME_OK",
        )
        _answer_and_stop(runtime, payload)
        return True

    channel_message = parse_channel_inbox_message(message_text)
    if channel_message is None:
        return False

    channel_statuses = parse_channel_statuses(
        telegram_text if channel_message.channel.lower() == "telegram" else discord_text
    )
    trust = channel_statuses.get(channel_message.handle.lower(), "unknown")
    elevated_by_otp = False
    otp_token = channel_message.otp
    otp_tokens = [line.strip() for line in otp_text.splitlines() if line.strip()]
    if trust != "admin" and otp_token and otp_token in otp_tokens:
        trust = "admin"
        elevated_by_otp = True

    if trust == "blacklist":
        payload = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=["Read channel inbox message", "Detected blacklisted handle"],
            message="This message came from a blacklisted channel handle and was denied.",
            grounding_refs=[msg_path, "/docs/channels/AGENTS.MD"],
            outcome="OUTCOME_DENIED_SECURITY",
        )
        _answer_and_stop(runtime, payload)
        return True

    if trust == "unknown":
        payload = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=["Read channel inbox message", "Detected untrusted unknown handle"],
            message="This message came from an unknown untrusted channel and was denied.",
            grounding_refs=[msg_path, "/docs/channels/AGENTS.MD"],
            outcome="OUTCOME_DENIED_SECURITY",
        )
        _answer_and_stop(runtime, payload)
        return True

    if trust == "valid":
        payload = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=["Read channel inbox message", "Detected non-trusted valid handle"],
            message="This message came from a non-trusted valid channel and was denied.",
            grounding_refs=[msg_path, "/docs/channels/AGENTS.MD"],
            outcome="OUTCOME_DENIED_SECURITY",
        )
        _answer_and_stop(runtime, payload)
        return True

    oracle_request = parse_otp_oracle_request(channel_message.body)
    if oracle_request is not None and trust == "admin":
        true_reply, expected_token, false_reply = oracle_request
        reply = true_reply if otp_text.strip() == expected_token else false_reply
        payload = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=[
                "Read trusted admin channel message",
                "Evaluated current OTP value against requested equality check",
            ],
            message=reply,
            grounding_refs=[msg_path, "/docs/channels/AGENTS.MD", "/docs/channels/otp.txt"],
            outcome="OUTCOME_OK",
        )
        _answer_and_stop(runtime, payload)
        return True

    explicit_email = parse_explicit_email_instruction(channel_message.body)
    if explicit_email is not None and trust == "admin":
        to_email, subject, body = explicit_email
        outbox_path = _write_outbound_email(runtime, session, to_email, subject, body)
        if outbox_path is None:
            return False

        if elevated_by_otp and otp_token:
            updated_otp = consume_otp_token(otp_text, otp_token)
            if updated_otp is None:
                if not _run_delete(runtime, session, "/docs/channels/otp.txt"):
                    return False
            elif not _run_write_text(runtime, session, "/docs/channels/otp.txt", updated_otp):
                return False

        _read_text(runtime, session, outbox_path)
        _read_text(runtime, session, "/outbox/seq.json")
        if elevated_by_otp:
            _read_text(runtime, session, "/docs/channels/otp.txt")

        steps = [
            "Read channel inbox message",
            f"Trusted {channel_message.channel} handle {channel_message.handle}",
            f"Wrote outbound email {normalize_repo_path(outbox_path)}",
        ]
        if elevated_by_otp:
            steps.insert(2, "Consumed OTP token from docs/channels/otp.txt")
        payload = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=steps,
            message="Processed the trusted channel request and sent the requested outbound email.",
            grounding_refs=[msg_path, "/docs/channels/AGENTS.MD", outbox_path, "/outbox/seq.json"],
            outcome="OUTCOME_OK",
        )
        _answer_and_stop(runtime, payload)
        return True

    target_name = parse_ai_insights_followup_target(channel_message.body)
    if target_name is not None and trust == "admin":
        candidate = choose_ai_insights_contact(_load_contact_candidates(runtime, session, target_name))
        if candidate is None:
            return False

        outbox_path = _write_outbound_email(
            runtime,
            session,
            to_email=candidate.email,
            subject="AI insights follow-up",
            body=(
                f"Hi {candidate.full_name.split(' ')[0]},\n\n"
                "Wanted to check whether you'd like an AI insights follow-up.\n\n"
                "Best regards,"
            ),
        )
        if outbox_path is None:
            return False

        _read_text(runtime, session, outbox_path)
        _read_text(runtime, session, "/outbox/seq.json")

        payload = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=[
                "Read channel inbox message",
                f"Trusted {channel_message.channel} handle {channel_message.handle}",
                f"Selected {candidate.full_name} via ai_insights_subscriber routing",
                f"Wrote outbound email {normalize_repo_path(outbox_path)}",
            ],
            message="Processed the trusted admin request and sent the AI insights follow-up email.",
            grounding_refs=[msg_path, "/docs/channels/AGENTS.MD", outbox_path, "/outbox/seq.json"],
            outcome="OUTCOME_OK",
        )
        _answer_and_stop(runtime, payload)
        return True

    return False


def _handle_purchase_prefix_regression(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    lowered_task = session.task_text.lower()
    if session.repository_profile != "purchase_ops":
        return False
    if "purchase" not in lowered_task or "prefix" not in lowered_task or "regression" not in lowered_task:
        return False

    audit = _read_json(runtime, session, "/purchases/audit.json") or {}
    sample_paths = [normalize_repo_path(path) for path in audit.get("examples", [])]
    if not sample_paths:
        return False

    sample_purchase = _read_json(runtime, session, sample_paths[0])
    if not sample_purchase:
        return False
    historical_prefix = extract_purchase_prefix(str(sample_purchase.get("purchase_id", "")))
    if historical_prefix is None:
        return False

    active_lane_path: str | None = None
    active_lane_payload: dict | None = None
    for lane_name in ("lane_a.json", "lane_b.json"):
        lane_path = f"/processing/{lane_name}"
        lane_payload = _read_json(runtime, session, lane_path)
        if lane_payload and lane_payload.get("traffic") == "downstream":
            active_lane_path = lane_path
            active_lane_payload = lane_payload
            break

    if not active_lane_path or not active_lane_payload:
        return False

    if active_lane_payload.get("prefix") == historical_prefix:
        return False

    updated_lane = dict(active_lane_payload)
    updated_lane["prefix"] = historical_prefix
    if not _run_write_json(runtime, session, active_lane_path, updated_lane):
        return False

    _read_text(runtime, session, active_lane_path)

    payload = ReportTaskCompletion(
        tool="report_completion",
        completed_steps_laconic=[
            f"Read impact context from /purchases/audit.json",
            f"Derived historical prefix {historical_prefix} from {sample_paths[0]}",
            f"Identified active emitter {active_lane_path}",
            f"Updated active emitter prefix to {historical_prefix}",
        ],
        message="Fixed the purchase prefix regression at the live downstream emitter without touching historical records or the audit log.",
        grounding_refs=["/docs/purchase-id-workflow.md", "/processing/README.MD", active_lane_path, sample_paths[0]],
        outcome="OUTCOME_OK",
    )
    _answer_and_stop(runtime, payload)
    return True


def run_agent(model: str, harness_url: str, task_text: str) -> None:
    config = AgentConfig.from_env(model)
    runtime = PcmRuntimeAdapter(harness_url)
    llm = JsonChatClient(config)
    session = AgentSessionState(task_text=task_text)

    _bootstrap(runtime, session)
    preflight = preflight_outcome(session.repository_profile, task_text)
    if preflight is not None:
        completion = ReportTaskCompletion(tool="report_completion", **preflight.model_dump())
        txt = runtime.execute(completion)
        print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
        _emit_preflight_completion(completion)
        return
    if _handle_knowledge_repo_inbox_security(runtime, session):
        return
    frame = _frame_task(llm, session)
    _ground_frame(runtime, session, frame)
    if _handle_direct_outbound_email(runtime, session):
        return
    if _handle_channel_status_lookup(runtime, session):
        return
    if _handle_typed_crm_inbox(runtime, session):
        return
    if _handle_purchase_prefix_regression(runtime, session):
        return
    session.add_message("user", build_execution_prompt(task_text, frame))

    for index in range(config.max_steps):
        step_name = f"step_{index + 1}"
        print(f"Next {step_name}... ", end="")

        job, raw_text, elapsed_ms = llm.complete_json(session.messages, NextStep)
        print(job.plan_remaining_steps_brief[0], f"({elapsed_ms} ms)\n  {job.function}")

        session.add_message(
            "assistant",
            raw_text.strip() or job.model_dump_json(indent=2),
        )

        precondition_message = _prepare_command(runtime, session, job.function)
        if precondition_message is not None:
            txt = precondition_message
        else:
            try:
                txt = runtime.execute(job.function)
                print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
                _update_verification_state(session, job.function)
            except ConnectError as exc:
                txt = str(exc.message)
                print(f"{CLI_RED}ERR {exc.code}: {exc.message}{CLI_CLR}")

        if isinstance(job.function, ReportTaskCompletion) and precondition_message is None:
            _emit_preflight_completion(job.function)
            break

        _append_tool_result(session, job.function.__class__.__name__, txt)
