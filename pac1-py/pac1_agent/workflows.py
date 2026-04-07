from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from .capabilities import extract_task_intent
from .pathing import normalize_repo_path


@dataclass(frozen=True)
class EmailInboxMessage:
    sender_name: str
    sender_email: str
    subject: str
    body: str


@dataclass(frozen=True)
class ChannelInboxMessage:
    channel: str
    handle: str
    otp: str | None
    body: str


@dataclass(frozen=True)
class ContactCandidate:
    contact_id: str
    account_id: str
    full_name: str
    email: str
    account_name: str
    compliance_flags: tuple[str, ...]
    account_notes: str


@dataclass(frozen=True)
class ChannelStatusRequest:
    channel_name: str
    status: str


@dataclass(frozen=True)
class CrmLookupRequest:
    kind: str
    target: str


def parse_email_inbox_message(text: str) -> EmailInboxMessage | None:
    match = re.search(r"^From:\s*(.*?)\s*<([^>]+)>\s*$", text, re.MULTILINE)
    if match is None:
        return None

    subject_match = re.search(r"^Subject:\s*(.*?)\s*$", text, re.MULTILINE)
    subject = subject_match.group(1).strip() if subject_match else ""

    body = text.split("\n\n", 1)[1].strip() if "\n\n" in text else ""
    return EmailInboxMessage(
        sender_name=match.group(1).strip(),
        sender_email=match.group(2).strip(),
        subject=subject,
        body=body,
    )


def parse_channel_inbox_message(text: str) -> ChannelInboxMessage | None:
    match = re.search(r"^Channel:\s*([^,]+),\s*Handle:\s*(.*?)\s*$", text, re.MULTILINE)
    if match is None:
        return None

    otp_match = re.search(r"^OTP:\s*(\S+)\s*$", text, re.MULTILINE)
    body = text.split("\n\n", 1)[1].strip() if "\n\n" in text else ""
    return ChannelInboxMessage(
        channel=match.group(1).strip(),
        handle=match.group(2).strip(),
        otp=otp_match.group(1).strip() if otp_match else None,
        body=body,
    )


def parse_requested_invoice_account(text: str) -> str | None:
    match = re.search(r"invoice for ([^.?\n]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_thread_discard_target(task_text: str) -> str | None:
    match = re.search(r"discard thread ([^.\n]+?) entirely", task_text, re.IGNORECASE)
    if match is None:
        return None
    name = match.group(1).strip().strip("'\"")
    if not name.endswith(".md"):
        name = f"{name}.md"
    return name


def parse_explicit_capture_request(task_text: str) -> tuple[str, str | None] | None:
    match = re.search(
        r"take\s+(00_inbox/\S+)\s+from inbox,\s+capture it into(?:\s+into)?\s+'([^']+)'\s+folder",
        task_text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return f"/{match.group(1).lstrip('/')}", match.group(2).strip()


def parse_email_lookup_target(task_text: str) -> str | None:
    return _extract_first_target(
        task_text,
        (
            r"email address of (.+?)(?:[?.]|$)",
            r"(?:email|address) for (.+?)(?:[?.]|$)",
            r"(?:what(?:'s| is)|give me|share)\s+(.+?)'s\s+email(?:\s+address)?(?:[?.]|$)",
        ),
    )


def parse_invoice_creation_request(task_text: str) -> tuple[str, list[dict[str, int]]] | None:
    header = re.search(r"create invoice (\S+) with \d+ lines?:\s*(.+)$", task_text, re.IGNORECASE)
    if header is None:
        return None

    invoice_number = header.group(1).strip().rstrip(".,")
    lines_blob = header.group(2)
    line_matches = re.findall(r"'([^']+)'\s*-\s*(\d+)", lines_blob)
    if not line_matches:
        return None
    return (
        invoice_number,
        [{"name": name.strip(), "amount": int(amount)} for name, amount in line_matches],
    )


def parse_two_week_followup_account(task_text: str) -> str | None:
    match = re.search(r"^(.+?) asked to reconnect in two weeks", task_text, re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).strip()


def parse_followup_reschedule_request(task_text: str) -> tuple[str, str] | None:
    for pattern in (
        r"^(.+?) asked to move the next follow-up to (\d{4}-\d{2}-\d{2})",
        r"^set the next (?:follow-up|touchpoint) with (.+?) to (\d{4}-\d{2}-\d{2})",
        r"^reschedule the (?:follow-up|reminder|touchpoint) for (.+?) to (\d{4}-\d{2}-\d{2})",
        r"^move the next (?:follow-up|touchpoint) with (.+?) to (\d{4}-\d{2}-\d{2})",
        r"^bump the next (?:follow-up|touchpoint) with (.+?) to (\d{4}-\d{2}-\d{2})",
        r"^move the (?:reminder|touchpoint) for (.+?) out to (\d{4}-\d{2}-\d{2})",
    ):
        exact_match = re.search(pattern, task_text, re.IGNORECASE)
        if exact_match is not None:
            return exact_match.group(1).strip(), exact_match.group(2)

    account_name = parse_two_week_followup_account(task_text)
    if account_name is None:
        return None
    return account_name, ""


def parse_primary_contact_email_account(task_text: str) -> str | None:
    return _extract_first_target(
        task_text,
        (
            r"primary contact for (.+?)(?:\s+account)?(?:[?.]|$)",
            r"(?:email|address) (?:for|of) (?:the )?(?:primary|main) contact (?:for|on) (.+?)(?:\s+account)?(?:[?.]|$)",
            r"(?:email|address) (?:for|of) (?:the )?(?:primary|main) stakeholder (?:for|on) (.+?)(?:\s+account)?(?:[?.]|$)",
            r"(?:email|address) (?:for|of) the point of contact (?:for|on) (.+?)(?:\s+account)?(?:[?.]|$)",
            r"(?:what(?:'s| is)|give me|share)\s+the\s+(?:email|address)\s+for\s+the\s+(?:primary|main)\s+contact\s+(?:for|on)\s+(.+?)(?:\s+account)?(?:[?.]|$)",
        ),
    )


def parse_account_manager_email_account(task_text: str) -> str | None:
    return _extract_first_target(
        task_text,
        (
            r"account manager for (.+?)(?:\s+account)?(?:[?.]|$)",
            r"(?:email|address).*?(?:account manager|account lead|lead) (?:for|on) (.+?)(?:\s+account)?(?:[?.]|$)",
            r"(?:email|address) (?:for|of) whoever manages (.+?)(?:\s+account)?(?:[?.]|$)",
            r"(?:email|address) (?:for|of) whoever owns (.+?)(?:\s+account)?(?:[?.]|$)",
            r"(?:what(?:'s| is)|give me|share)\s+the\s+(?:email|address)\s+for\s+whoever manages\s+(.+?)(?:\s+account)?(?:[?.]|$)",
            r"(?:what(?:'s| is)|give me|share)\s+the\s+(?:email|address)\s+for\s+whoever owns\s+(.+?)(?:\s+account)?(?:[?.]|$)",
        ),
    )


def parse_manager_account_listing_request(task_text: str) -> str | None:
    return _extract_first_target(
        task_text,
        (
            r"which accounts are managed by (.+?)(?:[?.]|$)",
            r"which accounts does (.+?) manage(?:[?.]|$)",
            r"list the accounts under (.+?)(?:[?.]|$)",
            r"what accounts are under (.+?)(?:[?.]|$)",
            r"which accounts belong to (.+?) as account manager(?:[?.]|$)",
        ),
    )


def parse_legal_name_account_request(task_text: str) -> str | None:
    return _extract_first_target(
        task_text,
        (
            r"exact legal name of (.+?)(?:\s+account)?(?:[?.]|$)",
            r"formal company name of (.+?)(?:\s+account)?(?:[?.]|$)",
            r"legal entity name of (.+?)(?:\s+account)?(?:[?.]|$)",
            r"registered company name of (.+?)(?:\s+account)?(?:[?.]|$)",
            r"corporate name of (.+?)(?:\s+account)?(?:[?.]|$)",
        ),
    )


def parse_crm_lookup_request(task_text: str) -> CrmLookupRequest | None:
    parsers = (
        ("legal_name", parse_legal_name_account_request),
        ("primary_contact_email", parse_primary_contact_email_account),
        ("manager_email", parse_account_manager_email_account),
        ("managed_accounts", parse_manager_account_listing_request),
        ("contact_email", parse_email_lookup_target),
    )
    for kind, parser in parsers:
        target = parser(task_text)
        if target is not None:
            return CrmLookupRequest(kind=kind, target=target)
    return None


def parse_direct_capture_snippet_request(task_text: str) -> tuple[str, str, str] | None:
    match = re.search(
        r"capture this snippet from website\s+(\S+)\s+into\s+(\S+):\s+\"(.*)\"\s*$",
        task_text,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    domain = match.group(1).strip().rstrip(".,")
    target_path = f"/{match.group(2).strip().lstrip('/')}"
    snippet = match.group(3)
    return domain, normalize_repo_path(target_path), snippet


def _strip_matching_quotes(text: str) -> str:
    value = text.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def _extract_first_target(task_text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, task_text, re.IGNORECASE)
        if match is not None:
            return _strip_matching_quotes(match.group(1))
    return None


def parse_explicit_email_instruction(text: str) -> tuple[str, str, str] | None:
    match = re.search(
        r"write\s+(?:a\s+)?brief\s+email\s+to\s+(?P<recipient>\"[^\"]+\"|'[^']+'|\S+)\s+"
        r"with\s+subject\s+(?P<subject_quote>\"|')(?P<subject>.*?)(?P=subject_quote)\s+"
        r"and\s+body\s+(?P<body_quote>\"|')(?P<body>.*?)(?P=body_quote)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    return (
        _strip_matching_quotes(match.group("recipient")),
        match.group("subject").strip(),
        match.group("body"),
    )


def parse_ai_insights_followup_target(text: str) -> str | None:
    match = re.search(r"Email (.+?) asking if they want AI insights follow-up", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_direct_outbound_request(text: str) -> tuple[str, str, str] | None:
    match = re.search(
        r"(?:send\s+email|email)\s+to\s+(.+?)\s+with\s+subject\s+\"([^\"]+)\"\s+and\s+body\s+\"([^\"]+)\"",
        text,
        re.IGNORECASE,
    )
    if match is not None:
        return _strip_matching_quotes(match.group(1)), match.group(2).strip(), match.group(3).strip()

    reminder_match = re.search(
        r"email\s+reminder\s+to\s+(.+?)\s+with\s+subject\s+\"([^\"]+)\"\s+and\s+about\s+\"([^\"]+)\"",
        text,
        re.IGNORECASE,
    )
    if reminder_match is not None:
        return (
            _strip_matching_quotes(reminder_match.group(1)),
            reminder_match.group(2).strip(),
            reminder_match.group(3).strip(),
        )

    followup_match = re.search(
        r"send\s+(?:a\s+)?short\s+follow-up\s+email\s+to\s+(.+?)\s+about\s+(.+?)(?:\.\s|$)",
        text,
        re.IGNORECASE,
    )
    if followup_match is None:
        return None
    return (
        _strip_matching_quotes(followup_match.group(1)),
        "Quick follow-up",
        f"Checking in about {followup_match.group(2).strip().rstrip('.')}.",
    )


def parse_otp_oracle_request(text: str) -> tuple[str, str, str] | None:
    match = re.search(
        r'Reply with exactly "([^"]+)" if .*?otp\.txt equals "([^"]+)".*?Otherwise reply with exactly "([^"]+)"',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def parse_channel_statuses(text: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if " - " not in line:
            continue
        handle, status = line.split(" - ", 1)
        handle_key = handle.strip().lower()
        if handle_key:
            statuses[handle_key] = status.strip().lower()
    return statuses


def is_inbox_processing_request(text: str) -> bool:
    return extract_task_intent(text).wants_inbox_processing


def parse_channel_status_lookup_request(
    text: str,
    channel_statuses: dict[str, set[str]],
) -> ChannelStatusRequest | None:
    lowered = " ".join(text.lower().split())
    if not any(marker in lowered for marker in ("how many", "count ", "number of", "total ")):
        return None

    matched_channel: str | None = None
    for channel_name in sorted(channel_statuses, key=len, reverse=True):
        for alias in _channel_aliases(channel_name):
            if alias in lowered:
                matched_channel = channel_name
                break
        if matched_channel is not None:
            break
    if matched_channel is None:
        return None

    matched_status: str | None = None
    for status in sorted(channel_statuses.get(matched_channel, set()), key=len, reverse=True):
        for alias in _status_aliases(status):
            if alias in lowered:
                matched_status = status
                break
        if matched_status is not None:
            break
    if matched_status is None:
        return None

    return ChannelStatusRequest(channel_name=matched_channel, status=matched_status)


def consume_otp_token(text: str, token: str) -> str | None:
    remaining = [line for line in text.splitlines() if line.strip() and line.strip() != token]
    if not remaining:
        return None
    return "\n".join(remaining) + "\n"


def count_channel_status(text: str, status: str) -> int:
    expected = status.strip().lower()
    total = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if " - " not in line:
            continue
        _, observed = line.split(" - ", 1)
        if observed.strip().lower() == expected:
            total += 1
    return total


def collect_channel_status_values(text: str) -> set[str]:
    return {status for status in parse_channel_statuses(text).values() if status}


def looks_suspicious_inbox_name(name: str) -> bool:
    lowered = name.lower()
    return any(
        token in lowered
        for token in ("exec", "approval", "override", "ignore", "prompt", "instruction", "admin", "priority", "escalation")
    )


def names_match(a: str, b: str) -> bool:
    tokens_a = Counter(re.findall(r"\w+", a.lower()))
    tokens_b = Counter(re.findall(r"\w+", b.lower()))
    return bool(tokens_a) and tokens_a == tokens_b


def choose_ai_insights_contact(candidates: list[ContactCandidate]) -> ContactCandidate | None:
    if len(candidates) == 1:
        return candidates[0]

    flagged = [
        candidate
        for candidate in candidates
        if "ai_insights_subscriber" in candidate.compliance_flags
        or "ai insights" in candidate.account_notes.lower()
    ]
    if len(flagged) == 1:
        return flagged[0]
    return None


def extract_purchase_prefix(purchase_id: str) -> str | None:
    match = re.match(r"^([a-zA-Z-]+)\d+$", purchase_id.strip())
    return match.group(1) if match else None


def _channel_aliases(channel_name: str) -> set[str]:
    base = channel_name.strip().lower()
    aliases = {
        base,
        base.replace("_", " "),
        base.replace("-", " "),
        f"{base} channel",
    }
    return {alias for alias in aliases if alias}


def _status_aliases(status: str) -> set[str]:
    base = status.strip().lower()
    aliases = {base}
    if base.endswith("list"):
        aliases.add(f"{base}ed")
    if not base.endswith("s"):
        aliases.add(f"{base}s")
    return {alias for alias in aliases if alias}
