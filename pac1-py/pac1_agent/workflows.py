from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from .capabilities import extract_task_intent


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


def _strip_matching_quotes(text: str) -> str:
    value = text.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


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
