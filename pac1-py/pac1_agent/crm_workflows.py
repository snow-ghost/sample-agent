from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable

from .pathing import normalize_repo_path
from .workflows import (
    ChannelStatusRequest,
    ContactCandidate,
    collect_channel_status_values,
    names_match,
    parse_channel_status_lookup_request,
)


ListNamesFn = Callable[[Any, Any, str], list[str]]
ReadJsonFn = Callable[[Any, Any, str], dict | None]
ReadTextFn = Callable[[Any, Any, str], str | None]
SearchPathsFn = Callable[[Any, Any, str, str, int], list[str]]
RunWriteJsonFn = Callable[[Any, Any, str, dict], bool]
RunWriteTextFn = Callable[[Any, Any, str, str], bool]


@dataclass(frozen=True)
class CrmWorkflowOps:
    list_names: ListNamesFn
    read_json: ReadJsonFn
    read_text: ReadTextFn
    search_paths: SearchPathsFn
    run_write_json: RunWriteJsonFn
    run_write_text: RunWriteTextFn


GENERIC_QUERY_STOPWORDS = {
    "a",
    "account",
    "accounts",
    "address",
    "an",
    "answer",
    "are",
    "by",
    "customer",
    "email",
    "exact",
    "for",
    "legal",
    "managed",
    "manager",
    "of",
    "only",
    "one",
    "per",
    "primary",
    "return",
    "the",
    "what",
    "which",
    "with",
}


def load_contact_candidates(
    ops: CrmWorkflowOps,
    runtime: Any,
    session: Any,
    full_name: str,
) -> list[ContactCandidate]:
    candidates: list[ContactCandidate] = []
    for path in ops.search_paths(runtime, session, full_name, "/contacts", 20):
        contact = ops.read_json(runtime, session, path)
        if not contact or contact.get("full_name") != full_name:
            continue
        account_id = contact.get("account_id", "")
        account = ops.read_json(runtime, session, f"/accounts/{account_id}.json") or {}
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


def find_exact_contact(
    ops: CrmWorkflowOps,
    runtime: Any,
    session: Any,
    full_name: str,
) -> tuple[str, dict] | None:
    target = full_name.strip().lower()
    matches: list[tuple[str, dict]] = []
    for name in ops.list_names(runtime, session, "/contacts"):
        if not name.endswith(".json"):
            continue
        path = f"/contacts/{name}"
        contact = ops.read_json(runtime, session, path)
        if contact and str(contact.get("full_name", "")).strip().lower() == target:
            matches.append((path, contact))
    if len(matches) != 1:
        return None
    return matches[0]


def find_account_contact_by_name(
    ops: CrmWorkflowOps,
    runtime: Any,
    session: Any,
    account_id: str,
    full_name: str,
) -> tuple[str, dict] | None:
    matches: list[tuple[str, dict]] = []
    for name in ops.list_names(runtime, session, "/contacts"):
        if not name.endswith(".json"):
            continue
        path = f"/contacts/{name}"
        contact = ops.read_json(runtime, session, path)
        if not contact:
            continue
        if str(contact.get("account_id", "")).strip() != account_id:
            continue
        if names_match(str(contact.get("full_name", "")), full_name):
            matches.append((path, contact))
    if len(matches) != 1:
        return None
    return matches[0]


def find_exact_account(
    ops: CrmWorkflowOps,
    runtime: Any,
    session: Any,
    account_name: str,
) -> tuple[str, dict] | None:
    target = account_name.strip().lower()
    matches: list[tuple[str, dict]] = []
    for name in ops.list_names(runtime, session, "/accounts"):
        if not name.endswith(".json"):
            continue
        path = f"/accounts/{name}"
        account = ops.read_json(runtime, session, path)
        if account and str(account.get("name", "")).strip().lower() == target:
            matches.append((path, account))
    if len(matches) != 1:
        return None
    return matches[0]


def iter_account_records(
    ops: CrmWorkflowOps,
    runtime: Any,
    session: Any,
) -> list[tuple[str, dict]]:
    records: list[tuple[str, dict]] = []
    for name in ops.list_names(runtime, session, "/accounts"):
        if not name.endswith(".json"):
            continue
        path = f"/accounts/{name}"
        account = ops.read_json(runtime, session, path)
        if account:
            records.append((path, account))
    return records


def descriptor_words(text: str) -> set[str]:
    lowered = text.lower()
    words = {word for word in re.findall(r"[a-z0-9]+", lowered) if word}
    if "dutch" in words:
        words.update({"netherlands", "benelux"})
    if "german" in words:
        words.update({"germany", "dach"})
    if "banking" in words:
        words.update({"finance", "bank"})
    if "shipping" in words or "port" in words or "logistics" in words:
        words.update({"shipping", "logistics", "port"})
    if "forecasting" in words or "consultancy" in words or "consulting" in words:
        words.update({"forecasting", "professional", "services"})
    if "retail" in words:
        words.add("retail")
    return words


def account_query_score(account: dict, query_text: str) -> int:
    query_words = descriptor_words(query_text)
    haystack = " ".join(
        [
            str(account.get("name", "")),
            str(account.get("legal_name", "")),
            str(account.get("industry", "")),
            str(account.get("region", "")),
            str(account.get("country", "")),
            str(account.get("tier", "")),
            str(account.get("status", "")),
            str(account.get("notes", "")),
            " ".join(str(item) for item in account.get("compliance_flags", [])),
        ]
    ).lower()
    score = 0
    for word in query_words:
        if word in GENERIC_QUERY_STOPWORDS or len(word) < 3:
            continue
        if word in haystack:
            score += 1

    account_name_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", str(account.get("name", "")).lower())
        if token not in GENERIC_QUERY_STOPWORDS and len(token) >= 3
    }
    score += 3 * len(account_name_tokens & query_words)

    flags = {str(item).lower() for item in account.get("compliance_flags", [])}
    if "security" in query_words and "review" in query_words and "security_review_open" in flags:
        score += 5
    if "ai" in query_words and "insights" in query_words and "ai_insights_subscriber" in flags:
        score += 5
    if "weak" in query_words and "sponsorship" in query_words and "weak" in haystack and "sponsorship" in haystack:
        score += 5

    return score


def resolve_account_by_descriptor(
    ops: CrmWorkflowOps,
    runtime: Any,
    session: Any,
    descriptor: str,
) -> tuple[str, dict] | None:
    exact = find_exact_account(ops, runtime, session, descriptor)
    if exact is not None:
        return exact

    scored: list[tuple[int, str, dict]] = []
    for path, account in iter_account_records(ops, runtime, session):
        score = account_query_score(account, descriptor)
        if score > 0:
            scored.append((score, path, account))

    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    if scored[0][0] < 3:
        return None
    return scored[0][1], scored[0][2]


def find_internal_contact_by_name(
    ops: CrmWorkflowOps,
    runtime: Any,
    session: Any,
    full_name: str,
) -> tuple[str, dict] | None:
    matches: list[tuple[str, dict]] = []
    for name in ops.list_names(runtime, session, "/contacts"):
        if not name.endswith(".json"):
            continue
        path = f"/contacts/{name}"
        contact = ops.read_json(runtime, session, path)
        if not contact or not names_match(str(contact.get("full_name", "")), full_name):
            continue
        role = str(contact.get("role", "")).strip().lower()
        tags = {str(item).strip().lower() for item in contact.get("tags", [])}
        if role == "account manager" or "account_manager" in tags:
            matches.append((path, contact))
    if len(matches) != 1:
        return None
    return matches[0]


def resolve_direct_email_target(
    ops: CrmWorkflowOps,
    runtime: Any,
    session: Any,
    target: str,
) -> tuple[str, list[str]] | None:
    cleaned = target.strip()
    if "@" in cleaned and " " not in cleaned:
        return cleaned, []

    contact_match = find_exact_contact(ops, runtime, session, cleaned)
    if contact_match is not None:
        contact_path, contact = contact_match
        email = str(contact.get("email", "")).strip()
        if email:
            return email, [contact_path]

    account_match = find_exact_account(ops, runtime, session, cleaned)
    if account_match is not None:
        account_path, account = account_match
        primary_contact_id = str(account.get("primary_contact_id", "")).strip()
        if not primary_contact_id:
            return None
        contact_path = f"/contacts/{primary_contact_id}.json"
        contact = ops.read_json(runtime, session, contact_path)
        if not contact:
            return None
        email = str(contact.get("email", "")).strip()
        if email:
            return email, [account_path, contact_path]

    if " at " in cleaned.lower():
        person_name, account_name = re.split(r"\s+at\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)
        account_match = find_exact_account(ops, runtime, session, account_name)
        if account_match is None:
            return None
        account_path, account = account_match
        account_id = str(account.get("id", "")).strip()
        if not account_id:
            return None
        contact_match = find_account_contact_by_name(ops, runtime, session, account_id, person_name)
        if contact_match is None:
            return None
        contact_path, contact = contact_match
        email = str(contact.get("email", "")).strip()
        if email:
            return email, [account_path, contact_path]

    account_match = resolve_account_by_descriptor(ops, runtime, session, cleaned)
    if account_match is not None:
        account_path, account = account_match
        primary_contact_id = str(account.get("primary_contact_id", "")).strip()
        if primary_contact_id:
            contact_path = f"/contacts/{primary_contact_id}.json"
            contact = ops.read_json(runtime, session, contact_path)
            if contact:
                email = str(contact.get("email", "")).strip()
                if email:
                    return email, [account_path, contact_path]
    return None


def select_latest_invoice(
    ops: CrmWorkflowOps,
    runtime: Any,
    session: Any,
    account_id: str,
) -> tuple[str, dict] | None:
    invoice_paths = ops.search_paths(runtime, session, f'"account_id": "{account_id}"', "/my-invoices", 20)
    best: tuple[str, str, str, dict] | None = None
    for path in invoice_paths:
        invoice = ops.read_json(runtime, session, path)
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


def write_outbound_email(
    ops: CrmWorkflowOps,
    runtime: Any,
    session: Any,
    to_email: str,
    subject: str,
    body: str,
    attachments: list[str] | None = None,
) -> str | None:
    seq = ops.read_json(runtime, session, "/outbox/seq.json")
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
    if not ops.run_write_json(runtime, session, outbox_path, email_payload):
        return None
    if not ops.run_write_text(runtime, session, "/outbox/seq.json", json.dumps({"id": current_id + 1})):
        return None
    return outbox_path


def parse_channel_status_request(
    ops: CrmWorkflowOps,
    runtime: Any,
    session: Any,
    task_text: str,
) -> ChannelStatusRequest | None:
    channel_statuses: dict[str, set[str]] = {}
    for name in ops.list_names(runtime, session, "/docs/channels"):
        if not name.endswith(".txt"):
            continue
        stem = name[:-4]
        if stem.lower() == "otp":
            continue
        channel_text = ops.read_text(runtime, session, f"/docs/channels/{name}")
        if channel_text is None:
            continue
        statuses = collect_channel_status_values(channel_text)
        if not statuses:
            continue
        channel_statuses[stem] = statuses
    if not channel_statuses:
        return None
    return parse_channel_status_lookup_request(task_text, channel_statuses)


def read_named_channel_status_text(
    ops: CrmWorkflowOps,
    runtime: Any,
    session: Any,
    channel_name: str,
) -> tuple[str | None, str | None]:
    desired = channel_name.strip().lower()
    for name in ops.list_names(runtime, session, "/docs/channels"):
        if not name.lower().endswith(".txt"):
            continue
        stem = name[:-4]
        if stem.lower() == "otp":
            continue
        if stem.lower() != desired:
            continue
        path = f"/docs/channels/{name}"
        return path, ops.read_text(runtime, session, path)
    return None, None


def normalize_search_paths(paths: list[str]) -> list[str]:
    return [normalize_repo_path(path) for path in paths]
