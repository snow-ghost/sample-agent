from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal


RepositoryProfile = Literal["generic", "knowledge_repo", "typed_crm_fs", "purchase_ops"]
INBOX_REFERENCE_MARKERS = (
    "inbox",
    "queue",
    "incoming queue",
    "inbound queue",
    "inbound note",
    "inbound message",
    "incoming message",
    "incoming note",
    "inbound item",
    "incoming item",
    "inbox item",
    "inbound drop",
    "incoming drop",
)
INBOX_ACTION_MARKERS = (
    "process",
    "handle",
    "take care of",
    "triage",
    "review",
    "resolve",
    "work through",
    "work the oldest",
    "act on it",
)
INBOX_SEQUENCE_MARKERS = (
    "next file",
    "next message",
    "next inbox",
    "oldest inbox",
    "oldest message",
    "oldest pending",
    "oldest unresolved",
    "earliest unread",
    "earliest pending",
    "review the next",
)
OUTBOUND_EMAIL_MARKERS = (
    "email",
    "e-mail",
    "send email",
    "send a note",
    "send a message",
    "write an email",
    "write a brief email",
    "reply by email",
    "reply via email",
)
CALENDAR_MARKERS = (
    "calendar invite",
    "meeting invite",
    "schedule meeting",
    "schedule a meeting",
    "book meeting",
)
EXTERNAL_DELIVERY_MARKERS = ("upload", "deploy", "push", "post", "publish", "submit", "send", "call", "invoke")
EXTERNAL_SYSTEM_MARKERS = ("salesforce", "hubspot", "zendesk", "marketo", "netsuite", "intercom", "airtable")
CHANNEL_SURFACE_MARKERS = ("channel", "telegram", "discord")
CHANNEL_STATUS_MARKERS = ("status", "blacklist", "verified", "admin", "valid")
COUNT_STYLE_MARKERS = ("how many", "count ", "number of", "total ")
PURCHASE_MARKERS = ("purchase", "invoice id", "id prefix")
PURCHASE_FIX_MARKERS = ("prefix", "regression", "downstream", "lane", "workflow", "emitter", "processing")
FOLLOW_UP_MARKERS = ("follow-up", "follow up", "reminder", "next follow-up", "followup")
FOLLOW_UP_UPDATE_MARKERS = ("move", "reschedule", "postpone", "shift", "change", "fix", "update", "set to", "bump", "move out")
LOOKUP_EMAIL_MARKERS = ("email address", "primary contact email", "return only the email", "answer with the email")
CLEANUP_KNOWLEDGE_MARKERS = (
    "thread",
    "card",
    "captured",
    "remove",
    "discard",
    "delete",
    "start over",
    "clear",
    "purge",
)
CAPTURE_DISTILL_MARKERS = ("capture", "captur", "distill", "snippet", "excerpt")
ACTION_WORDS = frozenset({"process", "handle", "triage", "review", "resolve", "work", "act", "sort"})
ORDER_WORDS = frozenset({"next", "oldest", "earliest", "pending", "unread", "unresolved", "lowest", "first"})
INBOX_WORDS = frozenset({"inbox", "inbound", "incoming"})
ITEM_WORDS = frozenset(
    {"message", "messages", "note", "notes", "item", "items", "drop", "drops", "file", "files", "queue"}
)
EMAIL_WORDS = frozenset({"email", "mail", "address"})
OUTBOUND_WORDS = frozenset({"send", "write", "reply", "draft", "compose"})
ROLE_WORDS = frozenset({"primary", "contact", "manager", "lead", "owner"})
ROLE_ACTION_WORDS = frozenset({"manage", "manages", "managed", "own", "owns", "owned"})
ANSWER_STYLE_WORDS = frozenset({"return", "answer", "give", "just", "only", "what", "provide", "share"})
FOLLOW_UP_WORDS = frozenset({"follow", "followup", "reminder", "touchpoint", "reconnect", "checkin"})
UPDATE_WORDS = frozenset(
    {"move", "reschedule", "postpone", "shift", "change", "fix", "update", "set", "push", "delay", "bump"}
)
CAPTURE_WORDS = frozenset({"capture", "captur", "distill", "distillation", "snippet", "excerpt", "clip", "quote"})
DELETE_WORDS = frozenset({"remove", "discard", "delete", "purge", "clear"})
TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class WorkspaceCapabilities:
    profile: RepositoryProfile
    roots: frozenset[str]
    has_inbox: bool
    has_knowledge_inbox: bool
    has_outbox: bool
    has_contacts: bool
    has_accounts: bool
    has_channel_docs: bool
    has_invoices: bool
    has_purchase_processing: bool
    supports_outbound_email: bool
    supports_inbox_processing: bool
    supports_calendar: bool = False
    supports_external_delivery: bool = False
    supports_external_system_sync: bool = False


@dataclass(frozen=True)
class TaskIntent:
    normalized_text: str
    word_count: int
    mentions_deictic_reference: bool
    wants_inbox_processing: bool
    wants_outbound_email: bool
    wants_calendar_workflow: bool
    wants_external_delivery: bool
    wants_external_system_sync: bool
    wants_channel_status_lookup: bool
    wants_purchase_fix: bool
    wants_follow_up_update: bool
    wants_lookup_email: bool
    wants_capture_or_distill: bool
    wants_cleanup_or_delete: bool


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(text.lower()))


def _has_any_token(tokens: tuple[str, ...], words: frozenset[str]) -> bool:
    return any(token in words for token in tokens)


def _references_inbox_surface(text: str) -> bool:
    return _contains_any(text, INBOX_REFERENCE_MARKERS)


def _references_inbox_surface_tokens(tokens: tuple[str, ...]) -> bool:
    return _has_any_token(tokens, INBOX_WORDS) and _has_any_token(tokens, ITEM_WORDS)


def _wants_inbox_processing(text: str, tokens: tuple[str, ...]) -> bool:
    return (_references_inbox_surface(text) or _references_inbox_surface_tokens(tokens)) and (
        _contains_any(text, INBOX_ACTION_MARKERS)
        or _contains_any(text, INBOX_SEQUENCE_MARKERS)
        or _has_any_token(tokens, ACTION_WORDS)
        or _has_any_token(tokens, ORDER_WORDS)
    )


def _wants_outbound_email(text: str, tokens: tuple[str, ...]) -> bool:
    if _contains_any(text, OUTBOUND_EMAIL_MARKERS):
        return True
    if _has_any_token(tokens, EMAIL_WORDS) and _has_any_token(tokens, OUTBOUND_WORDS):
        return True
    return "subject" in tokens and "body" in tokens and (
        _has_any_token(tokens, EMAIL_WORDS) or _has_any_token(tokens, OUTBOUND_WORDS)
    )


def _wants_follow_up_update(text: str, tokens: tuple[str, ...]) -> bool:
    if _contains_any(text, FOLLOW_UP_MARKERS) and _contains_any(text, FOLLOW_UP_UPDATE_MARKERS):
        return True
    if _has_any_token(tokens, FOLLOW_UP_WORDS) and _has_any_token(tokens, UPDATE_WORDS):
        return True
    return "two" in tokens and "weeks" in tokens and "reconnect" in tokens


def _wants_lookup_email(text: str, tokens: tuple[str, ...]) -> bool:
    if _contains_any(text, LOOKUP_EMAIL_MARKERS):
        return True
    if not _has_any_token(tokens, EMAIL_WORDS):
        return False
    has_role_shape = (_has_any_token(tokens, ROLE_WORDS) or _has_any_token(tokens, ROLE_ACTION_WORDS)) and (
        "account" in tokens
    )
    has_contact_shape = "contact" in tokens and ("primary" in tokens or "account" in tokens)
    if not (has_role_shape or has_contact_shape):
        return False
    if _has_any_token(tokens, ANSWER_STYLE_WORDS):
        return True
    return not _has_any_token(tokens, OUTBOUND_WORDS)


def _wants_capture_or_distill(text: str, tokens: tuple[str, ...]) -> bool:
    if _contains_any(text, CAPTURE_DISTILL_MARKERS):
        return True
    return _has_any_token(tokens, CAPTURE_WORDS) and (
        '"' in text or "website" in tokens or "into" in tokens or "from" in tokens
    )


def _wants_cleanup_or_delete(text: str, tokens: tuple[str, ...]) -> bool:
    if _contains_any(text, CLEANUP_KNOWLEDGE_MARKERS):
        return True
    return _has_any_token(tokens, DELETE_WORDS) and any(token in {"thread", "card", "captured"} for token in tokens)


def infer_repository_profile(root_entries: set[str]) -> RepositoryProfile:
    normalized = {entry.lower() for entry in root_entries}
    if {"00_inbox", "01_capture", "02_distill"}.issubset(normalized):
        return "knowledge_repo"
    if {"accounts", "contacts", "outbox", "docs"}.issubset(normalized):
        return "typed_crm_fs"
    if {"purchases", "processing", "docs"}.issubset(normalized):
        return "purchase_ops"
    return "generic"


def infer_workspace_capabilities(
    root_entries: Iterable[str] | None = None,
    profile: RepositoryProfile | None = None,
) -> WorkspaceCapabilities:
    normalized = frozenset(entry.lower() for entry in (root_entries or []))
    resolved_profile = profile or infer_repository_profile(set(normalized))

    has_knowledge_inbox = "00_inbox" in normalized or resolved_profile == "knowledge_repo"
    has_inbox = "inbox" in normalized or resolved_profile == "typed_crm_fs"
    has_outbox = "outbox" in normalized or resolved_profile == "typed_crm_fs"
    has_contacts = "contacts" in normalized or resolved_profile == "typed_crm_fs"
    has_accounts = "accounts" in normalized or resolved_profile == "typed_crm_fs"
    has_channel_docs = ("docs" in normalized and (has_inbox or has_outbox)) or resolved_profile == "typed_crm_fs"
    has_invoices = "my-invoices" in normalized
    has_purchase_processing = (
        {"purchases", "processing", "docs"}.issubset(normalized) or resolved_profile == "purchase_ops"
    )

    return WorkspaceCapabilities(
        profile=resolved_profile,
        roots=normalized,
        has_inbox=has_inbox,
        has_knowledge_inbox=has_knowledge_inbox,
        has_outbox=has_outbox,
        has_contacts=has_contacts,
        has_accounts=has_accounts,
        has_channel_docs=has_channel_docs,
        has_invoices=has_invoices,
        has_purchase_processing=has_purchase_processing,
        supports_outbound_email=has_outbox,
        supports_inbox_processing=has_inbox or has_knowledge_inbox,
    )


def extract_task_intent(task_text: str) -> TaskIntent:
    normalized_text = " ".join(task_text.lower().split())
    tokens = _tokenize(normalized_text)
    word_count = len(task_text.strip().split())
    mentions_deictic_reference = bool(
        re.search(r"(^|\s)(this|that|these|those)(\s|$)", normalized_text)
    )

    wants_inbox_processing = _wants_inbox_processing(normalized_text, tokens)
    wants_outbound_email = _wants_outbound_email(normalized_text, tokens)
    wants_calendar_workflow = _contains_any(normalized_text, CALENDAR_MARKERS) or (
        "calendar" in normalized_text and "invite" in normalized_text
    )
    has_endpoint = bool(re.search(r"https?://|\bapi\.", normalized_text))
    wants_external_delivery = has_endpoint and _contains_any(normalized_text, EXTERNAL_DELIVERY_MARKERS)
    wants_external_system_sync = _contains_any(normalized_text, ("sync", "mirror", "export", "replicate", "push")) and _contains_any(
        normalized_text,
        EXTERNAL_SYSTEM_MARKERS,
    )
    wants_channel_status_lookup = _contains_any(normalized_text, COUNT_STYLE_MARKERS) and (
        _contains_any(normalized_text, CHANNEL_SURFACE_MARKERS)
        or _contains_any(normalized_text, CHANNEL_STATUS_MARKERS)
    )
    wants_purchase_fix = _contains_any(normalized_text, PURCHASE_MARKERS) and _contains_any(
        normalized_text,
        PURCHASE_FIX_MARKERS,
    )
    wants_follow_up_update = _wants_follow_up_update(normalized_text, tokens)
    wants_lookup_email = _wants_lookup_email(normalized_text, tokens)
    wants_capture_or_distill = _wants_capture_or_distill(normalized_text, tokens)
    wants_cleanup_or_delete = _wants_cleanup_or_delete(normalized_text, tokens)

    return TaskIntent(
        normalized_text=normalized_text,
        word_count=word_count,
        mentions_deictic_reference=mentions_deictic_reference,
        wants_inbox_processing=wants_inbox_processing,
        wants_outbound_email=wants_outbound_email,
        wants_calendar_workflow=wants_calendar_workflow,
        wants_external_delivery=wants_external_delivery,
        wants_external_system_sync=wants_external_system_sync,
        wants_channel_status_lookup=wants_channel_status_lookup,
        wants_purchase_fix=wants_purchase_fix,
        wants_follow_up_update=wants_follow_up_update,
        wants_lookup_email=wants_lookup_email,
        wants_capture_or_distill=wants_capture_or_distill,
        wants_cleanup_or_delete=wants_cleanup_or_delete,
    )
