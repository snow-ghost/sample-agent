from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Literal


RepositoryProfile = Literal["generic", "knowledge_repo", "typed_crm_fs", "purchase_ops"]


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
    word_count = len(task_text.strip().split())
    mentions_deictic_reference = bool(
        re.search(r"(^|\s)(this|that|these|those)(\s|$)", normalized_text)
    )

    wants_inbox_processing = "inbox" in normalized_text and any(
        marker in normalized_text
        for marker in (
            "process",
            "handle",
            "triage",
            "review",
            "resolve",
            "next file",
            "next message",
            "work through",
            "oldest inbox",
            "oldest message",
            "work the oldest",
        )
    ) or (
        any(marker in normalized_text for marker in ("inbound note", "inbound message"))
        and any(marker in normalized_text for marker in ("review", "act on it", "handle", "process"))
    )
    wants_outbound_email = bool(
        re.search(
            r"\b(email|e-mail|send email|write a brief email|write an email|reply by email)\b",
            normalized_text,
        )
    )
    wants_calendar_workflow = (
        "calendar invite" in normalized_text
        or ("calendar" in normalized_text and "invite" in normalized_text)
        or ("schedule" in normalized_text and "meeting" in normalized_text)
    )
    has_endpoint = bool(re.search(r"https?://|\bapi\.", normalized_text))
    wants_external_delivery = has_endpoint and any(
        marker in normalized_text
        for marker in ("upload", "deploy", "push", "post", "publish", "submit", "send", "call", "invoke")
    )
    wants_external_system_sync = any(
        marker in normalized_text for marker in ("sync", "mirror", "export", "replicate", "push")
    ) and any(
        system in normalized_text
        for system in (
            "salesforce",
            "hubspot",
            "zendesk",
            "marketo",
            "netsuite",
            "intercom",
            "airtable",
        )
    )
    wants_channel_status_lookup = any(
        marker in normalized_text for marker in ("how many", "count ", "number of", "total ")
    ) and any(
        marker in normalized_text
        for marker in ("channel", "telegram", "discord", "status", "blacklist", "verified", "admin", "valid")
    )
    wants_purchase_fix = "purchase" in normalized_text and any(
        marker in normalized_text
        for marker in ("prefix", "regression", "downstream", "lane", "workflow", "emitter", "processing")
    )

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
    )
