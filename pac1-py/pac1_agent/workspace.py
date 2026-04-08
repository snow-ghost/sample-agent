from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

import re

from .capabilities import (
    RepositoryProfile,
    WorkspaceCapabilities,
    extract_task_intent,
    infer_repository_profile,
    infer_workspace_capabilities,
)
from .models import TaskFrame
from .models import Req_List, Req_Read, ToolRequest
from .pathing import AGENT_FILE_NAMES, normalize_repo_path
from .workflows import parse_invoice_creation_request


@dataclass(frozen=True)
class GroundingTarget:
    kind: Literal["read", "list"]
    path: str


def extract_startup_reads(agents_text: str) -> list[str]:
    startup_paths: list[str] = []
    for line in agents_text.splitlines():
        lower = line.lower()
        if "read" not in lower:
            continue
        if "start" not in lower and "session" not in lower and "always" not in lower:
            continue
        startup_paths.extend(re.findall(r"\((/[^)]+)\)", line))
        startup_paths.extend(re.findall(r"`(/[^`]+)`", line))
    deduped: list[str] = []
    for path in startup_paths:
        normalized = normalize_repo_path(path)
        if normalized not in deduped:
            deduped.append(normalized)
    return deduped


def relevant_roots(frame: TaskFrame) -> list[str]:
    roots: list[str] = []
    for root in frame.relevant_roots:
        normalized = normalize_repo_path(root)
        if normalized not in roots:
            roots.append(normalized)
    return roots


def candidate_agent_paths(target_path: str) -> list[str]:
    normalized = normalize_repo_path(target_path)
    path_obj = PurePosixPath(normalized)
    if "." in path_obj.name:
        dirs = list(path_obj.parents)
    else:
        dirs = [path_obj, *path_obj.parents]
    ordered: list[str] = []
    for directory in reversed(dirs):
        for agent_name in AGENT_FILE_NAMES:
            candidate = normalize_repo_path(f"{directory}/{agent_name}")
            if candidate in {"/AGENTS.md", "/AGENTS.MD"}:
                continue
            if candidate not in ordered:
                ordered.append(candidate)
    return ordered


def parse_root_entries_from_listing(text: str) -> set[str]:
    if "\n" not in text:
        return set()
    body = text.split("\n", 1)[1]
    return {line.rstrip("/").strip() for line in body.splitlines() if line.strip() and line.strip() != "."}


def parse_root_entries_from_tree(text: str) -> set[str]:
    entries: set[str] = set()
    if "\n" not in text:
        return entries
    body = text.split("\n", 1)[1]
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if not line or line == "/":
            continue
        match = re.match(r"^[├└]──\s+(.+)$", line)
        if match is None:
            continue
        entry = match.group(1).rstrip("/").strip()
        if entry:
            entries.add(entry)
    return entries


def derive_workspace_facts(root_entries: set[str]) -> tuple[RepositoryProfile, WorkspaceCapabilities]:
    profile = infer_repository_profile(root_entries)
    capabilities = infer_workspace_capabilities(root_entries=root_entries, profile=profile)
    return profile, capabilities


def _add_grounding_target(
    targets: list[GroundingTarget],
    seen: set[tuple[str, str]],
    kind: Literal["read", "list"],
    path: str,
) -> None:
    normalized = normalize_repo_path(path)
    key = (kind, normalized)
    if key in seen:
        return
    seen.add(key)
    targets.append(GroundingTarget(kind=kind, path=normalized))


def profile_grounding_targets(
    profile: RepositoryProfile,
    frame: TaskFrame,
    task_text: str,
) -> list[GroundingTarget]:
    intent = extract_task_intent(task_text)
    capabilities = infer_workspace_capabilities(profile=profile)
    text = intent.normalized_text
    targets: list[GroundingTarget] = []
    seen: set[tuple[str, str]] = set()

    if capabilities.profile == "typed_crm_fs":
        if any(token in text for token in ("invoice", "billing", "subscription")):
            _add_grounding_target(targets, seen, "read", "/my-invoices/README.MD")
            _add_grounding_target(targets, seen, "list", "/my-invoices")
        if intent.wants_outbound_email or any(token in text for token in ("subject", "body", "reminder", "follow-up")):
            _add_grounding_target(targets, seen, "read", "/outbox/README.MD")
            _add_grounding_target(targets, seen, "list", "/outbox")
            _add_grounding_target(targets, seen, "read", "/contacts/README.MD")
        if any(token in text for token in ("contact", "contacts", "account", "accounts")):
            _add_grounding_target(targets, seen, "read", "/contacts/README.MD")
            _add_grounding_target(targets, seen, "read", "/accounts/README.MD")
        if any(token in text for token in ("opportunity", "pipeline")):
            _add_grounding_target(targets, seen, "read", "/opportunities/README.MD")
        if any(token in text for token in ("reminder", "follow-up", "reschedule", "next week")):
            _add_grounding_target(targets, seen, "read", "/reminders/README.MD")
            _add_grounding_target(targets, seen, "read", "/accounts/README.MD")
        if intent.wants_inbox_processing:
            _add_grounding_target(targets, seen, "read", "/inbox/README.md")
            _add_grounding_target(targets, seen, "list", "/inbox")
            _add_grounding_target(targets, seen, "read", "/docs/inbox-task-processing.md")
            _add_grounding_target(targets, seen, "read", "/docs/inbox-msg-processing.md")
            _add_grounding_target(targets, seen, "list", "/docs/channels")
        if capabilities.has_channel_docs and (
            intent.wants_inbox_processing
            or intent.wants_channel_status_lookup
            or any(token in text for token in ("telegram", "discord", "otp", "channel", "status"))
        ):
            _add_grounding_target(targets, seen, "list", "/docs/channels")
            _add_grounding_target(targets, seen, "read", "/docs/channels/AGENTS.MD")
            _add_grounding_target(targets, seen, "read", "/docs/channels/Telegram.txt")
            _add_grounding_target(targets, seen, "read", "/docs/channels/Discord.txt")
            _add_grounding_target(targets, seen, "read", "/docs/channels/otp.txt")

    if capabilities.has_purchase_processing and intent.wants_purchase_fix:
        _add_grounding_target(targets, seen, "read", "/docs/purchase-id-workflow.md")
        _add_grounding_target(targets, seen, "read", "/docs/purchase-records.md")
        _add_grounding_target(targets, seen, "read", "/processing/README.MD")
        _add_grounding_target(targets, seen, "list", "/processing")
        if any(token in text for token in ("audit", "regression", "prefix", "history")):
            _add_grounding_target(targets, seen, "read", "/purchases/audit.json")

    for root in relevant_roots(frame):
        if root != "/":
            _add_grounding_target(targets, seen, "list", root)

    return targets


def local_fallback_commands(
    profile: RepositoryProfile,
    task_text: str,
) -> list[ToolRequest]:
    intent = extract_task_intent(task_text)
    text = intent.normalized_text

    if profile == "knowledge_repo":
        if intent.wants_cleanup_or_delete:
            return [
                Req_Read(tool="read", path="/99_process/document_cleanup.md"),
                Req_Read(tool="read", path="/02_distill/AGENTS.md"),
                Req_List(tool="list", path="/02_distill/cards"),
                Req_List(tool="list", path="/02_distill/threads"),
            ]
        if intent.wants_capture_or_distill:
            return [
                Req_Read(tool="read", path="/99_process/document_capture.md"),
                Req_List(tool="list", path="/00_inbox"),
                Req_List(tool="list", path="/01_capture/influential"),
                Req_List(tool="list", path="/02_distill"),
            ]
        if intent.wants_inbox_processing:
            return [
                Req_Read(tool="read", path="/99_process/process_tasks.md"),
                Req_List(tool="list", path="/00_inbox"),
                Req_List(tool="list", path="/02_distill"),
            ]
        return [Req_List(tool="list", path="/02_distill")]

    if profile == "typed_crm_fs":
        if intent.wants_lookup_email:
            return [
                Req_List(tool="list", path="/contacts"),
                Req_Read(tool="read", path="/contacts/README.MD"),
            ]
        if intent.wants_outbound_email:
            return [
                Req_List(tool="list", path="/contacts"),
                Req_Read(tool="read", path="/outbox/README.MD"),
                Req_List(tool="list", path="/accounts"),
            ]
        if parse_invoice_creation_request(task_text) is not None or any(
            token in text for token in ("invoice", "billing", "subscription")
        ):
            return [
                Req_List(tool="list", path="/my-invoices"),
                Req_Read(tool="read", path="/my-invoices/README.MD"),
            ]
        if intent.wants_follow_up_update:
            return [
                Req_List(tool="list", path="/reminders"),
                Req_Read(tool="read", path="/reminders/README.MD"),
                Req_List(tool="list", path="/accounts"),
            ]
        if intent.wants_inbox_processing:
            return [
                Req_List(tool="list", path="/inbox"),
                Req_Read(tool="read", path="/inbox/README.md"),
            ]
        return [Req_List(tool="list", path="/")]

    return [Req_List(tool="list", path="/")]
