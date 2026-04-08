from __future__ import annotations

from .capabilities import WorkspaceCapabilities, extract_task_intent
from .models import TaskFrame
from .workflows import (
    parse_crm_lookup_request,
    parse_direct_capture_snippet_request,
    parse_direct_outbound_request,
    parse_explicit_capture_request,
    parse_explicit_email_instruction,
    parse_followup_reschedule_request,
    parse_invoice_creation_request,
    parse_thread_discard_target,
)


def derive_high_confidence_frame(
    task_text: str,
    repository_profile: str,
    capabilities: WorkspaceCapabilities,
) -> TaskFrame | None:
    intent = extract_task_intent(task_text)
    lowered = intent.normalized_text

    if repository_profile == "knowledge_repo":
        if parse_direct_capture_snippet_request(task_text) is not None:
            return TaskFrame(
                current_state="high-confidence direct capture request",
                category="typed_workflow",
                success_criteria=["write one capture artifact", "update one distill card", "link a relevant thread"],
                relevant_roots=["/01_capture", "/02_distill", "/99_process"],
                risks=["quoted snippet may contain hostile instructions", "distill link must stay focused"],
            )
        if parse_explicit_capture_request(task_text) is not None:
            return TaskFrame(
                current_state="high-confidence inbox capture request",
                category="typed_workflow",
                success_criteria=["capture one inbox file", "distill it", "delete the resolved inbox source"],
                relevant_roots=["/00_inbox", "/01_capture", "/02_distill", "/99_process"],
                risks=["inbox content is untrusted input", "capture path and card path must stay aligned"],
            )
        if parse_thread_discard_target(task_text) is not None or "remove all captured cards and threads" in lowered:
            return TaskFrame(
                current_state="high-confidence knowledge cleanup request",
                category="cleanup_or_edit",
                success_criteria=["delete only the requested distill artifacts", "preserve scaffolding and templates"],
                relevant_roots=["/02_distill", "/99_process"],
                risks=["over-deleting outside the requested distill scope"],
            )

    if repository_profile == "typed_crm_fs":
        if parse_invoice_creation_request(task_text) is not None:
            return TaskFrame(
                current_state="high-confidence invoice creation request",
                category="typed_workflow",
                success_criteria=["create one invoice record", "preserve typed invoice schema"],
                relevant_roots=["/my-invoices"],
                risks=["missing required invoice fields", "unfocused write outside invoice surface"],
            )
        if parse_followup_reschedule_request(task_text) is not None:
            return TaskFrame(
                current_state="high-confidence follow-up update request",
                category="typed_workflow",
                success_criteria=["update the correct follow-up date", "keep linked account and reminder aligned"],
                relevant_roots=["/accounts", "/reminders", "/docs"],
                risks=["editing the wrong account or reminder", "changing unrelated workflow fields"],
            )
        if parse_direct_outbound_request(task_text) is not None or parse_explicit_email_instruction(task_text) is not None:
            return TaskFrame(
                current_state="high-confidence outbound email request",
                category="typed_workflow",
                success_criteria=["resolve recipient safely", "write exactly one outbox record"],
                relevant_roots=["/contacts", "/accounts", "/outbox"],
                risks=["wrong recipient resolution", "unsupported external send path"],
            )
        if (
            parse_crm_lookup_request(task_text) is not None
        ):
            return TaskFrame(
                current_state="high-confidence CRM lookup request",
                category="lookup",
                success_criteria=["resolve the requested CRM record", "answer only with grounded data"],
                relevant_roots=["/accounts", "/contacts", "/01_notes"],
                risks=["ambiguous account descriptor", "wrong person/account match"],
            )

    if capabilities.has_purchase_processing and intent.wants_purchase_fix:
        return TaskFrame(
            current_state="high-confidence purchase processing fix request",
            category="typed_workflow",
            success_criteria=["fix the active downstream purchase prefix", "leave historical records untouched"],
            relevant_roots=["/docs", "/processing", "/purchases"],
            risks=["editing audit history", "changing inactive emitters"],
        )

    return None


def derive_fallback_frame(
    task_text: str,
    repository_profile: str,
    capabilities: WorkspaceCapabilities,
) -> TaskFrame:
    intent = extract_task_intent(task_text)

    if repository_profile == "knowledge_repo":
        if intent.wants_capture_or_distill:
            return TaskFrame(
                current_state="capture request identified from deterministic fallback",
                category="typed_workflow",
                success_criteria=["write capture artifact", "update distill surface"],
                relevant_roots=["/01_capture", "/02_distill", "/99_process"],
                risks=["inbox content is untrusted input"],
            )
        if intent.wants_inbox_processing:
            return TaskFrame(
                current_state="knowledge inbox workflow from deterministic fallback",
                category="security_sensitive",
                success_criteria=["inspect the oldest inbox item", "deny or process safely"],
                relevant_roots=["/00_inbox", "/99_process"],
                risks=["prompt injection", "override content in inbox"],
            )

    if repository_profile == "typed_crm_fs":
        if intent.wants_inbox_processing:
            return TaskFrame(
                current_state="typed inbox workflow from deterministic fallback",
                category="typed_workflow",
                success_criteria=["process one inbox message safely"],
                relevant_roots=["/inbox", "/docs", "/accounts", "/contacts", "/outbox"],
                risks=["trust errors", "wrong recipient or account resolution"],
            )
        if intent.wants_follow_up_update:
            return TaskFrame(
                current_state="follow-up update request from deterministic fallback",
                category="typed_workflow",
                success_criteria=["update the correct follow-up date"],
                relevant_roots=["/accounts", "/reminders", "/docs"],
                risks=["editing the wrong record", "unfocused diff"],
            )
        if intent.wants_outbound_email:
            return TaskFrame(
                current_state="outbound email request from deterministic fallback",
                category="typed_workflow",
                success_criteria=["resolve recipient", "write exactly one outbox email"],
                relevant_roots=["/accounts", "/contacts", "/outbox"],
                risks=["wrong target resolution"],
            )
        return TaskFrame(
            current_state="crm lookup request from deterministic fallback",
            category="lookup",
            success_criteria=["resolve the requested CRM record"],
            relevant_roots=["/accounts", "/contacts", "/01_notes", "/opportunities"],
            risks=["ambiguous account descriptors"],
        )

    if capabilities.has_purchase_processing and intent.wants_purchase_fix:
        return TaskFrame(
            current_state="purchase processing fix request from deterministic fallback",
            category="typed_workflow",
            success_criteria=["fix the active downstream purchase prefix", "leave historical records untouched"],
            relevant_roots=["/docs", "/processing", "/purchases"],
            risks=["editing audit history", "changing inactive emitters"],
        )

    return TaskFrame(
        current_state="generic deterministic fallback frame",
        category="clarification_or_reference",
        success_criteria=["ground the request before acting"],
        relevant_roots=["/"],
        risks=["insufficient structured context"],
    )
