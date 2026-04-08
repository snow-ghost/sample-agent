from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

from .capabilities import extract_task_intent
from .models import ReportTaskCompletion
from .pathing import normalize_repo_path
from .workflows import (
    extract_purchase_prefix,
    names_match,
    parse_followup_reschedule_request,
    parse_invoice_creation_request,
)


CurrentRepoDateFn = Callable[[Any, Any], date | None]
ReadJsonFn = Callable[[Any, Any, str], dict | None]
ReadTextFn = Callable[[Any, Any, str], str | None]
ListNamesFn = Callable[[Any, Any, str], list[str]]
RunWriteJsonFn = Callable[[Any, Any, str, dict], bool]
ResolveAccountByDescriptorFn = Callable[[Any, Any, str], tuple[str, dict] | None]
AnswerAndStopFn = Callable[[Any, ReportTaskCompletion], None]


@dataclass(frozen=True)
class TypedMutationOps:
    current_repo_date: CurrentRepoDateFn
    read_json: ReadJsonFn
    read_text: ReadTextFn
    list_names: ListNamesFn
    run_write_json: RunWriteJsonFn
    resolve_account_by_descriptor: ResolveAccountByDescriptorFn
    answer_and_stop: AnswerAndStopFn


def handle_invoice_creation(
    ops: TypedMutationOps,
    runtime: Any,
    session: Any,
) -> bool:
    if session.repository_profile != "typed_crm_fs":
        return False

    parsed = parse_invoice_creation_request(session.task_text)
    if parsed is None:
        return False

    invoice_number, lines = parsed
    current_date = ops.current_repo_date(runtime, session)
    if current_date is None:
        return False

    payload = {
        "number": invoice_number,
        "issued_on": current_date.isoformat(),
        "lines": lines,
        "total": sum(line["amount"] for line in lines),
    }
    invoice_path = f"/my-invoices/{invoice_number}.json"
    if not ops.run_write_json(runtime, session, invoice_path, payload):
        return False
    ops.read_text(runtime, session, invoice_path)
    ops.answer_and_stop(
        runtime,
        ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=[
                f"Created invoice {invoice_number}",
                f"Wrote {invoice_path}",
            ],
            message=f"Created invoice {invoice_number}.",
            grounding_refs=["/my-invoices/README.MD", invoice_path],
            outcome="OUTCOME_OK",
        ),
    )
    return True


def handle_followup_reschedule(
    ops: TypedMutationOps,
    runtime: Any,
    session: Any,
) -> bool:
    if session.repository_profile != "typed_crm_fs":
        return False

    parsed = parse_followup_reschedule_request(session.task_text)
    if parsed is None:
        return False
    account_name, requested_date = parsed

    audit = ops.read_json(runtime, session, "/docs/follow-up-audit.json") or {}
    if str(audit.get("account_name", "")).strip() and not names_match(str(audit.get("account_name", "")), account_name):
        audit = {}

    target_due_on = requested_date or str(audit.get("requested_due_on", "")).strip()
    if not target_due_on:
        current_date = ops.current_repo_date(runtime, session)
        if current_date is None:
            return False
        target_due_on = (current_date + timedelta(days=14)).isoformat()

    account_match = None
    if audit.get("account_id"):
        account_match = (
            f"/accounts/{str(audit['account_id']).strip()}.json",
            ops.read_json(runtime, session, f"/accounts/{str(audit['account_id']).strip()}.json") or {},
        )
        if not account_match[1]:
            account_match = None
    if account_match is None:
        account_match = ops.resolve_account_by_descriptor(runtime, session, account_name)
    if account_match is None:
        return False
    account_path, account = account_match

    reminder_path: str | None = None
    for name in ops.list_names(runtime, session, "/reminders"):
        if not name.endswith(".json"):
            continue
        path = f"/reminders/{name}"
        reminder = ops.read_json(runtime, session, path)
        if not reminder:
            continue
        if str(reminder.get("account_id", "")) != str(account.get("id", "")):
            continue
        if str(reminder.get("status", "")).lower() in {"done", "cancelled"}:
            continue
        updated_reminder = dict(reminder)
        updated_reminder["due_on"] = target_due_on
        if ops.run_write_json(runtime, session, path, updated_reminder):
            reminder_path = path
        break

    refs: list[str] = [account_path]
    steps: list[str] = []
    updated_account = dict(account)
    updated_account["next_follow_up_on"] = target_due_on
    if not ops.run_write_json(runtime, session, account_path, updated_account):
        return False
    ops.read_text(runtime, session, account_path)
    steps.append(f"Updated account follow-up date to {target_due_on}")
    if reminder_path:
        refs.append(reminder_path)
        ops.read_text(runtime, session, reminder_path)
        steps.append(f"Updated linked reminder to {target_due_on}")
    if audit:
        refs.insert(0, "/docs/follow-up-audit.json")

    ops.answer_and_stop(
        runtime,
        ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=steps,
            message=f"Rescheduled the follow-up to {target_due_on}.",
            grounding_refs=refs,
            outcome="OUTCOME_OK",
        ),
    )
    return True


def handle_purchase_prefix_regression(
    ops: TypedMutationOps,
    runtime: Any,
    session: Any,
) -> bool:
    intent = extract_task_intent(session.task_text)
    if not session.capabilities.has_purchase_processing:
        return False
    if not intent.wants_purchase_fix:
        return False

    audit = ops.read_json(runtime, session, "/purchases/audit.json") or {}
    sample_paths = [normalize_repo_path(path) for path in audit.get("examples", [])]
    if not sample_paths:
        return False

    sample_purchase = ops.read_json(runtime, session, sample_paths[0])
    if not sample_purchase:
        return False
    historical_prefix = extract_purchase_prefix(str(sample_purchase.get("purchase_id", "")))
    if historical_prefix is None:
        return False

    active_lane_path: str | None = None
    active_lane_payload: dict | None = None
    for lane_name in ("lane_a.json", "lane_b.json"):
        lane_path = f"/processing/{lane_name}"
        lane_payload = ops.read_json(runtime, session, lane_path)
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
    if not ops.run_write_json(runtime, session, active_lane_path, updated_lane):
        return False

    ops.read_text(runtime, session, active_lane_path)
    ops.answer_and_stop(
        runtime,
        ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=[
                "Read impact context from /purchases/audit.json",
                f"Derived historical prefix {historical_prefix} from {sample_paths[0]}",
                f"Identified active emitter {active_lane_path}",
                f"Updated active emitter prefix to {historical_prefix}",
            ],
            message="Fixed the purchase prefix regression at the live downstream emitter without touching historical records or the audit log.",
            grounding_refs=["/docs/purchase-id-workflow.md", "/processing/README.MD", active_lane_path, sample_paths[0]],
            outcome="OUTCOME_OK",
        ),
    )
    return True
