from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
import time

from connectrpc.errors import ConnectError

from .capabilities import WorkspaceCapabilities, extract_task_intent, infer_workspace_capabilities
from .config import AgentConfig
from .crm_handlers import (
    CrmHandlerOps,
    handle_channel_status_lookup,
    handle_contact_email_lookup,
    handle_direct_outbound_email,
)
from .crm_inbox import CrmInboxOps, handle_typed_crm_inbox
from .crm_workflows import (
    CrmWorkflowOps,
    account_query_score,
    find_internal_contact_by_name,
    iter_account_records,
    load_contact_candidates,
    parse_channel_status_request,
    read_named_channel_status_text,
    resolve_account_by_descriptor,
    resolve_direct_email_target,
    select_latest_invoice,
    write_outbound_email,
)
from .fastpath import run_fastpath_handlers
from .framing import derive_fallback_frame, derive_high_confidence_frame
from .grounding import (
    auto_command as grounding_auto_command,
    bootstrap as grounding_bootstrap,
    ensure_agent_grounding as grounding_ensure_agent_grounding,
    ground_frame as grounding_ground_frame,
    read_first_available as grounding_read_first_available,
    run_grounding_target as grounding_run_grounding_target,
    run_startup_reads as grounding_run_startup_reads,
)
from .knowledge_repo import (
    KnowledgeRepoOps,
    choose_thread_path as choose_knowledge_thread_path,
    handle_direct_capture_snippet,
    handle_knowledge_repo_capture,
    handle_knowledge_repo_cleanup,
    handle_knowledge_repo_inbox_security,
)
from .llm import JsonChatClient, StructuredResponseError
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
    build_execution_prompt,
    build_system_prompt,
    build_task_frame_prompt,
    build_tool_result_prompt,
    build_workspace_context_prompt,
    command_paths,
    is_mutating_command,
    preflight_outcome,
)
from .pathing import AGENT_FILE_NAMES, candidate_read_paths, is_agent_instruction_path, normalize_repo_path
from .safety import pre_bootstrap_outcome
from .runtime import PcmRuntimeAdapter
from .telemetry import AgentRunTelemetry
from .tool_ops import (
    answer_and_stop as tool_answer_and_stop,
    current_repo_date as tool_current_repo_date,
    extract_tool_body,
    list_names as tool_list_names,
    read_json as tool_read_json,
    read_text as tool_read_text,
    run_delete as tool_run_delete,
    run_write_json as tool_run_write_json,
    run_write_text as tool_run_write_text,
    search_paths as tool_search_paths,
)
from .typed_mutations import (
    TypedMutationOps,
    handle_followup_reschedule,
    handle_invoice_creation,
    handle_purchase_prefix_regression,
)
from .verifier import next_pending_verification_paths, prepare_command
from .workspace import (
    candidate_agent_paths,
    derive_workspace_facts,
    extract_startup_reads,
    local_fallback_commands,
    parse_root_entries_from_listing,
    parse_root_entries_from_tree,
    profile_grounding_targets,
    relevant_roots,
)
from .workflows import (
    ChannelStatusRequest,
    ContactCandidate,
    names_match,
)

CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_CLR = "\x1B[0m"
CLI_BLUE = "\x1B[34m"
CLI_YELLOW = "\x1B[33m"
def _crm_ops() -> CrmWorkflowOps:
    return CrmWorkflowOps(
        list_names=_list_names,
        read_json=_read_json,
        read_text=_read_text,
        search_paths=_search_paths,
        run_write_json=_run_write_json,
        run_write_text=_run_write_text,
    )


def _crm_handler_ops() -> CrmHandlerOps:
    return CrmHandlerOps(
        resolve_account_by_descriptor=_resolve_account_by_descriptor,
        find_internal_contact_by_name=_find_internal_contact_by_name,
        iter_account_records=_iter_account_records,
        find_named_contact=_find_named_contact,
        resolve_direct_email_target=_resolve_direct_email_target,
        write_outbound_email=_write_outbound_email,
        parse_channel_status_request=_parse_channel_status_request,
        read_named_channel_status_text=_read_named_channel_status_text,
        read_json=_read_json,
        read_text=_read_text,
        answer_and_stop=_answer_and_stop,
    )


def _typed_mutation_ops() -> TypedMutationOps:
    return TypedMutationOps(
        current_repo_date=_current_repo_date,
        read_json=_read_json,
        read_text=_read_text,
        list_names=_list_names,
        run_write_json=_run_write_json,
        resolve_account_by_descriptor=_resolve_account_by_descriptor,
        answer_and_stop=_answer_and_stop,
    )


def _crm_inbox_ops() -> CrmInboxOps:
    return CrmInboxOps(
        list_names=_list_names,
        read_text=_read_text,
        read_json=_read_json,
        search_paths=_search_paths,
        resolve_account_by_descriptor=_resolve_account_by_descriptor,
        select_latest_invoice=_select_latest_invoice,
        write_outbound_email=_write_outbound_email,
        read_named_channel_status_text=_read_named_channel_status_text,
        load_contact_candidates=_load_contact_candidates,
        run_delete=_run_delete,
        run_write_text=_run_write_text,
        answer_and_stop=_answer_and_stop,
    )


def _knowledge_repo_ops() -> KnowledgeRepoOps:
    return KnowledgeRepoOps(
        list_names=_list_names,
        read_text=_read_text,
        run_write_text=_run_write_text,
        run_delete=_run_delete,
        answer_and_stop=_answer_and_stop,
        current_repo_date=_current_repo_date,
    )


@dataclass
class AgentSessionState:
    task_text: str
    messages: list[dict[str, str]] = field(default_factory=list)
    grounded_agent_paths: set[str] = field(default_factory=set)
    attempted_agent_paths: set[str] = field(default_factory=set)
    pending_verification_paths: set[str] = field(default_factory=set)
    root_entries: set[str] = field(default_factory=set)
    repository_profile: str = "generic"
    capabilities: WorkspaceCapabilities = field(default_factory=infer_workspace_capabilities)
    frame: TaskFrame | None = None
    local_fallback_count: int = 0
    last_failed_command: str | None = None
    last_failed_error: str | None = None
    repeated_failure_count: int = 0

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})


def _append_tool_result(
    session: AgentSessionState,
    tool_name: str,
    text: str,
) -> None:
    session.add_message("user", build_tool_result_prompt(tool_name, text))


def _run_grounding_target(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    kind: str,
    path: str,
) -> None:
    grounding_run_grounding_target(
        runtime,
        session,
        kind,
        path,
        read_first_available_fn=_read_first_available,
        auto_command_fn=_auto_command,
    )


def _run_startup_reads(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    paths: list[str],
) -> None:
    grounding_run_startup_reads(runtime, session, paths, _auto_command)


def _read_first_available(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
    label: str = "AUTO",
) -> str | None:
    return grounding_read_first_available(runtime, session, path, _auto_command, label=label)


def _auto_command(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    cmd: ToolRequest,
    label: str = "AUTO",
) -> str | None:
    return grounding_auto_command(
        runtime,
        session,
        cmd,
        label=label,
        append_tool_result=_append_tool_result,
        run_startup_reads=_run_startup_reads,
        cli_green=CLI_GREEN,
        cli_yellow=CLI_YELLOW,
        cli_clr=CLI_CLR,
    )


def _ensure_agent_grounding(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    target_path: str,
) -> None:
    grounding_ensure_agent_grounding(runtime, session, target_path, _auto_command)


def _bootstrap(runtime: PcmRuntimeAdapter, session: AgentSessionState) -> None:
    grounding_bootstrap(
        runtime,
        session,
        run_grounding_target_fn=_run_grounding_target,
        auto_command_fn=_auto_command,
        read_first_available_fn=_read_first_available,
    )


def _frame_task(
    llm: JsonChatClient,
    session: AgentSessionState,
    telemetry: AgentRunTelemetry,
) -> TaskFrame:
    try:
        frame, raw_text, elapsed_ms, usage = llm.complete_json(
            [*session.messages, {"role": "user", "content": build_task_frame_prompt(session.task_text)}],
            TaskFrame,
        )
    except StructuredResponseError as exc:
        telemetry.record_llm_call(exc.elapsed_ms, exc.usage)
        raise
    telemetry.record_llm_call(elapsed_ms, usage)
    print(f"{CLI_BLUE}FRAME{CLI_CLR}: {frame.category} ({elapsed_ms} ms)")
    print(f"  success: {', '.join(frame.success_criteria)}")
    if frame.risks:
        print(f"  risks: {', '.join(frame.risks)}")
    session.frame = frame
    session.add_message("assistant", raw_text.strip() or frame.model_dump_json(indent=2))
    return frame


def _ground_frame(runtime: PcmRuntimeAdapter, session: AgentSessionState, frame: TaskFrame) -> None:
    grounding_ground_frame(
        runtime,
        session,
        frame,
        run_grounding_target_fn=_run_grounding_target,
        ensure_agent_grounding_fn=_ensure_agent_grounding,
    )


def _emit_preflight_completion(payload: ReportTaskCompletion) -> None:
    status = CLI_GREEN if payload.outcome == "OUTCOME_OK" else CLI_YELLOW
    print(f"{status}agent {payload.outcome}{CLI_CLR}. Summary:")
    for item in payload.completed_steps_laconic:
        print(f"- {item}")
    print(f"\n{CLI_BLUE}AGENT SUMMARY: {payload.message}{CLI_CLR}")
    for ref in payload.grounding_refs:
        print(f"- {CLI_BLUE}{ref}{CLI_CLR}")


def _command_signature(cmd: ToolRequest) -> str:
    return f"{cmd.__class__.__name__}:{json.dumps(cmd.model_dump(mode='json'), sort_keys=True)}"


def _reset_repeated_failure_state(session: AgentSessionState) -> None:
    session.last_failed_command = None
    session.last_failed_error = None
    session.repeated_failure_count = 0


def _track_repeated_failure(
    session: AgentSessionState,
    cmd: ToolRequest,
    error_message: str,
) -> ReportTaskCompletion | None:
    signature = _command_signature(cmd)
    normalized_error = error_message.strip().lower()
    if signature == session.last_failed_command and normalized_error == session.last_failed_error:
        session.repeated_failure_count += 1
    else:
        session.last_failed_command = signature
        session.last_failed_error = normalized_error
        session.repeated_failure_count = 1

    if session.repeated_failure_count < 3:
        return None

    refs = command_paths(cmd) or ["/AGENTS.md"]
    return ReportTaskCompletion(
        tool="report_completion",
        completed_steps_laconic=[
            f"Observed repeated failing {cmd.tool} call",
            f"Stopped after {session.repeated_failure_count} identical failures",
        ],
        message=(
            "The current planning loop repeated the same failing tool call and could not recover. "
            "Stopping instead of looping further."
        ),
        grounding_refs=refs,
        outcome="OUTCOME_ERR_INTERNAL",
    )


def _local_fallback_command(session: AgentSessionState) -> ToolRequest | None:
    index = session.local_fallback_count
    sequence = local_fallback_commands(session.repository_profile, session.task_text)
    session.local_fallback_count += 1
    return sequence[min(index, len(sequence) - 1)]


def _extract_tool_body(text: str | None) -> str:
    return extract_tool_body(text)


def _list_names(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
) -> list[str]:
    return tool_list_names(runtime, session, path, auto_command_fn=_auto_command)


def _read_text(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
) -> str | None:
    return tool_read_text(runtime, session, path, read_first_available_fn=_read_first_available)


def _read_json(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
) -> dict | None:
    return tool_read_json(runtime, session, path, read_text_fn=_read_text)


def _search_paths(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    pattern: str,
    root: str,
    limit: int = 20,
) -> list[str]:
    return tool_search_paths(runtime, session, pattern, root, limit, auto_command_fn=_auto_command)


def _run_write_json(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
    payload: dict,
) -> bool:
    return tool_run_write_json(runtime, session, path, payload, auto_command_fn=_auto_command)


def _run_write_text(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
    content: str,
) -> bool:
    return tool_run_write_text(runtime, session, path, content, auto_command_fn=_auto_command)


def _run_delete(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    path: str,
) -> bool:
    return tool_run_delete(runtime, session, path, auto_command_fn=_auto_command)


def _answer_and_stop(
    runtime: PcmRuntimeAdapter,
    payload: ReportTaskCompletion,
) -> None:
    tool_answer_and_stop(runtime, payload, emit_completion_fn=_emit_preflight_completion)


def _load_contact_candidates(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    full_name: str,
) -> list[ContactCandidate]:
    return load_contact_candidates(_crm_ops(), runtime, session, full_name)


def _iter_account_records(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> list[tuple[str, dict]]:
    return iter_account_records(_crm_ops(), runtime, session)


def _account_query_score(account: dict, query_text: str) -> int:
    return account_query_score(account, query_text)


def _resolve_account_by_descriptor(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    descriptor: str,
) -> tuple[str, dict] | None:
    return resolve_account_by_descriptor(_crm_ops(), runtime, session, descriptor)


def _find_internal_contact_by_name(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    full_name: str,
) -> tuple[str, dict] | None:
    return find_internal_contact_by_name(_crm_ops(), runtime, session, full_name)


def _resolve_direct_email_target(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    target: str,
) -> tuple[str, list[str]] | None:
    return resolve_direct_email_target(_crm_ops(), runtime, session, target)


def _select_latest_invoice(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    account_id: str,
) -> tuple[str, dict] | None:
    return select_latest_invoice(_crm_ops(), runtime, session, account_id)


def _write_outbound_email(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    to_email: str,
    subject: str,
    body: str,
    attachments: list[str] | None = None,
) -> str | None:
    return write_outbound_email(_crm_ops(), runtime, session, to_email, subject, body, attachments)


def _parse_channel_status_request(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    task_text: str,
) -> ChannelStatusRequest | None:
    return parse_channel_status_request(_crm_ops(), runtime, session, task_text)


def _read_named_channel_status_text(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    channel_name: str,
) -> tuple[str | None, str | None]:
    return read_named_channel_status_text(_crm_ops(), runtime, session, channel_name)


def _handle_knowledge_repo_inbox_security(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    return handle_knowledge_repo_inbox_security(_knowledge_repo_ops(), runtime, session)


def _current_repo_date(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> date | None:
    return tool_current_repo_date(runtime, session, auto_command_fn=_auto_command)


def _find_named_contact(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    full_name: str,
) -> tuple[str, dict] | None:
    matches: list[tuple[str, dict]] = []
    for name in _list_names(runtime, session, "/contacts"):
        if not name.endswith(".json"):
            continue
        path = f"/contacts/{name}"
        payload = _read_json(runtime, session, path)
        if not payload:
            continue
        if names_match(str(payload.get("full_name", "")), full_name):
            matches.append((path, payload))
    if len(matches) != 1:
        return None
    return matches[0]


def _choose_thread_path(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
    text: str,
) -> str | None:
    return choose_knowledge_thread_path(_knowledge_repo_ops(), runtime, session, text)


def _handle_direct_capture_snippet(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    return handle_direct_capture_snippet(_knowledge_repo_ops(), runtime, session)


def _handle_knowledge_repo_capture(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    return handle_knowledge_repo_capture(_knowledge_repo_ops(), runtime, session)


def _handle_knowledge_repo_cleanup(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    return handle_knowledge_repo_cleanup(_knowledge_repo_ops(), runtime, session)


def _handle_invoice_creation(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    return handle_invoice_creation(_typed_mutation_ops(), runtime, session)


def _handle_followup_reschedule(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    return handle_followup_reschedule(_typed_mutation_ops(), runtime, session)


def _handle_contact_email_lookup(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    return handle_contact_email_lookup(_crm_handler_ops(), runtime, session)


def _handle_direct_outbound_email(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    return handle_direct_outbound_email(_crm_handler_ops(), runtime, session)


def _handle_channel_status_lookup(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    return handle_channel_status_lookup(_crm_handler_ops(), runtime, session)


def _handle_typed_crm_inbox(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    return handle_typed_crm_inbox(_crm_inbox_ops(), runtime, session)


def _handle_purchase_prefix_regression(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    return handle_purchase_prefix_regression(_typed_mutation_ops(), runtime, session)


def _run_fastpath_handlers(
    runtime: PcmRuntimeAdapter,
    session: AgentSessionState,
) -> bool:
    return run_fastpath_handlers(
        (
            lambda: _handle_direct_capture_snippet(runtime, session),
            lambda: _handle_knowledge_repo_capture(runtime, session),
            lambda: _handle_knowledge_repo_cleanup(runtime, session),
            lambda: _handle_invoice_creation(runtime, session),
            lambda: _handle_followup_reschedule(runtime, session),
            lambda: _handle_contact_email_lookup(runtime, session),
            lambda: _handle_direct_outbound_email(runtime, session),
            lambda: _handle_channel_status_lookup(runtime, session),
            lambda: _handle_typed_crm_inbox(runtime, session),
            lambda: _handle_purchase_prefix_regression(runtime, session),
        )
    )


def run_agent(model: str, harness_url: str, task_text: str) -> AgentRunTelemetry:
    started = time.time()
    telemetry = AgentRunTelemetry()
    config = AgentConfig.from_env(model)
    runtime = PcmRuntimeAdapter(harness_url)
    llm = JsonChatClient(config)
    session = AgentSessionState(task_text=task_text)

    try:
        early_preflight = pre_bootstrap_outcome(task_text)
        if early_preflight is not None:
            completion = ReportTaskCompletion(tool="report_completion", **early_preflight.model_dump())
            txt = runtime.execute(completion)
            print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
            _emit_preflight_completion(completion)
            return telemetry
        _bootstrap(runtime, session)
        preflight = preflight_outcome(session.repository_profile, task_text)
        if preflight is not None:
            completion = ReportTaskCompletion(tool="report_completion", **preflight.model_dump())
            txt = runtime.execute(completion)
            print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
            _emit_preflight_completion(completion)
            return telemetry
        if _handle_knowledge_repo_inbox_security(runtime, session):
            return telemetry
        if config.fastpath_mode == "all":
            if _run_fastpath_handlers(runtime, session):
                return telemetry
        shortcut_frame = derive_high_confidence_frame(task_text, session.repository_profile, session.capabilities)
        if shortcut_frame is not None:
            frame = shortcut_frame
            session.frame = frame
            print(f"{CLI_BLUE}FRAME SHORTCUT{CLI_CLR}: {frame.category}")
            session.add_message("assistant", frame.model_dump_json(indent=2))
        else:
            try:
                frame = _frame_task(llm, session, telemetry)
            except Exception as exc:
                if config.use_gbnf_grammar:
                    frame = derive_fallback_frame(session.task_text, session.repository_profile, session.capabilities)
                    session.frame = frame
                    print(f"{CLI_YELLOW}FRAME FALLBACK{CLI_CLR}: {exc}")
                    session.add_message("assistant", frame.model_dump_json(indent=2))
                else:
                    raise
        _ground_frame(runtime, session, frame)
        if config.fastpath_mode in {"framed", "all"}:
            if _run_fastpath_handlers(runtime, session):
                return telemetry
        session.add_message("user", build_execution_prompt(task_text, frame))

        for index in range(config.max_steps):
            step_name = f"step_{index + 1}"
            print(f"Next {step_name}... ", end="")

            try:
                job, raw_text, elapsed_ms, usage = llm.complete_json(session.messages, NextStep)
            except StructuredResponseError as exc:
                telemetry.record_llm_call(exc.elapsed_ms, exc.usage)
                raise
            telemetry.record_llm_call(elapsed_ms, usage)
            print(job.plan_remaining_steps_brief[0], f"({elapsed_ms} ms)\n  {job.function}")

            session.add_message(
                "assistant",
                raw_text.strip() or job.model_dump_json(indent=2),
            )

            if config.use_gbnf_grammar and isinstance(job.function, Req_Context):
                fallback_cmd = _local_fallback_command(session)
                if fallback_cmd is not None:
                    print(f"{CLI_YELLOW}LOCAL FALLBACK{CLI_CLR}: {fallback_cmd}")
                    job = job.model_copy(update={"function": fallback_cmd})

            for path in command_paths(job.function):
                if is_mutating_command(job.function):
                    _ensure_agent_grounding(runtime, session, path)

            precondition_message = prepare_command(
                session.task_text,
                session.pending_verification_paths,
                job.function,
            )
            if precondition_message is not None:
                txt = precondition_message
                if isinstance(job.function, ReportTaskCompletion):
                    label = "VERIFY" if session.pending_verification_paths else "POLICY"
                    print(f"{CLI_YELLOW}{label}{CLI_CLR}: {precondition_message}")
                else:
                    print(f"{CLI_YELLOW}POLICY{CLI_CLR}: {precondition_message}")
                _reset_repeated_failure_state(session)
            else:
                try:
                    txt = runtime.execute(job.function)
                    print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
                    session.pending_verification_paths = next_pending_verification_paths(
                        session.pending_verification_paths,
                        job.function,
                    )
                    _reset_repeated_failure_state(session)
                except ConnectError as exc:
                    txt = str(exc.message)
                    print(f"{CLI_RED}ERR {exc.code}: {exc.message}{CLI_CLR}")
                    repeated_failure = _track_repeated_failure(session, job.function, exc.message)
                    if repeated_failure is not None:
                        runtime.execute(repeated_failure)
                        _emit_preflight_completion(repeated_failure)
                        break

            if isinstance(job.function, ReportTaskCompletion) and precondition_message is None:
                _emit_preflight_completion(job.function)
                break

            _append_tool_result(session, job.function.__class__.__name__, txt)
        return telemetry
    finally:
        telemetry.wall_time_ms = int((time.time() - started) * 1000)
