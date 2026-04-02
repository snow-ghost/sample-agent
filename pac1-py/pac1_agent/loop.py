from __future__ import annotations

from dataclasses import dataclass, field

from connectrpc.errors import ConnectError

from .config import AgentConfig
from .llm import JsonChatClient
from .models import (
    NextStep,
    ReportTaskCompletion,
    Req_Context,
    Req_Read,
    Req_Tree,
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
    frame = _frame_task(llm, session)
    _ground_frame(runtime, session, frame)
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
