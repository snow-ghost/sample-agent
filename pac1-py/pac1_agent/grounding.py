from __future__ import annotations

from typing import Any, Callable

from connectrpc.errors import ConnectError

from .models import Req_Context, Req_List, Req_Read, Req_Tree, TaskFrame, ToolRequest
from .pathing import AGENT_FILE_NAMES, candidate_read_paths, is_agent_instruction_path, normalize_repo_path
from .policy import build_system_prompt, build_workspace_context_prompt
from .workspace import (
    candidate_agent_paths,
    derive_workspace_facts,
    extract_startup_reads,
    parse_root_entries_from_listing,
    parse_root_entries_from_tree,
    profile_grounding_targets,
    relevant_roots,
)


def record_agent_file(session: Any, agent_path: str, content: str) -> list[str]:
    normalized = normalize_repo_path(agent_path)
    session.grounded_agent_paths.add(normalized)
    return extract_startup_reads(content)


def update_workspace_facts_from_root_entries(session: Any, root_entries: set[str]) -> None:
    if not root_entries:
        return
    session.root_entries = root_entries
    session.repository_profile, session.capabilities = derive_workspace_facts(session.root_entries)


def auto_command(
    runtime: Any,
    session: Any,
    cmd: ToolRequest,
    *,
    label: str = "AUTO",
    append_tool_result: Callable[[Any, str, str], None],
    run_startup_reads: Callable[[Any, Any, list[str]], None],
    cli_green: str,
    cli_yellow: str,
    cli_clr: str,
) -> str | None:
    try:
        text = runtime.execute(cmd)
        print(f"{cli_green}{label}{cli_clr}: {text}")
        append_tool_result(session, cmd.__class__.__name__, text)
        if isinstance(cmd, Req_Read) and is_agent_instruction_path(cmd.path):
            startup_reads = record_agent_file(
                session,
                cmd.path,
                text.split("\n", 1)[1] if "\n" in text else "",
            )
            run_startup_reads(runtime, session, startup_reads)
        return text
    except ConnectError as exc:
        print(f"{cli_yellow}{label} ERR {exc.code}: {exc.message}{cli_clr}")
        return None


def read_first_available(
    runtime: Any,
    session: Any,
    path: str,
    auto_command_fn: Callable[..., str | None],
    *,
    label: str = "AUTO",
) -> str | None:
    for candidate in candidate_read_paths(path):
        text = auto_command_fn(runtime, session, Req_Read(tool="read", path=candidate), label=label)
        if text is not None:
            return text
    return None


def run_startup_reads(
    runtime: Any,
    session: Any,
    paths: list[str],
    auto_command_fn: Callable[..., str | None],
) -> None:
    for extra_path in paths:
        for candidate in candidate_read_paths(extra_path):
            normalized = normalize_repo_path(candidate)
            if normalized in session.attempted_agent_paths:
                continue
            session.attempted_agent_paths.add(normalized)
            text = auto_command_fn(runtime, session, Req_Read(tool="read", path=candidate))
            if text is not None:
                break


def run_grounding_target(
    runtime: Any,
    session: Any,
    kind: str,
    path: str,
    *,
    read_first_available_fn: Callable[..., str | None],
    auto_command_fn: Callable[..., str | None],
) -> None:
    if kind == "read":
        read_first_available_fn(runtime, session, path)
        return
    if kind == "list":
        listing = auto_command_fn(runtime, session, Req_List(tool="list", path=path))
        if path == "/" and listing:
            update_workspace_facts_from_root_entries(session, parse_root_entries_from_listing(listing))
        return
    raise ValueError(f"Unknown grounding target kind: {kind}")


def ensure_agent_grounding(
    runtime: Any,
    session: Any,
    target_path: str,
    auto_command_fn: Callable[..., str | None],
) -> None:
    for agent_path in candidate_agent_paths(target_path):
        if agent_path in session.attempted_agent_paths:
            continue
        session.attempted_agent_paths.add(agent_path)
        auto_command_fn(runtime, session, Req_Read(tool="read", path=agent_path))


def bootstrap(
    runtime: Any,
    session: Any,
    *,
    run_grounding_target_fn: Callable[[Any, Any, str, str], None],
    auto_command_fn: Callable[..., str | None],
    read_first_available_fn: Callable[..., str | None],
) -> None:
    session.add_message("system", build_system_prompt())

    run_grounding_target_fn(runtime, session, "list", "/")
    tree_text = auto_command_fn(runtime, session, Req_Tree(tool="tree", root="/", level=2))
    if not session.root_entries and tree_text:
        update_workspace_facts_from_root_entries(session, parse_root_entries_from_tree(tree_text))
    for agent_name in AGENT_FILE_NAMES:
        session.attempted_agent_paths.add(normalize_repo_path(f"/{agent_name}"))
    read_first_available_fn(runtime, session, "/AGENTS.md")
    auto_command_fn(runtime, session, Req_Context(tool="context"))
    session.add_message(
        "user",
        build_workspace_context_prompt(session.repository_profile, session.capabilities),
    )


def ground_frame(
    runtime: Any,
    session: Any,
    frame: TaskFrame,
    *,
    run_grounding_target_fn: Callable[[Any, Any, str, str], None],
    ensure_agent_grounding_fn: Callable[[Any, Any, str], None],
) -> None:
    for target in profile_grounding_targets(session.repository_profile, frame, session.task_text):
        run_grounding_target_fn(runtime, session, target.kind, target.path)
    for root in relevant_roots(frame):
        ensure_agent_grounding_fn(runtime, session, root)
