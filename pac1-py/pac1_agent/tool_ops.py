from __future__ import annotations

from datetime import date
import json
from typing import Any, Callable

from .models import Req_Context, Req_Delete, Req_List, Req_Search, Req_Write
from .pathing import normalize_repo_path


AutoCommandFn = Callable[[Any, Any, Any], str | None]
ReadFirstAvailableFn = Callable[[Any, Any, str], str | None]


def extract_tool_body(text: str | None) -> str:
    if not text:
        return ""
    return text.split("\n", 1)[1] if "\n" in text else ""


def list_names(
    runtime: Any,
    session: Any,
    path: str,
    auto_command_fn: AutoCommandFn,
) -> list[str]:
    text = auto_command_fn(runtime, session, Req_List(tool="list", path=path))
    body = extract_tool_body(text)
    return [line.rstrip("/").strip() for line in body.splitlines() if line.strip() and line.strip() != "."]


def read_text(
    runtime: Any,
    session: Any,
    path: str,
    read_first_available_fn: ReadFirstAvailableFn,
) -> str | None:
    return extract_tool_body(read_first_available_fn(runtime, session, path)) or None


def read_json(
    runtime: Any,
    session: Any,
    path: str,
    read_text_fn: Callable[[Any, Any, str], str | None],
) -> dict | None:
    text = read_text_fn(runtime, session, path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def search_paths(
    runtime: Any,
    session: Any,
    pattern: str,
    root: str,
    limit: int,
    auto_command_fn: AutoCommandFn,
) -> list[str]:
    text = auto_command_fn(
        runtime,
        session,
        Req_Search(tool="search", pattern=pattern, root=root, limit=limit),
    )
    body = extract_tool_body(text)
    paths: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        path = normalize_repo_path(line.split(":", 1)[0])
        if path not in paths:
            paths.append(path)
    return paths


def run_write_json(
    runtime: Any,
    session: Any,
    path: str,
    payload: dict,
    auto_command_fn: AutoCommandFn,
) -> bool:
    text = auto_command_fn(
        runtime,
        session,
        Req_Write(
            tool="write",
            path=path,
            content=json.dumps(payload, indent=2),
        ),
    )
    return text is not None


def run_write_text(
    runtime: Any,
    session: Any,
    path: str,
    content: str,
    auto_command_fn: AutoCommandFn,
) -> bool:
    text = auto_command_fn(
        runtime,
        session,
        Req_Write(tool="write", path=path, content=content),
    )
    return text is not None


def run_delete(
    runtime: Any,
    session: Any,
    path: str,
    auto_command_fn: AutoCommandFn,
) -> bool:
    text = auto_command_fn(runtime, session, Req_Delete(tool="delete", path=path))
    return text is not None


def answer_and_stop(
    runtime: Any,
    payload: Any,
    emit_completion_fn: Callable[[Any], None],
) -> None:
    txt = runtime.execute(payload)
    print(f"\x1B[32mOUT\x1B[0m: {txt}")
    emit_completion_fn(payload)


def current_repo_date(
    runtime: Any,
    session: Any,
    auto_command_fn: AutoCommandFn,
) -> date | None:
    text = auto_command_fn(runtime, session, Req_Context(tool="context"), label="AUTO")
    body = (text or "").strip()
    candidate = body if body.startswith("{") and body.endswith("}") else extract_tool_body(text)
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    iso_value = str(payload.get("time") or "").strip()
    if not iso_value:
        return None
    return date.fromisoformat(iso_value[:10])
