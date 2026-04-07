from __future__ import annotations

import json
import os
import shlex
import time
from typing import Any

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from bitgn.vm.pcm_pb2 import (
    AnswerRequest,
    ContextRequest,
    DeleteRequest,
    FindRequest,
    ListRequest,
    MkDirRequest,
    MoveRequest,
    Outcome,
    ReadRequest,
    SearchRequest,
    TreeRequest,
    WriteRequest,
)
from google.protobuf.json_format import MessageToDict

from .models import (
    ReportTaskCompletion,
    Req_Context,
    Req_Delete,
    Req_Find,
    Req_List,
    Req_MkDir,
    Req_Move,
    Req_Read,
    Req_Search,
    Req_Tree,
    Req_Write,
    ToolRequest,
)


OUTCOME_BY_NAME = {
    "OUTCOME_OK": Outcome.OUTCOME_OK,
    "OUTCOME_DENIED_SECURITY": Outcome.OUTCOME_DENIED_SECURITY,
    "OUTCOME_NONE_CLARIFICATION": Outcome.OUTCOME_NONE_CLARIFICATION,
    "OUTCOME_NONE_UNSUPPORTED": Outcome.OUTCOME_NONE_UNSUPPORTED,
    "OUTCOME_ERR_INTERNAL": Outcome.OUTCOME_ERR_INTERNAL,
}

TRANSIENT_CONNECT_CODES = {
    Code.UNAVAILABLE,
    Code.INTERNAL,
    Code.UNKNOWN,
    Code.DEADLINE_EXCEEDED,
}


def _format_tree_entry(entry: Any, prefix: str = "", is_last: bool = True) -> list[str]:
    branch = "└── " if is_last else "├── "
    lines = [f"{prefix}{branch}{entry.name}"]
    child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
    children = list(entry.children)
    for index, child in enumerate(children):
        lines.extend(
            _format_tree_entry(
                child,
                prefix=child_prefix,
                is_last=index == len(children) - 1,
            )
        )
    return lines


def _render_command(command: str, body: str) -> str:
    return f"{command}\n{body}"


def _format_tree_response(cmd: Req_Tree, result: Any) -> str:
    root = result.root
    if not root.name:
        body = "."
    else:
        lines = [root.name]
        children = list(root.children)
        for index, child in enumerate(children):
            lines.extend(_format_tree_entry(child, is_last=index == len(children) - 1))
        body = "\n".join(lines)

    root_arg = cmd.root or "/"
    level_arg = f" -L {cmd.level}" if cmd.level > 0 else ""
    return _render_command(f"tree{level_arg} {root_arg}", body)


def _format_list_response(cmd: Req_List, result: Any) -> str:
    if not result.entries:
        body = "."
    else:
        body = "\n".join(
            f"{entry.name}/" if entry.is_dir else entry.name for entry in result.entries
        )
    return _render_command(f"ls {cmd.path}", body)


def _format_read_response(cmd: Req_Read, result: Any) -> str:
    if cmd.start_line > 0 or cmd.end_line > 0:
        start = cmd.start_line if cmd.start_line > 0 else 1
        end = cmd.end_line if cmd.end_line > 0 else "$"
        command = f"sed -n '{start},{end}p' {cmd.path}"
    elif cmd.number:
        command = f"cat -n {cmd.path}"
    else:
        command = f"cat {cmd.path}"
    return _render_command(command, result.content)


def _format_search_response(cmd: Req_Search, result: Any) -> str:
    root = shlex.quote(cmd.root or "/")
    pattern = shlex.quote(cmd.pattern)
    body = "\n".join(
        f"{match.path}:{match.line}:{match.line_text}" for match in result.matches
    )
    return _render_command(f"rg -n --no-heading -e {pattern} {root}", body)


def format_result(cmd: ToolRequest, result: Any) -> str:
    if result is None:
        return "{}"
    if isinstance(cmd, Req_Tree):
        return _format_tree_response(cmd, result)
    if isinstance(cmd, Req_List):
        return _format_list_response(cmd, result)
    if isinstance(cmd, Req_Read):
        return _format_read_response(cmd, result)
    if isinstance(cmd, Req_Search):
        return _format_search_response(cmd, result)
    return json.dumps(MessageToDict(result), indent=2)


class PcmRuntimeAdapter:
    def __init__(self, harness_url: str) -> None:
        self.client = PcmRuntimeClientSync(harness_url)
        self.retry_attempts = int(os.getenv("PCM_RETRY_ATTEMPTS", "4"))
        self.retry_delay_seconds = float(os.getenv("PCM_RETRY_DELAY_SECONDS", "0.25"))

    def _is_transient_error(self, exc: Exception) -> bool:
        if isinstance(exc, ConnectError):
            return exc.code in TRANSIENT_CONNECT_CODES
        message = str(exc).lower()
        return any(marker in message for marker in ("bad gateway", "502", "gateway timeout", "temporarily unavailable"))

    def _dispatch_with_retry(self, cmd: ToolRequest) -> Any:
        attempts = self.retry_attempts + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return self.dispatch(cmd)
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts - 1 or not self._is_transient_error(exc):
                    raise
                time.sleep(self.retry_delay_seconds * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def dispatch(self, cmd: ToolRequest) -> Any:
        if isinstance(cmd, Req_Context):
            return self.client.context(ContextRequest())
        if isinstance(cmd, Req_Tree):
            return self.client.tree(TreeRequest(root=cmd.root, level=cmd.level))
        if isinstance(cmd, Req_Find):
            return self.client.find(
                FindRequest(
                    root=cmd.root,
                    name=cmd.name,
                    type={"all": 0, "files": 1, "dirs": 2}[cmd.kind],
                    limit=cmd.limit,
                )
            )
        if isinstance(cmd, Req_Search):
            return self.client.search(
                SearchRequest(root=cmd.root, pattern=cmd.pattern, limit=cmd.limit)
            )
        if isinstance(cmd, Req_List):
            return self.client.list(ListRequest(name=cmd.path))
        if isinstance(cmd, Req_Read):
            return self.client.read(
                ReadRequest(
                    path=cmd.path,
                    number=cmd.number,
                    start_line=cmd.start_line,
                    end_line=cmd.end_line,
                )
            )
        if isinstance(cmd, Req_Write):
            return self.client.write(
                WriteRequest(
                    path=cmd.path,
                    content=cmd.content,
                    start_line=cmd.start_line,
                    end_line=cmd.end_line,
                )
            )
        if isinstance(cmd, Req_Delete):
            return self.client.delete(DeleteRequest(path=cmd.path))
        if isinstance(cmd, Req_MkDir):
            return self.client.mk_dir(MkDirRequest(path=cmd.path))
        if isinstance(cmd, Req_Move):
            return self.client.move(MoveRequest(from_name=cmd.from_name, to_name=cmd.to_name))
        if isinstance(cmd, ReportTaskCompletion):
            return self.client.answer(
                AnswerRequest(
                    message=cmd.message,
                    outcome=OUTCOME_BY_NAME[cmd.outcome],
                    refs=cmd.grounding_refs,
                )
            )
        raise ValueError(f"Unknown command: {cmd}")

    def execute(self, cmd: ToolRequest) -> str:
        return format_result(cmd, self._dispatch_with_retry(cmd))
