from __future__ import annotations

import json
import time
from typing import Any, Literal, TypeVar
from urllib import request as urllib_request

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import AgentConfig
from .models import (
    NextStep,
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
    TaskFrame,
)
from .telemetry import TokenUsage

ModelT = TypeVar("ModelT", bound=BaseModel)


GBNF_TASK_FRAME = r"""
root ::= "{" ws "\"current_state\"" ws ":" ws string ws "," ws "\"category\"" ws ":" ws category ws "," ws "\"success_criteria\"" ws ":" ws "[" ws string ws "]" ws "," ws "\"relevant_roots\"" ws ":" ws "[" ws string ws "]" ws "," ws "\"risks\"" ws ":" ws "[" ws string ws "]" ws "}" ws
category ::= "\"cleanup_or_edit\"" | "\"lookup\"" | "\"typed_workflow\"" | "\"security_sensitive\"" | "\"clarification_or_reference\"" | "\"mixed\""
string ::= "\"" chars "\""
chars ::= [-a-zA-Z0-9_ .,:/@()+*#%&!?=]*
ws ::= [ \n]*
""".strip()


GBNF_NEXT_STEP = r"""
root ::= "{" ws "\"current_state\"" ws ":" ws string ws "," ws "\"plan_step\"" ws ":" ws string ws "," ws "\"task_completed\"" ws ":" ws boolean ws "," ws "\"tool\"" ws ":" ws tool ws "," ws "\"arg1\"" ws ":" ws string ws "," ws "\"arg2\"" ws ":" ws string ws "," ws "\"arg3\"" ws ":" ws string ws "," ws "\"arg4\"" ws ":" ws string ws "," ws "\"num1\"" ws ":" ws integer ws "," ws "\"num2\"" ws ":" ws integer ws "," ws "\"flag1\"" ws ":" ws boolean ws "," ws "\"flag2\"" ws ":" ws boolean ws "," ws "\"outcome\"" ws ":" ws outcome ws "}" ws
tool ::= "\"report_completion\"" | "\"context\"" | "\"tree\"" | "\"find\"" | "\"search\"" | "\"list\"" | "\"read\"" | "\"write\"" | "\"delete\"" | "\"mkdir\"" | "\"move\""
outcome ::= "\"OUTCOME_OK\"" | "\"OUTCOME_DENIED_SECURITY\"" | "\"OUTCOME_NONE_CLARIFICATION\"" | "\"OUTCOME_NONE_UNSUPPORTED\"" | "\"OUTCOME_ERR_INTERNAL\""
boolean ::= "true" | "false"
integer ::= [0-9]+
string ::= "\"" chars "\""
chars ::= [-a-zA-Z0-9_ .,:/@()+*#%&!?=]*
ws ::= [ \n]*
""".strip()


class LocalFlatNextStep(BaseModel):
    current_state: str = "continue"
    plan_step: str = "continue"
    task_completed: bool = False
    tool: Literal[
        "report_completion",
        "context",
        "tree",
        "find",
        "search",
        "list",
        "read",
        "write",
        "delete",
        "mkdir",
        "move",
    ] = "context"
    arg1: str = ""
    arg2: str = ""
    arg3: str = ""
    arg4: str = ""
    num1: int = 0
    num2: int = 0
    flag1: bool = False
    flag2: bool = False
    outcome: Literal[
        "OUTCOME_OK",
        "OUTCOME_DENIED_SECURITY",
        "OUTCOME_NONE_CLARIFICATION",
        "OUTCOME_NONE_UNSUPPORTED",
        "OUTCOME_ERR_INTERNAL",
    ] = "OUTCOME_OK"


class StructuredResponseError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        raw_text: str,
        elapsed_ms: int,
        usage: TokenUsage,
    ) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.elapsed_ms = elapsed_ms
        self.usage = usage


def _extract_json_payload(text: str) -> Any:
    decoder = json.JSONDecoder()
    candidates = [text.strip()]

    if "```" in text:
        parts = text.split("```")
        for index in range(1, len(parts), 2):
            block = parts[index]
            if block.startswith("json"):
                block = block[4:]
            candidates.append(block.strip())

    for candidate in candidates:
        if not candidate:
            continue
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(candidate[index:])
                return payload
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Model did not return a valid JSON object: {text}")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                else:
                    parts.append(json.dumps(part, ensure_ascii=False))
            else:
                text_value = getattr(part, "text", None)
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "\n".join(part for part in parts if part)
    return str(content or "")


class JsonChatClient:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            timeout=config.request_timeout_seconds,
        )

    def complete_json(
        self,
        messages: list[dict[str, str]],
        response_model: type[ModelT],
    ) -> tuple[ModelT, str, int, TokenUsage]:
        request_model = _request_model_for_response(self.config, response_model)
        attempt_messages = _augment_messages_for_request_model(messages, request_model)
        last_error: Exception | None = None
        last_raw_text = ""
        total_elapsed_ms = 0
        total_usage = TokenUsage()

        for attempt in range(self.config.json_repair_retries + 1):
            started = time.time()
            response = self._create_completion(attempt_messages, request_model)
            elapsed_ms = int((time.time() - started) * 1000)
            raw_text = _extract_response_text(response)
            usage = TokenUsage.from_response_usage(_extract_response_usage(response))
            total_elapsed_ms += elapsed_ms
            total_usage = _merge_usage(total_usage, usage)
            last_raw_text = raw_text

            try:
                payload = _extract_json_payload(raw_text)
                if request_model is LocalFlatNextStep:
                    payload = _normalize_local_next_step_payload(payload)
                parsed = request_model.model_validate(payload)
                typed = _coerce_response_model(parsed, response_model)
                return typed, raw_text, total_elapsed_ms, total_usage
            except (ValueError, ValidationError) as exc:
                last_error = exc
                if attempt >= self.config.json_repair_retries:
                    break
                attempt_messages = [
                    *attempt_messages,
                    {"role": "assistant", "content": raw_text},
                    {
                        "role": "user",
                        "content": (
                            _local_retry_prompt(exc) if request_model is LocalFlatNextStep else (
                                "Your previous reply was invalid for the required JSON schema.\n"
                                f"Validation error: {exc}\n"
                                "Return only a corrected JSON object."
                            )
                        ),
                    },
                ]

        raise StructuredResponseError(
            f"Unable to parse structured model response: {last_error}",
            raw_text=last_raw_text,
            elapsed_ms=total_elapsed_ms,
            usage=total_usage,
        )

    def _create_completion(
        self,
        messages: list[dict[str, str]],
        response_model: type[ModelT],
    ) -> Any:
        if self.config.use_gbnf_grammar:
            grammar = _grammar_for_model(response_model)
            if grammar is not None:
                return self._create_completion_with_raw_gbnf(
                    messages,
                    grammar,
                )
        return self.client.chat.completions.create(
            **self._request_kwargs(messages, response_model)
        )

    def _request_kwargs(
        self,
        messages: list[dict[str, str]],
        response_model: type[ModelT],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": 0,
        }
        if self.config.use_gbnf_grammar:
            grammar = _grammar_for_model(response_model)
            if grammar is not None:
                kwargs["extra_body"] = {"grammar": grammar}
        return kwargs

    def _create_completion_with_raw_gbnf(
        self,
        messages: list[dict[str, str]],
        grammar: str,
    ) -> dict[str, Any]:
        base_url = (self.config.openai_base_url or "").rstrip("/")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": 0,
            "grammar": grammar,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.openai_api_key:
            headers["Authorization"] = f"Bearer {self.config.openai_api_key}"
        req = urllib_request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib_request.urlopen(
            req,
            timeout=self.config.request_timeout_seconds,
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))


def _grammar_for_model(response_model: type[BaseModel]) -> str | None:
    if response_model is TaskFrame:
        return GBNF_TASK_FRAME
    if response_model in {NextStep, LocalFlatNextStep}:
        return GBNF_NEXT_STEP
    return None


def _request_model_for_response(
    config: AgentConfig,
    response_model: type[ModelT],
) -> type[BaseModel]:
    if config.use_gbnf_grammar and response_model is NextStep:
        return LocalFlatNextStep
    return response_model


def _augment_messages_for_request_model(
    messages: list[dict[str, str]],
    request_model: type[BaseModel],
) -> list[dict[str, str]]:
    if request_model is not LocalFlatNextStep:
        return list(messages)
    return [
        *messages,
        {
            "role": "user",
            "content": (
                "For this response, use the compact flat JSON schema.\n"
                "Pick one grounded next action.\n"
                "Prefer list, read, find, or search before any mutation.\n"
                "Use context only when current repo time/date is needed.\n"
                "If an exact target cannot be resolved from repository data, use report_completion with a clarification outcome.\n"
                "Do not use report_completion with OUTCOME_OK unless observed repo state proves the task is done.\n"
                "For OUTCOME_OK, arg1 must be a specific completed step, arg2 a concrete message, and arg3 a concrete grounding ref.\n"
                "If the request still looks truncated or underspecified, use report_completion with OUTCOME_NONE_CLARIFICATION.\n"
                "Return exactly one object with these fields:\n"
                '- current_state: short string\n'
                '- plan_step: short string, not an array\n'
                '- task_completed: boolean\n'
                '- tool: one of report_completion, context, tree, find, search, list, read, write, delete, mkdir, move\n'
                '- arg1, arg2, arg3, arg4: strings for tool arguments\n'
                '- num1, num2: integers for numeric arguments\n'
                '- flag1, flag2: booleans for boolean arguments\n'
                '- outcome: terminal outcome, use OUTCOME_OK unless reporting completion\n'
                "Argument mapping:\n"
                "- tree: arg1=root, num1=level\n"
                "- find: arg1=name, arg2=root, arg3=kind, num1=limit\n"
                "- search: arg1=pattern, arg2=root, num1=limit\n"
                "- list/delete/mkdir: arg1=path\n"
                "- read: arg1=path, flag1=number, num1=start_line, num2=end_line\n"
                "- write: arg1=path, arg2=content, num1=start_line, num2=end_line\n"
                "- move: arg1=from_name, arg2=to_name\n"
                "- report_completion: arg1=completed_step, arg2=message, arg3=grounding_ref, outcome=terminal outcome\n"
                "Use empty strings, false, or 0 for unused argument fields. Do not return schema notes."
            ),
        },
    ]


def _coerce_response_model(
    parsed: BaseModel,
    response_model: type[ModelT],
) -> ModelT:
    if response_model is NextStep and isinstance(parsed, LocalFlatNextStep):
        return _coerce_local_next_step(parsed)  # type: ignore[return-value]
    return parsed  # type: ignore[return-value]


def _coerce_local_next_step(step: LocalFlatNextStep) -> NextStep:
    plan_step = step.plan_step or "continue"
    if step.tool == "report_completion":
        function = ReportTaskCompletion(
            tool="report_completion",
            completed_steps_laconic=[step.arg1 or "Completed the requested work"],
            message=step.arg2 or "Task completed.",
            grounding_refs=[step.arg3] if step.arg3 else [],
            outcome=step.outcome,
        )
    elif step.tool == "context":
        function = Req_Context(tool="context")
    elif step.tool == "tree":
        function = Req_Tree(
            tool="tree",
            level=max(0, step.num1 or 2),
            root=step.arg1 or "/",
        )
    elif step.tool == "find":
        function = Req_Find(
            tool="find",
            name=step.arg1 or "",
            root=step.arg2 or "/",
            kind=_normalize_find_kind(step.arg3),
            limit=max(1, min(20, step.num1 or 10)),
        )
    elif step.tool == "search":
        function = Req_Search(
            tool="search",
            pattern=step.arg1 or "",
            limit=max(1, min(20, step.num1 or 10)),
            root=step.arg2 or "/",
        )
    elif step.tool == "list":
        function = Req_List(tool="list", path=step.arg1 or "/")
    elif step.tool == "read":
        function = Req_Read(
            tool="read",
            path=step.arg1 or "/",
            number=step.flag1,
            start_line=max(0, step.num1),
            end_line=max(0, step.num2),
        )
    elif step.tool == "write":
        function = Req_Write(
            tool="write",
            path=step.arg1 or "/",
            content=step.arg2,
            start_line=max(0, step.num1),
            end_line=max(0, step.num2),
        )
    elif step.tool == "delete":
        function = Req_Delete(tool="delete", path=step.arg1 or "/")
    elif step.tool == "mkdir":
        function = Req_MkDir(tool="mkdir", path=step.arg1 or "/")
    else:
        function = Req_Move(
            tool="move",
            from_name=step.arg1,
            to_name=step.arg2,
        )

    return NextStep(
        current_state=step.current_state,
        plan_remaining_steps_brief=[plan_step],
        task_completed=step.task_completed,
        function=function,
    )


def _extract_response_usage(response: Any) -> Any:
    if isinstance(response, dict):
        return response.get("usage")
    return getattr(response, "usage", None)


def _extract_response_text(response: Any) -> str:
    if isinstance(response, dict):
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Malformed raw completion response: {response}") from exc
    return _message_text(response.choices[0].message)


def _merge_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


def _normalize_find_kind(value: str) -> Literal["all", "files", "dirs"]:
    lowered = (value or "").strip().lower()
    if lowered in {"files", "file"}:
        return "files"
    if lowered in {"dirs", "dir", "directories", "directory"}:
        return "dirs"
    return "all"


def _local_retry_prompt(exc: Exception) -> str:
    return (
        "Your previous reply was invalid for the compact step schema.\n"
        f"Validation error: {exc}\n"
        "Return only one JSON object with keys: "
        '"current_state","plan_step","task_completed","tool","arg1","arg2","arg3","arg4","num1","num2","flag1","flag2","outcome".\n'
        "Do not return explanations, schema descriptions, or arrays for plan_step.\n"
        "Prefer list/read/find/search; use context only for current repo time/date.\n"
        "Do not emit generic OUTCOME_OK completion text. OUTCOME_OK requires a specific completed step and concrete grounding ref.\n"
        "If the target is still ambiguous or the request is truncated, use report_completion with OUTCOME_NONE_CLARIFICATION."
    )


def _normalize_local_next_step_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object for LocalFlatNextStep, got: {payload!r}")

    normalized = dict(payload)

    if isinstance(normalized.get("plan_step"), list):
        steps = [str(item).strip() for item in normalized["plan_step"] if str(item).strip()]
        normalized["plan_step"] = steps[0] if steps else "continue"

    tool_aliases = {
        "answer": "report_completion",
        "complete": "report_completion",
        "completion": "report_completion",
        "done": "report_completion",
        "local": "list",
    }
    if "tool" in normalized and isinstance(normalized["tool"], str):
        normalized["tool"] = tool_aliases.get(normalized["tool"].strip().lower(), normalized["tool"])

    if _looks_like_schema_echo(normalized):
        return {
            "current_state": "recover from invalid structured response",
            "plan_step": "list root",
            "task_completed": False,
            "tool": "list",
            "arg1": "/",
            "arg2": "",
            "arg3": "",
            "arg4": "",
            "num1": 0,
            "num2": 0,
            "flag1": False,
            "flag2": False,
            "outcome": "OUTCOME_OK",
        }

    # Backward-compatible normalization from the previous flat schema.
    if any(key in normalized for key in ("path", "root", "pattern", "name", "kind", "limit", "level", "content")):
        tool = str(normalized.get("tool") or "context")
        arg1 = ""
        arg2 = ""
        arg3 = ""
        num1 = 0
        num2 = 0
        flag1 = False
        if tool == "tree":
            arg1 = str(normalized.get("root") or "/")
            num1 = int(normalized.get("level") or 0)
        elif tool == "find":
            arg1 = str(normalized.get("name") or "")
            arg2 = str(normalized.get("root") or "/")
            arg3 = str(normalized.get("kind") or "all")
            num1 = int(normalized.get("limit") or 10)
        elif tool == "search":
            arg1 = str(normalized.get("pattern") or "")
            arg2 = str(normalized.get("root") or "/")
            num1 = int(normalized.get("limit") or 10)
        elif tool in {"list", "delete", "mkdir"}:
            arg1 = str(normalized.get("path") or "/")
        elif tool == "read":
            arg1 = str(normalized.get("path") or "/")
            flag1 = bool(normalized.get("number") or False)
            num1 = int(normalized.get("start_line") or 0)
            num2 = int(normalized.get("end_line") or 0)
        elif tool == "write":
            arg1 = str(normalized.get("path") or "/")
            arg2 = str(normalized.get("content") or "")
            num1 = int(normalized.get("start_line") or 0)
            num2 = int(normalized.get("end_line") or 0)
        elif tool == "move":
            arg1 = str(normalized.get("from_name") or "")
            arg2 = str(normalized.get("to_name") or "")
        elif tool == "report_completion":
            completed_steps = normalized.get("completed_steps_laconic") or []
            grounding_refs = normalized.get("grounding_refs") or []
            arg1 = str(completed_steps[0]) if completed_steps else ""
            arg2 = str(normalized.get("message") or "")
            arg3 = str(grounding_refs[0]) if grounding_refs else ""

        normalized = {
            "current_state": str(normalized.get("current_state") or "continue"),
            "plan_step": str(normalized.get("plan_step") or "continue"),
            "task_completed": bool(normalized.get("task_completed") or False),
            "tool": tool,
            "arg1": arg1,
            "arg2": arg2,
            "arg3": arg3,
            "arg4": "",
            "num1": num1,
            "num2": num2,
            "flag1": flag1,
            "flag2": False,
            "outcome": str(normalized.get("outcome") or "OUTCOME_OK"),
        }

    normalized.setdefault("current_state", "continue")
    normalized.setdefault("plan_step", "continue")
    normalized.setdefault("task_completed", False)
    normalized.setdefault("tool", "context")
    normalized.setdefault("arg1", "")
    normalized.setdefault("arg2", "")
    normalized.setdefault("arg3", "")
    normalized.setdefault("arg4", "")
    normalized.setdefault("num1", 0)
    normalized.setdefault("num2", 0)
    normalized.setdefault("flag1", False)
    normalized.setdefault("flag2", False)
    normalized.setdefault("outcome", "OUTCOME_OK")
    return normalized


def _looks_like_schema_echo(payload: dict[str, Any]) -> bool:
    echo_keys = {"json", "return", "required_top_level_fields"}
    return bool(payload) and set(payload).issubset(echo_keys)
