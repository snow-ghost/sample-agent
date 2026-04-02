from __future__ import annotations

import json
import time
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import AgentConfig

ModelT = TypeVar("ModelT", bound=BaseModel)


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
        )

    def complete_json(
        self,
        messages: list[dict[str, str]],
        response_model: type[ModelT],
    ) -> tuple[ModelT, str, int]:
        attempt_messages = list(messages)
        last_error: Exception | None = None

        for attempt in range(self.config.json_repair_retries + 1):
            started = time.time()
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=attempt_messages,
                max_tokens=self.config.max_tokens,
                temperature=0,
            )
            elapsed_ms = int((time.time() - started) * 1000)
            raw_text = _message_text(response.choices[0].message)

            try:
                payload = _extract_json_payload(raw_text)
                return response_model.model_validate(payload), raw_text, elapsed_ms
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
                            "Your previous reply was invalid for the required JSON schema.\n"
                            f"Validation error: {exc}\n"
                            "Return only a corrected JSON object."
                        ),
                    },
                ]

        raise ValueError(f"Unable to parse structured model response: {last_error}")
