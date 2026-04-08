from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse


@dataclass(frozen=True)
class AgentConfig:
    model: str
    openai_api_key: str | None
    openai_base_url: str | None
    max_steps: int = 30
    max_tokens: int = 4096
    request_timeout_seconds: float = 60.0
    json_repair_retries: int = 2
    use_gbnf_grammar: bool = False
    fastpath_mode: Literal["off", "framed", "all"] = "framed"

    @classmethod
    def from_env(cls, model: str) -> "AgentConfig":
        base_url = os.getenv("OPENAI_BASE_URL") or None
        grammar_env = (os.getenv("AGENT_USE_GBNF") or "auto").strip().lower()
        if grammar_env == "auto":
            use_gbnf = _should_use_gbnf(base_url)
        else:
            use_gbnf = grammar_env in {"1", "true", "yes", "on"}
        fastpath_mode = (os.getenv("AGENT_FASTPATH_MODE") or "framed").strip().lower()
        if fastpath_mode not in {"off", "framed", "all"}:
            fastpath_mode = "framed"
        return cls(
            model=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=base_url,
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "30")),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "4096")),
            request_timeout_seconds=float(os.getenv("AGENT_REQUEST_TIMEOUT_SECONDS", "60")),
            json_repair_retries=int(os.getenv("AGENT_JSON_REPAIR_RETRIES", "2")),
            use_gbnf_grammar=use_gbnf,
            fastpath_mode=fastpath_mode,  # type: ignore[arg-type]
        )


def _should_use_gbnf(base_url: str | None) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1"}
