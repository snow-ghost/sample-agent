from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    model: str
    openai_api_key: str | None
    openai_base_url: str | None
    max_steps: int = 30
    max_tokens: int = 4096
    json_repair_retries: int = 2

    @classmethod
    def from_env(cls, model: str) -> "AgentConfig":
        return cls(
            model=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "30")),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "4096")),
            json_repair_retries=int(os.getenv("AGENT_JSON_REPAIR_RETRIES", "2")),
        )
