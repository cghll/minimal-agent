from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    database_path: Path
    max_agent_steps: int = 6
    context_char_budget: int = 24_000
    llm_timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        if not api_key:
            raise ValueError("LLM_API_KEY is required")
        if not model:
            raise ValueError("LLM_MODEL is required")

        max_steps = _positive_int("MAX_AGENT_STEPS", 6)
        context_budget = _positive_int("CONTEXT_CHAR_BUDGET", 24_000)
        timeout = _positive_float("LLM_TIMEOUT_SECONDS", 30.0)
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            llm_api_key=api_key,
            llm_model=model,
            database_path=Path(os.getenv("DATABASE_PATH", "data/agent.db")),
            max_agent_steps=max_steps,
            context_char_budget=context_budget,
            llm_timeout_seconds=timeout,
        )


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
