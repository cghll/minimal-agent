from pathlib import Path

import pytest

from minimal_agent.config import Settings


def test_settings_rejects_missing_llm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        Settings.from_env()


def test_settings_reads_values_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("DATABASE_PATH", "var/test.db")

    settings = Settings.from_env()

    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert settings.database_path == Path("var/test.db")
    assert settings.max_agent_steps == 6
    assert settings.context_char_budget == 24000


def test_settings_rejects_non_positive_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("MAX_AGENT_STEPS", "0")

    with pytest.raises(ValueError, match="MAX_AGENT_STEPS"):
        Settings.from_env()
