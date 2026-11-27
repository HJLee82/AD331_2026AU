from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Also run tests that call the real OpenAI API (costs a fraction of a cent).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: calls the real OpenAI API (run with --live)")
    config.addinivalue_line(
        "markers", "models: uses the real local embedding + cross-encoder models (free, slower)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--live"):
        return
    skip_live = pytest.mark.skip(reason="live API test: run with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def live_llm():
    """Real OpenAI client built from .env. Skips when no key is configured."""
    from dotenv import load_dotenv

    from src.llm import OpenAIChatLLM

    load_dotenv(ROOT / ".env")
    key, model = os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL")
    if not key or not model:
        pytest.skip("OPENAI_API_KEY / OPENAI_MODEL not set in .env")
    return OpenAIChatLLM(api_key=key, model=model, temperature=0.0, timeout_s=30.0)
