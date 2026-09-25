"""Portable transport configuration; no credentials or provider-specific defaults."""
import os
from causal_agent.propose import openai_chat_complete


def credentials():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("Export OPENAI_API_KEY before running API-backed experiments")
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"), key


def completion(model, temperature):
    base, key = credentials()
    return openai_chat_complete(base, key, model, temperature=temperature)
