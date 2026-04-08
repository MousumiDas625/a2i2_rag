"""
llm_client.py — Unified, Provider-Agnostic LLM Client
=======================================================

PURPOSE:
    Single function call_llm() that works with any supported LLM backend.
    All other modules call this — they never talk to Ollama / OpenAI directly.

SUPPORTED PROVIDERS:
    "ollama"  — Local Ollama server (default).  Requires Ollama running
                with the model pulled (e.g.  ollama pull llama3).
    "openai"  — OpenAI-compatible API (GPT-4o, GPT-3.5, Azure OpenAI, etc.).
                Set OPENAI_API_KEY and optionally OPENAI_BASE_URL.

TO ADD A NEW PROVIDER:
    1. Write a _call_<provider>() function below.
    2. Add the name to _DISPATCH.
    3. Set LLM_PROVIDER in config/settings.py or via env var.

USAGE:
    from simulation.llm_client import call_llm
    reply = call_llm("What is 2+2?", temperature=0.3)
"""

import sys
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    LLM_PROVIDER, LLM_MODEL,
    OLLAMA_URL,
    OPENAI_API_KEY, OPENAI_BASE_URL,
)

# ─────────────────────────────────────────────────────────────────────────────
# Ollama backend
# ─────────────────────────────────────────────────────────────────────────────
def _call_ollama(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except Exception as e:
        print(f"[ERR] Ollama call failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible backend
# ─────────────────────────────────────────────────────────────────────────────
def _call_openai(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY not set.  Export it or set it in config/settings.py."
        )
    try:
        r = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return (
            r.json()["choices"][0]["message"]["content"].strip()
        )
    except Exception as e:
        print(f"[ERR] OpenAI call failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch table — add new providers here
# ─────────────────────────────────────────────────────────────────────────────
_DISPATCH = {
    "ollama": _call_ollama,
    "openai": _call_openai,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════
def call_llm(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 128,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    timeout: int = 90,
    fallback: str = "",
) -> str:
    """
    Send a prompt to the configured LLM and return the response text.

    Parameters
    ----------
    prompt      : The full prompt string.
    temperature : Sampling temperature.
    max_tokens  : Maximum tokens in the response.
    model       : Override the default model name.
    provider    : Override the default provider ("ollama" / "openai").
    timeout     : HTTP timeout in seconds.
    fallback    : String returned if the call fails and no text is generated.

    Returns
    -------
    The model's text response, or *fallback* on error.
    """
    prov = (provider or LLM_PROVIDER).lower()
    mdl  = model or LLM_MODEL

    handler = _DISPATCH.get(prov)
    if handler is None:
        raise ValueError(
            f"Unknown LLM provider '{prov}'.  "
            f"Supported: {list(_DISPATCH.keys())}"
        )

    result = handler(prompt, mdl, temperature, max_tokens, timeout)
    return result if result else fallback
