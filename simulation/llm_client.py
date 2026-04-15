"""
llm_client.py — Unified, Provider-Agnostic LLM Client
=======================================================

PURPOSE:
    Single function call_llm() that works with any supported LLM backend.
    All other modules call this — they never talk to Ollama / OpenAI directly.

    Includes a global TokenTracker that accumulates prompt / completion
    tokens and estimates USD cost for every call.

SUPPORTED PROVIDERS:
    "ollama"  — Local Ollama server (default).  Requires Ollama running
                with the model pulled (e.g.  ollama pull llama3).
    "openai"  — OpenAI-compatible API (GPT-4o-mini, GPT-4o, etc.).
                Set OPENAI_API_KEY and optionally OPENAI_BASE_URL.

USAGE:
    from simulation.llm_client import call_llm, token_tracker
    reply = call_llm("What is 2+2?", temperature=0.3)
    token_tracker.summary()          # print running totals
    token_tracker.total_cost_usd()   # float
"""

import sys
import threading
from pathlib import Path
from typing import Optional, Dict

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    LLM_PROVIDER, LLM_MODEL,
    OLLAMA_URL,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_PROJECT,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Token / Cost Tracker
# ═══════════════════════════════════════════════════════════════════════════════
# Pricing per 1M tokens (gpt-4o-mini as of 2024-07)
_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}
_DEFAULT_PRICING = {"input": 0.15, "output": 0.60}


class TokenTracker:
    """Thread-safe accumulator for prompt / completion tokens and cost."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.num_calls = 0

    def record(self, usage: dict) -> None:
        with self._lock:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            self.num_calls += 1

    def total_cost_usd(self, model: str = LLM_MODEL) -> float:
        prices = _PRICING.get(model, _DEFAULT_PRICING)
        with self._lock:
            cost = (
                self.prompt_tokens * prices["input"] / 1_000_000
                + self.completion_tokens * prices["output"] / 1_000_000
            )
        return cost

    def summary(self, model: str = LLM_MODEL) -> str:
        cost = self.total_cost_usd(model)
        with self._lock:
            lines = (
                f"  LLM calls        : {self.num_calls}\n"
                f"  Prompt tokens     : {self.prompt_tokens:,}\n"
                f"  Completion tokens : {self.completion_tokens:,}\n"
                f"  Total tokens      : {self.total_tokens:,}\n"
                f"  Estimated cost    : ${cost:.6f}"
            )
        return lines

    def print_summary(self, model: str = LLM_MODEL) -> None:
        print("\n" + "=" * 50)
        print("  TOKEN USAGE SUMMARY")
        print("=" * 50)
        print(self.summary(model))
        print("=" * 50 + "\n")

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.num_calls = 0


token_tracker = TokenTracker()


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

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENAI_PROJECT:
        headers["OpenAI-Project"] = OPENAI_PROJECT

    try:
        r = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()

        usage = data.get("usage")
        if usage:
            token_tracker.record(usage)

        return data["choices"][0]["message"]["content"].strip()
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
