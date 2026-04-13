"""
decision.py — Conversation Success / Failure Judge
====================================================

PURPOSE:
    Determines whether a conversation has reached a conclusive outcome:
    SUCCESS (resident agreed to evacuate) or FAILURE (persistent refusal
    or turn-limit reached).

HOW IT WORKS:
    Primary judge: LLM reasoning.
        The last N turns are sent to the LLM with a classification prompt.
        The LLM replies with one token: SUCCESS, FAILURE, or UNCERTAIN.

    Secondary signal: keyword-based refusal counting.
        If the resident's last 6 turns contain 5+ refusal keywords,
        we declare FAILURE regardless of the LLM's opinion.

    Rules:
        • Early stop only on SUCCESS (resident clearly agreed).
        • FAILURE only after persistent refusal OR turn-limit exceeded.
        • Returns (decision, closing_message).

USAGE:
    from simulation.decision import is_successful_session
    decision, closing = is_successful_session(history)
    # decision: True (success) | False (failure) | None (continue)
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DECISION_MIN_TURNS, DECISION_TAIL_WINDOW, MAX_TURNS,
    MAX_REFUSAL_STREAK,
)
from simulation.llm_client import call_llm


FAILURE_KEYWORDS = [
    "not leaving", "won't go", "will not go", "refuse", "refusing",
    "stay", "staying", "staying put", "we're fine", "don't need to",
    "dont need to", "no danger", "no need", "safe here",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _norm_conversation(
    conv: Union[Dict, List[Dict]],
) -> List[Dict[str, str]]:
    if isinstance(conv, list):
        return [
            {"role": c.get("role", "").lower(), "text": c.get("text") or c.get("content", "")}
            for c in conv if "role" in c
        ]
    if isinstance(conv, dict) and "cumulative_roles" in conv:
        return [
            {"role": r.lower(), "text": t}
            for r, t in zip(conv["cumulative_roles"], conv["cumulative_contents"])
        ]
    raise ValueError("Unsupported conversation format")


def _build_judge_prompt(
    utterances: List[Dict[str, str]], window: int
) -> str:
    snippet = "\n".join(
        f"{u['role'].capitalize()}: {u['text']}"
        for u in utterances[-window:]
    )
    return (
        "Evaluate whether the RESIDENT is cooperating with evacuation.\n"
        "- SUCCESS: The resident has softened their stance after the "
        "operator addressed their specific concern. Signs include: "
        "agreeing conditionally ('okay, if you help with X I'll go'), "
        "asking practical questions about leaving ('where should I go?'), "
        "or showing willingness ('alright, let me get ready').\n"
        "- FAILURE: The resident is firmly refusing and has not shown "
        "any movement toward cooperation despite the operator's efforts.\n"
        "- UNCERTAIN: The resident is still pushing back but has not "
        "firmly refused — the conversation could go either way.\n"
        "Answer with ONE WORD ONLY: SUCCESS, FAILURE, or UNCERTAIN.\n\n"
        f"{snippet}\n\nDecision:"
    )


def _query_llm_decision(
    utterances: List[Dict[str, str]], window: int
) -> Optional[bool]:
    prompt = _build_judge_prompt(utterances, window)
    output = call_llm(prompt, temperature=0.0, max_tokens=8, timeout=20)
    output = output.lower().strip()
    if re.search(r"\bsuccess\b", output):
        return True
    if re.search(r"\bfailure\b", output):
        return False
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════
def is_successful_session(
    conversation: Union[Dict, List[Dict]],
    min_turns: int = DECISION_MIN_TURNS,
    tail_window: int = DECISION_TAIL_WINDOW,
    max_turns: int = MAX_TURNS,
    allow_early_stop: bool = True,
) -> Tuple[Optional[bool], Optional[str]]:
    """
    Evaluate whether the conversation has ended in success or failure.

    Returns
    -------
    (decision, closing_message)
        decision = True → SUCCESS, False → FAILURE, None → keep going.
    """
    utterances = _norm_conversation(conversation)
    n = len(utterances)

    if n < min_turns:
        return None, None

    # Count refusal keywords in recent resident turns
    recent_res = [
        u["text"].lower() for u in utterances[-6:] if u["role"] == "resident"
    ]
    refusal_hits = sum(
        any(kw in line for kw in FAILURE_KEYWORDS) for line in recent_res
    )

    # LLM judge
    llm_decision = _query_llm_decision(utterances, window=tail_window)

    if allow_early_stop and llm_decision is True:
        return True, (
            "Acknowledged. Assistance will arrive shortly — "
            "please stay safe while leaving the area."
        )

    if refusal_hits >= MAX_REFUSAL_STREAK:
        return False, (
            "Understood. The resident repeatedly refused to evacuate. "
            "Logging as unsuccessful."
        )

    if n >= max_turns:
        return False, (
            "Conversation ended due to turn limit. Marking as unsuccessful."
        )

    return None, None
