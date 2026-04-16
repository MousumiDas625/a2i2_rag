"""
decision.py — Conversation Success / Failure Judge
====================================================

PURPOSE:
    Determines whether a conversation has reached a conclusive outcome:
    SUCCESS (resident agreed to evacuate) or FAILURE (turn-limit reached).

HOW IT WORKS:
    LLM judge looks at the last 2 resident utterances and decides whether
    the resident has agreed to evacuate. Returns YES or NO.
    FAILURE fires only when the turn limit is reached without agreement.

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
from config.settings import DECISION_MIN_TURNS, MAX_TURNS
from simulation.llm_client import call_llm


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


def _last_resident_utterances(utterances: List[Dict[str, str]], n: int = 2) -> str:
    resident_turns = [u["text"] for u in utterances if u["role"] == "resident"]
    last = resident_turns[-n:] if len(resident_turns) >= n else resident_turns
    return "\n".join(f"Resident: {t}" for t in last)


def _query_llm_decision(utterances: List[Dict[str, str]]) -> bool:
    snippet = _last_resident_utterances(utterances, n=2)
    prompt = (
        "You are judging a wildfire evacuation call.\n\n"
        "Look at the resident's last 1-2 utterances below and answer "
        "ONE WORD ONLY: YES if the resident has clearly agreed to evacuate, "
        "NO otherwise.\n\n"
        "Agreement means they committed to leaving — e.g. 'okay I'll go', "
        "'fine, let me get ready', 'we'll head out', 'I'll start packing'. "
        "Asking questions, sounding calmer, or saying 'maybe' does NOT count.\n\n"
        f"{snippet}\n\n"
        "Has the resident agreed to evacuate? Answer YES or NO:"
    )
    output = call_llm(prompt, temperature=0.0, max_tokens=4, timeout=20).lower().strip()
    return bool(re.search(r"\byes\b", output))


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════
def is_successful_session(
    conversation: Union[Dict, List[Dict]],
    min_turns: int = DECISION_MIN_TURNS,
    max_turns: int = MAX_TURNS,
    allow_early_stop: bool = True,
) -> Tuple[Optional[bool], Optional[str]]:
    """
    Returns
    -------
    (decision, closing_message)
        True  → SUCCESS, False → FAILURE, None → keep going.
    """
    utterances = _norm_conversation(conversation)
    n = len(utterances)

    if n < min_turns:
        return None, None

    if allow_early_stop and _query_llm_decision(utterances):
        return True, (
            "Acknowledged. Assistance will arrive shortly — "
            "please stay safe while leaving the area."
        )

    if n >= max_turns:
        return False, (
            "Conversation ended due to turn limit. Marking as unsuccessful."
        )

    return None, None
