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


def _last_turns(utterances: List[Dict[str, str]], n: int = 6) -> str:
    last = utterances[-n:] if len(utterances) >= n else utterances
    return "\n".join(
        f"{'Operator' if u['role'] == 'operator' else 'Resident'}: {u['text']}"
        for u in last
    )


def _query_llm_decision(utterances: List[Dict[str, str]]) -> bool:
    snippet = _last_turns(utterances, n=6)
    prompt = (
        "You are judging a wildfire evacuation call.\n\n"
        "Read the last few turns below and answer ONE WORD ONLY: "
        "YES if the resident has agreed to evacuate, NO if they are still actively resisting.\n\n"
        "Count as YES (agreed):\n"
        "- Explicit commitment: 'okay I'll go', 'fine, let me get ready', 'we'll head out'\n"
        "- Implicit commitment: 'I'll keep everyone ready when help arrives', "
        "'once the team arrives I'll be ready', 'I'll be ready to evacuate'\n"
        "- Asking HOW to evacuate or about logistics WHILE cooperating: "
        "'I just want to make sure my dog gets in the van', "
        "'can you confirm there will be wheelchair access' (they are going, just asking about details)\n"
        "- Shift from resistance to cooperation: stopped arguing against leaving, "
        "now focused on making it work\n\n"
        "Count as NO (still resisting):\n"
        "- Still arguing they are safe: 'I still feel confident in our preparations'\n"
        "- Asking WHETHER to leave: 'how can I be sure evacuating is the right choice?'\n"
        "- Conditional refusal: 'I might leave if things get worse'\n\n"
        "Key distinction: is the resident asking about HOW the evacuation will work "
        "(YES) or WHETHER they should evacuate at all (NO)?\n\n"
        f"{snippet}\n\n"
        "Has the resident agreed to evacuate? Answer YES or NO:"
    )
    output = call_llm(prompt, temperature=0.0, max_tokens=5, timeout=20).lower().strip()
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
