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
    "staying put", "staying here", "staying home",
    "we're fine", "don't need to", "dont need to",
    "no danger", "no need", "safe here",
    "i'm not going", "not evacuating", "will not leave",
]

AGREEMENT_PATTERNS = [
    r"\bokay\b.*\b(go|leave|head out|evacuate|get ready|pack)",
    r"\balright\b.*\b(go|leave|head out|evacuate|get ready|pack)",
    r"\bfine\b.*\b(go|leave|head out|evacuate|i'll)",
    r"\bi'll go\b", r"\bi will go\b",
    r"\bi'll leave\b", r"\bi will leave\b",
    r"\blet me get ready\b", r"\blet me pack\b",
    r"\bwe'll head out\b", r"\bwe will head out\b",
    r"\bwe'll go\b", r"\bwe will go\b",
    r"\bwe'll leave\b", r"\bwe will leave\b",
    r"\bstart packing\b", r"\bstart getting ready\b",
    r"\bi'll start\b.*\b(pack|ready|leav)",
    r"\blet's go\b", r"\blet us go\b",
    r"\bready to (go|leave|evacuate)\b",
    r"\bi('ll| will) (head|move|evacuate)\b",
    r"\bi('m| am) ready\b",
    r"\bi('ll| will) (get|be) ready\b",
    r"\bwe('re| are) (leaving|going|heading)\b",
    r"\bwe('ll| will) be ready\b",
    r"\bi('ll| will) prepare\b",
    r"\bi can (do that|work with that)\b",
    r"\blet me grab\b.*\b(things|stuff|keys|bag|essentials)",
    r"\bthat works\b",
    r"\bi('ll| will) (start|begin) (getting|packing|preparing)\b",
    r"\bcount (me|us) in\b",
    r"\bokay\b.*\b(that works|makes sense|sounds good)\b",
    r"\bi('ll| will) (come|follow|meet)\b",
]


def _resident_explicitly_agreed(utterances: List[Dict[str, str]]) -> bool:
    """Check if the last resident utterance contains explicit agreement."""
    for u in reversed(utterances):
        if u["role"] == "resident":
            text = u["text"].lower()
            return any(re.search(p, text) for p in AGREEMENT_PATTERNS)
    return False


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
        "You are a STRICT judge evaluating a wildfire evacuation call.\n\n"
        "You must verify ALL THREE of the following before marking SUCCESS:\n\n"
        "  1. CONCERN IDENTIFIED: The resident stated a specific personal "
        "concern (e.g. elderly passengers, pets, children, work project, "
        "property, mobility issues, needing transport).\n\n"
        "  2. CONCERN ADDRESSED: The operator responded to THAT EXACT "
        "concern with a CONCRETE, SPECIFIC plan — not generic urgency. "
        "Examples of concrete plans:\n"
        "     - 'We are sending a van for your passengers'\n"
        "     - 'Animal control will pick up your dogs'\n"
        "     - 'A team will secure your equipment'\n"
        "     - 'We will contact the parents'\n"
        "   Examples that are NOT concrete (just generic urgency):\n"
        "     - 'Please leave now, it is dangerous'\n"
        "     - 'We will handle everything'\n"
        "     - 'Your safety is the priority'\n"
        "     - 'Help is on the way' (without specifying what help)\n\n"
        "  3. EXPLICIT AGREEMENT: The resident clearly committed to "
        "evacuating with phrases like 'okay I'll go', 'let me get "
        "ready', 'fine I'll leave', 'we'll head out'. Merely asking "
        "a follow-up question or sounding calmer does NOT count.\n\n"
        "ALL THREE must be present. If ANY is missing → NOT success.\n\n"
        "- SUCCESS: All three conditions above are clearly met.\n"
        "- FAILURE: The resident is firmly refusing and the operator "
        "has only given generic urgency without a concrete plan.\n"
        "- UNCERTAIN: Anything else — including partial progress.\n\n"
        "When in doubt, choose UNCERTAIN.\n"
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

    # Hard gate: resident's last utterance must contain explicit agreement
    resident_agreed = _resident_explicitly_agreed(utterances)

    # LLM judge (only trust SUCCESS if resident actually said they'll go)
    llm_decision = _query_llm_decision(utterances, window=tail_window)

    if allow_early_stop and llm_decision is True and resident_agreed:
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
