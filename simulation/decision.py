"""
decision.py — Conversation Success / Failure Judge
====================================================

PURPOSE:
    Determines whether a conversation has reached a conclusive outcome:
    SUCCESS (resident agreed to evacuate) or FAILURE (persistent refusal
    or turn-limit reached).

HOW IT WORKS (current):
    • Primary: an LLM reads recent dialogue and decides whether the
      RESIDENT has committed to evacuating (yes / no / unclear).
    • Hard stops: keyword-based refusal streak; turn cap.

    On SUCCESS, conversation_loop still appends a final operator line via
    generate_operator_reply (clarifying / wrap-up).  The string returned
    here is only a short acknowledgement hint for logs / metadata.

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

# --- Legacy regex agreement gate (kept, not used by current success path) ---
AGREEMENT_PATTERNS = [
    r"\bokay\b.*\b(go|leave|head out|evacuate|get ready|pack)",
    r"\balright\b.*\b(go|leave|head out|evacuate|get ready|pack)",
    r"\bfine\b.*\b(go|leave|head out|evacuate|i'll)",
    r"\bokay\b[,.\s]+\s*i\s+will\b",
    r"\balright\b[,.\s]+\s*i\s+will\b",
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
    r"\bi('ll| will)\s+make\s+sure\b.*\b(evacuat|leave|go|route|safe)\b",
]

RECENT_RESIDENT_AGREEMENT_WINDOW = 5


def _resident_explicitly_agreed(utterances: List[Dict[str, str]]) -> bool:
    """Legacy regex gate — retained for reference; success path uses LLM."""
    res_texts = [
        u["text"].lower()
        for u in utterances
        if u["role"] == "resident"
    ][-RECENT_RESIDENT_AGREEMENT_WINDOW:]
    for text in res_texts:
        if any(re.search(p, text) for p in AGREEMENT_PATTERNS):
            return True
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
    """Legacy strict session judge prompt (unused by current success path)."""
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
    """Legacy full-session LLM judge (unused by current success path)."""
    prompt = _build_judge_prompt(utterances, window)
    output = call_llm(prompt, temperature=0.0, max_tokens=8, timeout=20)
    output = output.lower().strip()
    if re.search(r"\bsuccess\b", output):
        return True
    if re.search(r"\bfailure\b", output):
        return False
    return None


def _build_resident_evacuation_commitment_prompt(
    utterances: List[Dict[str, str]], window: int,
) -> str:
    """LLM prompt focused only on whether the resident committed to evacuating."""
    snippet = "\n".join(
        f"{u['role'].capitalize()}: {u['text']}"
        for u in utterances[-window:]
    )
    res_lines = [
        f"- {u['text'].strip()}"
        for u in utterances
        if u["role"] == "resident"
    ]
    resident_only = "\n".join(res_lines)[-2000:]
    return (
        "You are judging ONLY the RESIDENT's side of a wildfire evacuation "
        "phone call.\n\n"
        "Question: Has the resident COMMITTED to evacuating or following "
        "through on leaving (e.g. will go, will get ready, will head out, "
        "will follow the route, agreed to the plan and intends to leave)?\n"
        "- If the resident's LAST message is only thanks / politeness but "
        "an EARLIER resident line already committed to evacuating, answer "
        "YES.\n"
        "- If the resident is still only asking questions, stalling, or "
        "refusing without commitment, answer NO.\n"
        "- If you truly cannot tell, answer UNCLEAR.\n\n"
        "Recent dialogue (both roles):\n"
        f"{snippet}\n\n"
        "All resident lines (most recent last):\n"
        f"{resident_only}\n\n"
        "Answer with EXACTLY ONE WORD: YES, NO, or UNCLEAR.\n\n"
        "Answer:"
    )


def _query_llm_resident_evacuation_commitment(
    utterances: List[Dict[str, str]], window: int,
) -> Optional[bool]:
    """
    True  = resident committed to evacuate.
    False = resident has not committed (still resisting / only questions).
    None  = model output unclear.
    """
    prompt = _build_resident_evacuation_commitment_prompt(utterances, window)
    output = call_llm(prompt, temperature=0.0, max_tokens=6, timeout=25)
    out = output.lower().strip()
    if re.search(r"\byes\b", out) or re.search(r"\bagreed\b", out):
        return True
    if re.search(r"\bno\b", out) or re.search(r"\bnot\b.*\b(agreed|commit)", out):
        return False
    if "unclear" in out or "uncertain" in out:
        return None
    if "yes" in out.replace(" ", ""):
        return True
    if "no" in out[:8]:
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

    # --- LEGACY success path (commented; replaced by LLM resident judge) ---
    # resident_agreed = _resident_explicitly_agreed(utterances)
    # llm_decision = _query_llm_decision(utterances, window=tail_window)
    # if allow_early_stop and resident_agreed and llm_decision is not False:
    #     return True, (
    #         "Acknowledged. Assistance will arrive shortly — "
    #         "please stay safe while leaving the area."
    #     )

    # Primary: LLM judges whether the resident committed to evacuating
    resident_commit = _query_llm_resident_evacuation_commitment(
        utterances, window=tail_window
    )
    if allow_early_stop and resident_commit is True:
        return True, (
            "Acknowledged. Continue following official evacuation routes; "
            "the operator will give a brief closing message next."
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
