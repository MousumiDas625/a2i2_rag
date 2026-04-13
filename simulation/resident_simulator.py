"""
resident_simulator.py — LLM-Driven Resident Response Generator
================================================================

PURPOSE:
    Generates the resident's side of the conversation using the LLM,
    grounded in the persona description from config/personas.py.

HOW IT WORKS:
    1. Looks up the resident's persona description.
    2. Builds a prompt with:
       - The persona + wildfire evacuation context.
       - Rules for realistic, in-character replies.
       - The last N turns of conversation history.
    3. Calls call_llm() and returns the raw text.

    Replaces the previous external dependency on a separate resident
    backend service.

USAGE:
    from simulation.resident_simulator import generate_resident_reply
    reply = generate_resident_reply(history, "ross")
"""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.personas import PERSONA
from config.settings import DEFAULT_TEMPERATURE_RES, DEFAULT_MAX_TOKENS_RES
from simulation.llm_client import call_llm


def _build_resident_prompt(
    history: List[Dict[str, str]],
    resident_name: str,
    max_context: int = 6,
) -> str:
    persona = PERSONA.get(resident_name)

    if persona and isinstance(persona, dict):
        display_name = persona.get("name", resident_name)
        scenario = persona.get("scenario", "")
        info = persona.get("information", "")
        persona_block = (
            f"{scenario}\n\n"
            f"Key facts about you:\n{info}"
        )
    elif persona and isinstance(persona, str):
        display_name = resident_name
        persona_block = persona
    else:
        display_name = resident_name
        persona_block = (
            f"{resident_name} is a resident in the area affected by the "
            "wildfire."
        )

    # system_block = (
    #     f"{persona_block}\n\n"
    #     "You are currently receiving an emergency call from an operator who "
    #     "is trying to persuade you to evacuate due to a nearby wildfire.\n"
    #     f"Stay STRICTLY in character as {display_name} throughout the "
    #     "entire conversation — NEVER break character.\n\n"
    #     "CRITICAL RULES:\n"
    #     "- Reply ONLY as the Resident — do NOT include any role label or "
    #     "meta commentary.\n"
    #     "- 1–3 natural, conversational sentences. No elaboration.\n"
    #     "- Do NOT repeat the operator's words back verbatim.\n"
    #     "- Show realistic emotions that match your persona: hesitation, "
    #     "doubt, fear, practicality, or urgency as appropriate.\n"
    #     "- Do NOT agree to evacuate immediately — you need the operator to "
    #     "genuinely address the specific concerns described in your "
    #     "situation above before you are convinced.\n"
    #     "- Only agree to evacuate once the operator has directly addressed "
    #     "what is holding you back (your work, your pets, your passengers, "
    #     "your property, your responsibilities, etc.).\n"
    #     "- Your willingness should build gradually across turns as concerns "
    #     "are addressed — a single persuasive line is not enough.\n"
    #     "- Only use information revealed in the conversation — do not "
    #     "introduce new facts about yourself that contradict your persona."
    # )

    system_block = (
        f"{persona_block}\n\n"
        f"You ARE {display_name}. You are receiving an emergency phone "
        "call about a wildfire evacuation. React exactly as this person "
        "would in real life.\n\n"
        "RULES:\n"
        "- Reply ONLY as the Resident. No role labels, no narration.\n"
        "- 1–3 sentences, natural and conversational.\n"
        "- Do NOT parrot back what the operator said.\n"
        "- You have real concerns from your situation above. Bring them "
        "up naturally.\n\n"
        "YOUR BEHAVIOUR:\n"
        "- You will NOT leave just because someone tells you to. You "
        "have real responsibilities and concerns.\n"
        "- If the operator gives GENERIC urgency ('leave now', 'it is "
        "dangerous') without showing they know YOUR specific situation "
        "→ push back FIRMLY. Tell them exactly what your problem is. "
        "Ask them: how are you going to help with MY situation?\n"
        "- If the operator vaguely acknowledges your concern ('I "
        "understand', 'help is on the way', 'we will handle it') but "
        "gives NO specific details about HOW → do NOT accept that. "
        "Press them for specifics. What vehicle? What team? When? "
        "You need a REAL answer, not empty promises.\n"
        "- ONLY agree to evacuate if the operator gives you a SPECIFIC, "
        "DETAILED plan that names YOUR concern and explains exactly "
        "what they will do about it. For example:\n"
        "  * 'We are dispatching a wheelchair-accessible van to your "
        "location on Route 5' — this is specific.\n"
        "  * 'We will send someone to help' — this is NOT specific "
        "enough. Ask: who? when? how?\n"
        "- When you DO agree, say it CLEARLY: 'Okay, I'll go', "
        "'Alright, let me get ready to leave', 'Fine, we'll head out'."
        "\n\n"
        "- Do NOT invent new problems outside your persona.\n"
        "- Once your concern is addressed with a real, specific plan, "
        "stop resisting and agree clearly."
    )

    context = history[-max_context:] if len(history) > max_context else history
    dialogue_lines = "\n".join(
        f"{'Operator' if h['role'] == 'operator' else 'Resident'}: "
        f"{h['text'].strip()}"
        for h in context
    )

    return (
        f"{system_block}\n\n"
        f"Conversation so far:\n{dialogue_lines}\n\n"
        "Resident:"
    ).strip()


def generate_resident_reply(
    history: List[Dict[str, str]],
    resident_name: str,
    temperature: float = DEFAULT_TEMPERATURE_RES,
    max_tokens: int = DEFAULT_MAX_TOKENS_RES,
) -> str:
    """
    Generate the next resident utterance.

    Parameters
    ----------
    history       : Conversation so far (list of {role, text} dicts).
    resident_name : Key into PERSONA dict.
    temperature   : Sampling temperature.
    max_tokens    : Max response length.

    Returns
    -------
    The resident's reply as a plain string.
    """
    prompt = _build_resident_prompt(history, resident_name)
    reply = call_llm(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        fallback="I'm not sure I want to leave right now.",
    )
    return reply
