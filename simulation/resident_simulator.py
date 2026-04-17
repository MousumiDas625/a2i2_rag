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
        barrier = persona.get("barrier", "")
        persona_block = (
            f"{scenario}\n\n"
            f"Key facts about you:\n{info}"
        )
        if barrier:
            persona_block += f"\n\nWhat will make you agree to evacuate:\n{barrier}"
    elif persona and isinstance(persona, str):
        display_name = resident_name
        persona_block = persona
    else:
        display_name = resident_name
        persona_block = (
            f"{resident_name} is a resident in the area affected by the "
            "wildfire."
        )

    # Count operator turns to calibrate how much the resident has revealed
    op_turns = sum(1 for h in history if h["role"] == "operator")

    if op_turns <= 1:
        reveal_instruction = (
            "You have just picked up the phone. Express general skepticism or "
            "distraction — do NOT yet reveal your specific barrier. Use phrases "
            "like 'I think we're okay', 'Is it really that bad?', or 'I'm a bit "
            "busy right now'. Keep it brief and non-specific."
        )
    elif op_turns == 2:
        reveal_instruction = (
            "The operator has pushed again. You may now hint at ONE aspect of "
            "what is holding you back, but do not spell out the full picture yet. "
            "Show your concern is real but still express hesitation."
        )
    else:
        reveal_instruction = (
            "The operator has persisted. You may now state your specific concern "
            "clearly. Only agree to evacuate once the operator has DIRECTLY and "
            "CONCRETELY addressed that concern — a vague acknowledgement is not "
            "enough."
        )

    system_block = (
        f"{persona_block}\n\n"
        f"You ARE {display_name}. You are on an emergency phone call "
        "about a wildfire evacuation.\n\n"
        "RULES:\n"
        "- Reply ONLY as the Resident. No role labels, no narration.\n"
        "- 1-3 sentences, natural and conversational.\n"
        "- Stay strictly in character. Only raise concerns that come "
        "from YOUR key facts above — never invent new ones.\n"
        f"- {reveal_instruction}\n"
        "- Generic urgency ('it's dangerous', 'please leave now', "
        "'your safety is the priority') does NOT move you on its own. "
        "When the operator gives only generic urgency, push back "
        "or express your hesitation more firmly.\n"
        "- Express your concern or resistance at least 2-3 times before "
        "you agree to evacuate — one persuasive line is not enough.\n"
        "- Once the operator directly and concretely addresses your specific "
        "barrier, agree to evacuate clearly and briefly. Do not keep asking "
        "for more details after that.\n"
        "- Do NOT agree before your barrier is addressed."
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
