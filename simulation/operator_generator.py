"""
operator_generator.py — LLM-Driven Operator Response Generator
================================================================

PURPOSE:
    Generates the operator's response using the LLM.  Supports four
    prompting strategies corresponding to the experiments:

    1. ZERO-SHOT  — No training data.  The LLM receives only a system
                    prompt describing the evacuation scenario.
    2. RAG-SUCCESSFUL — Injects top-K retrieved utterances from the
                        global successful-operator corpus as few-shot
                        examples.
    3. IQL+RAG   — Uses the IQL-selected policy to inject the matching
                   persona profile from config/personas.py, plus
                   per-policy FAISS RAG examples.
    4. IQL+GLOBAL-RAG — Same persona injection as IQL+RAG, but retrieves
                        examples from the global successful-operator
                        corpus instead of the per-policy index.
    5. IQL+PERSONA-ONLY — IQL policy selection + persona profile, NO RAG.
    6. RANDOM+PERSONA  — Random policy selection + persona profile, no RAG.
    7. RANDOM (no persona) — Random policy selected for tracking only;
                             prompt is identical to zero-shot.
    8. RANDOM+RAG      — Random policy selection + per-policy RAG examples,
                         no persona.
    9. RANDOM+RAG+PERSONA — Random policy selection + per-policy RAG +
                            persona profile.
   10. RANDOM+GLOBAL-RAG+PERSONA — Random policy selection + global RAG +
                                   persona profile.

HOW IT WORKS:
    Each strategy has a dedicated prompt builder.  The public function
    generate_operator_reply() dispatches to the correct one based on the
    `strategy` argument.

USAGE:
    from simulation.operator_generator import generate_operator_reply

    # Zero-shot
    reply = generate_operator_reply(history, strategy="zero_shot")

    # RAG over successful operators
    reply = generate_operator_reply(history, strategy="rag_successful")

    # IQL + per-policy RAG
    reply = generate_operator_reply(
        history, strategy="iql_rag",
        policy_name="bob", rag_examples=[...],
    )

    # IQL + global RAG (ablation baseline)
    reply = generate_operator_reply(
        history, strategy="iql_global_rag",
        policy_name="bob", rag_examples=[...],
    )
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DEFAULT_TEMPERATURE_OP, DEFAULT_MAX_TOKENS_OP
from config.personas import PERSONA
from simulation.llm_client import call_llm

FALLBACK_REPLY = "Please evacuate immediately; conditions can worsen quickly."

# Persuasion strategy per IQL policy — describes the TACTIC to use, not the identity of the training persona.
# This avoids the operator confusing the matched policy's identity with the actual resident.
POLICY_STRATEGIES = {
    "bob": (
        "Strategy: Direct authority — state the danger is real and immediate with no softening. "
        "This resident type is dismissive and work-focused. Generic warnings don't work. "
        "Use a firm, unambiguous command: 'The fire will reach you in X minutes. You must leave NOW.' "
        "No empathy needed — directness is the unlock."
    ),
    "niki": (
        "Strategy: Severity confirmation + simple direction — confirm the fire IS serious with a "
        "specific fact ('it's X miles away', 'it can reach you in under 30 minutes'), then give "
        "ONE clear action to take. This resident type is cooperative but underestimates the threat. "
        "Keep it brief. Don't over-explain — just confirm severity and point them in a direction."
    ),
    "lindsay": (
        "Strategy: Authority + concrete plan — give explicit permission to act and provide a clear "
        "destination or vehicle ('take them with you, a vehicle is on the way to X'). "
        "This resident type is held back by responsibility — they need to feel they have authority "
        "and a plan before they can move."
    ),
    "michelle": (
        "Strategy: Challenge safety logic directly — do NOT accept the resident's confidence in "
        "their preparations or safety plan. Use either (a) a firm ultimatum: 'No preparation "
        "can stop this fire — you must leave now', or (b) an empathy challenge: 'If this were "
        "happening to a neighbour, would you tell them to stay?' Generic urgency will not work."
    ),
    "ross": (
        "Strategy: Confirm immediate practical help — immediately state that specific assistance "
        "(a vehicle, a rescue team, a route) is on the way. Do NOT keep repeating urgency. "
        "This resident type already wants to leave but has a logistical blocker. "
        "Address the logistics in your very first sentence."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Builders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_zero_shot_prompt(
    history: List[Dict], max_context: int = 6
) -> str:
    context = history[-max_context:]
    context_text = "\n".join(
        f"{'Operator' if h['role'] == 'operator' else 'Resident'}: "
        f"{h['text'].strip()}"
        for h in context
    )

    # instruction = (
    #     "You are the OPERATOR talking to a RESIDENT during a wildfire "
    #     "evacuation call.\n"
    #     "Read the conversation so far, then produce the next operator reply.\n"
    #     "CRITICAL RULES:\n"
    #     "- KEEP IT BRIEF: Maximum 1-2 short sentences (20-30 words total).\n"
    #     "- Get straight to the point - no elaboration.\n"
    #     # "- Calm, professional, evacuation-focused.\n"
    #     # "- If resident resists, emphasize urgency/danger; if cooperative, "
    #     # "give clear next steps.\n"
    #     "- No role labels, no meta commentary.\n"
    #     "- Avoid gendered pronouns.\n"
    #     "- Be direct and concise.\n"
    #     "- Only use information revealed in the conversation - do not assume "
    #     "anything about the resident.\n"
    #     "Use the similar examples for style guidance."
    # )
    instruction = (
        "You are an emergency OPERATOR on a wildfire evacuation call.\n\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
        # "- Only use information revealed in the conversation — adapt your "
        # "response to what the resident has actually said."
    )
    return (
        f"{instruction}\n\n"
        f"Conversation so far:\n{context_text}\n\n"
        "Operator:"
    ).strip()


def _build_rag_successful_prompt(
    history: List[Dict],
    examples: List[Dict],
    max_context: int = 6,
) -> str:
    context = history[-max_context:]
    context_text = "\n".join(
        f"{'Operator' if h['role'] == 'operator' else 'Resident'}: "
        f"{h['text'].strip()}"
        for h in context
    )

    ex_lines = []
    for ex in examples[:3]:
        text = ex.get("text", "").strip()
        if text:
            ex_lines.append(f"  - {text}")
    examples_text = "\n".join(ex_lines) if ex_lines else "  (no examples available)"

    # instruction = (
    #     "You are the OPERATOR talking to a RESIDENT during a wildfire "
    #     "evacuation call.\n"
    #     f"Use the following sample utterances of the operator: "
    #     f"{examples_text} for style guidance.\n"
    #     "Read the conversation so far, then produce the next operator reply.\n"
    #     "CRITICAL RULES:\n"
    #     "- KEEP IT BRIEF: Maximum 1-2 short sentences (20-30 words total).\n"
    #     "- Get straight to the point - no elaboration.\n"
    #     # "- Calm, professional, evacuation-focused.\n"
    #     # "- If resident resists, emphasize urgency/danger; if cooperative, "
    #     # "give clear next steps.\n"
    #     "- No role labels, no meta commentary.\n"
    #     "- Avoid gendered pronouns.\n"
    #     "- Be direct and concise.\n"
    #     "- Only use information revealed in the conversation - do not assume "
    #     "anything about the resident.\n"
    #     "Use the similar examples for style guidance."
    # )

    instruction = (
        "You are an emergency OPERATOR on a wildfire evacuation call.\n\n"
        "Use the following sample utterances of the operator: "
        "{examples_text} for style guidance.\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
        # "- Only use information revealed in the conversation — adapt your "
        # "response to what the resident has actually said."
    )
    return (
        f"{instruction}\n\n"
        f"Conversation so far:\n{context_text}\n\n"
        f"Example utterances of the operator:\n{examples_text}\n\n"
        "Operator:"
    ).strip()


def _get_persona_block(policy_name: str) -> str:
    """Return the persuasion strategy for the IQL-selected policy."""
    return POLICY_STRATEGIES.get(
        policy_name,
        "Strategy: Address the resident's specific stated concern directly and concisely.",
    )


def _build_iql_rag_prompt(
    history: List[Dict],
    policy_name: str,
    rag_examples: List[Dict],
    max_context: int = 10,
) -> str:
    persona_block = _get_persona_block(policy_name)

    context = history[-max_context:]
    context_text = "\n".join(
        f"{'Operator' if h['role'] == 'operator' else 'Resident'}: "
        f"{h['text'].strip()}"
        for h in context
    )

    example_lines = []
    for ex in rag_examples[:3]:
        r_line = ex.get("resident_text", "").strip()
        o_line = ex.get("operator_text", "").strip()
        if r_line and o_line:
            example_lines.append(f"  Resident: {r_line}\n  Operator: {o_line}")
    ex_block = "\n\n".join(example_lines) if example_lines else None

    instruction = (
        "You are an emergency OPERATOR on a wildfire evacuation call.\n\n"
        "The IQL policy selector recommends the following persuasion strategy "
        f"for this resident:\n\n{persona_block}\n\n"
        "IMPORTANT: If the resident has already revealed a specific concern "
        "in the conversation that differs from this strategy, respond to "
        "THAT concern directly — the conversation always takes priority "
        "over the strategy suggestion.\n\n"
        "RULES:\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
        "- Target the resident's specific barrier. Generic urgency alone "
        "will not work.\n"
    )

    prompt = f"{instruction}\n\nConversation so far:\n{context_text}\n\n"
    if ex_block:
        prompt += f"Reference style examples:\n{ex_block}\n\n"
    prompt += "Operator:"
    return prompt.strip()


def _build_iql_global_rag_prompt(
    history: List[Dict],
    policy_name: str,
    rag_examples: List[Dict],
    max_context: int = 6,
) -> str:
    """Same as _build_iql_rag_prompt but RAG examples come from the global
    successful-operator corpus (dicts with "text" key)."""
    persona_block = _get_persona_block(policy_name)

    context = history[-max_context:]
    context_text = "\n".join(
        f"{'Operator' if h['role'] == 'operator' else 'Resident'}: "
        f"{h['text'].strip()}"
        for h in context
    )

    example_lines = []
    for ex in rag_examples[:3]:
        r_line = ex.get("resident_text", "").strip()
        o_line = ex.get("operator_text", "").strip()
        if r_line and o_line:
            example_lines.append(f"  Resident: {r_line}\n  Operator: {o_line}")
        else:
            text = ex.get("text", "").strip()
            if text:
                example_lines.append(f"  Operator: {text}")
    ex_block = "\n\n".join(example_lines) if example_lines else None

    instruction = (
        "You are an emergency OPERATOR on a wildfire evacuation call.\n\n"
        "The IQL policy selector recommends the following persuasion strategy "
        f"for this resident:\n\n{persona_block}\n\n"
        "IMPORTANT: If the resident has already revealed a specific concern "
        "in the conversation that differs from this strategy, respond to "
        "THAT concern directly — the conversation always takes priority "
        "over the strategy suggestion.\n\n"
        "RULES:\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
        "- Target the resident's specific barrier. Generic urgency alone "
        "will not work.\n"
    )

    prompt = f"{instruction}\n\nConversation so far:\n{context_text}\n\n"
    if ex_block:
        prompt += f"Reference style examples:\n{ex_block}\n\n"
    prompt += "Operator:"
    return prompt.strip()


def _build_random_rag_prompt(
    history: List[Dict],
    rag_examples: List[Dict],
    max_context: int = 10,
) -> str:
    """Random policy + per-policy RAG examples, no persona."""
    context = history[-max_context:]
    context_text = "\n".join(
        f"{'Operator' if h['role'] == 'operator' else 'Resident'}: "
        f"{h['text'].strip()}"
        for h in context
    )

    example_lines = []
    for ex in rag_examples[:3]:
        r_line = ex.get("resident_text", "").strip()
        o_line = ex.get("operator_text", "").strip()
        if r_line and o_line:
            example_lines.append(f"  Resident: {r_line}\n  Operator: {o_line}")
    ex_block = "\n\n".join(example_lines) if example_lines else None

    instruction = (
        "You are an emergency OPERATOR on a wildfire evacuation call.\n\n"
        "RULES:\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
    )

    prompt = f"{instruction}\n\nConversation so far:\n{context_text}\n\n"
    if ex_block:
        prompt += f"Reference style examples:\n{ex_block}\n\n"
    prompt += "Operator:"
    return prompt.strip()


def _build_iql_persona_only_prompt(
    history: List[Dict],
    policy_name: str,
    max_context: int = 10,
) -> str:
    """IQL policy selection + persona profile, no RAG examples."""
    persona_block = _get_persona_block(policy_name)

    context = history[-max_context:]
    context_text = "\n".join(
        f"{'Operator' if h['role'] == 'operator' else 'Resident'}: "
        f"{h['text'].strip()}"
        for h in context
    )

    instruction = (
        "You are an emergency OPERATOR on a wildfire evacuation call.\n\n"
        "The IQL policy selector recommends the following persuasion strategy "
        f"for this resident:\n\n{persona_block}\n\n"
        "IMPORTANT: If the resident has already revealed a specific concern "
        "in the conversation that differs from this strategy, respond to "
        "THAT concern directly — the conversation always takes priority "
        "over the strategy suggestion.\n\n"
        "RULES:\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
        "- Target the resident's specific barrier. Generic urgency alone "
        "will not work.\n"
    )

    return (
        f"{instruction}\n\nConversation so far:\n{context_text}\n\n"
        "Operator:"
    ).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

VALID_STRATEGIES = (
    "zero_shot", "rag_successful",
    "iql_rag", "iql_global_rag", "iql_persona_only",
    "random_persona", "random_no_persona",
    "random_rag", "random_rag_persona", "random_global_rag_persona",
    "fixed_policy",
)


def generate_operator_reply(
    history: List[Dict[str, str]],
    strategy: str = "zero_shot",
    temperature: float = DEFAULT_TEMPERATURE_OP,
    max_tokens: int = DEFAULT_MAX_TOKENS_OP,
    policy_name: Optional[str] = None,
    rag_examples: Optional[List[Dict]] = None,
) -> str:
    """
    Generate the next operator utterance.

    Parameters
    ----------
    history      : Conversation so far.
    strategy     : One of VALID_STRATEGIES.
    temperature  : Sampling temperature.
    max_tokens   : Max response length.
    policy_name  : Required for IQL / random strategies.
    rag_examples : Retrieved examples (not used by persona-only / random).

    Returns
    -------
    Operator reply string.
    """
    if strategy == "zero_shot":
        prompt = _build_zero_shot_prompt(history)

    elif strategy == "rag_successful":
        prompt = _build_rag_successful_prompt(history, rag_examples or [])

    elif strategy == "iql_rag":
        if policy_name is None:
            raise ValueError("policy_name is required for 'iql_rag' strategy.")
        prompt = _build_iql_rag_prompt(history, policy_name, rag_examples or [])

    elif strategy == "iql_global_rag":
        if policy_name is None:
            raise ValueError("policy_name is required for 'iql_global_rag' strategy.")
        prompt = _build_iql_global_rag_prompt(history, policy_name, rag_examples or [])

    elif strategy == "iql_persona_only":
        if policy_name is None:
            raise ValueError("policy_name is required for 'iql_persona_only'.")
        prompt = _build_iql_persona_only_prompt(history, policy_name)

    elif strategy == "random_persona":
        if policy_name is None:
            raise ValueError("policy_name is required for 'random_persona'.")
        prompt = _build_iql_persona_only_prompt(history, policy_name)

    elif strategy == "random_no_persona":
        prompt = _build_zero_shot_prompt(history)

    elif strategy == "random_rag":
        prompt = _build_random_rag_prompt(history, rag_examples or [])

    elif strategy == "random_rag_persona":
        if policy_name is None:
            raise ValueError("policy_name is required for 'random_rag_persona'.")
        prompt = _build_iql_rag_prompt(history, policy_name, rag_examples or [])

    elif strategy == "random_global_rag_persona":
        if policy_name is None:
            raise ValueError("policy_name is required for 'random_global_rag_persona'.")
        prompt = _build_iql_global_rag_prompt(history, policy_name, rag_examples or [])

    elif strategy == "fixed_policy":
        if policy_name is None:
            raise ValueError("policy_name is required for 'fixed_policy'.")
        prompt = _build_iql_persona_only_prompt(history, policy_name)

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Use one of: {VALID_STRATEGIES}"
        )

    reply = call_llm(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        fallback=FALLBACK_REPLY,
    )
    return reply
