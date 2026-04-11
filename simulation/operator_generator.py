"""
operator_generator.py — LLM-Driven Operator Response Generator
================================================================

PURPOSE:
    Generates the operator's response using the LLM.  Supports three
    prompting strategies corresponding to the three experiments:

    1. ZERO-SHOT  — No training data.  The LLM receives only a system
                    prompt describing the evacuation scenario.
    2. RAG-SUCCESSFUL — Injects top-K retrieved utterances from the
                        global successful-operator corpus as few-shot
                        examples.
    3. IQL+RAG   — Uses the IQL-selected policy's per-policy FAISS
                   index for policy-specific few-shot examples.

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
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DEFAULT_TEMPERATURE_OP, DEFAULT_MAX_TOKENS_OP
from simulation.llm_client import call_llm

FALLBACK_REPLY = "Please evacuate immediately; conditions can worsen quickly."


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

    instruction = (
        "You are the OPERATOR talking to a RESIDENT during a wildfire "
        "evacuation call.\n"
        "Read the conversation so far, then produce the next operator reply.\n"
        "CRITICAL RULES:\n"
        "- KEEP IT BRIEF: Maximum 1-2 short sentences (20-30 words total).\n"
        "- Get straight to the point - no elaboration.\n"
        "- Calm, professional, evacuation-focused.\n"
        "- If resident resists, emphasize urgency/danger; if cooperative, "
        "give clear next steps.\n"
        "- No role labels, no meta commentary.\n"
        "- Avoid gendered pronouns.\n"
        "- Be direct and concise.\n"
        "- Only use information revealed in the conversation - do not assume "
        "anything about the resident.\n"
        "Use the similar examples for style guidance."
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

    instruction = (
        "You are the OPERATOR talking to a RESIDENT during a wildfire "
        "evacuation call.\n"
        f"Use the following sample utterances of the operator: "
        f"{examples_text} for style guidance.\n"
        "Read the conversation so far, then produce the next operator reply.\n"
        "CRITICAL RULES:\n"
        "- KEEP IT BRIEF: Maximum 1-2 short sentences (20-30 words total).\n"
        "- Get straight to the point - no elaboration.\n"
        "- Calm, professional, evacuation-focused.\n"
        "- If resident resists, emphasize urgency/danger; if cooperative, "
        "give clear next steps.\n"
        "- No role labels, no meta commentary.\n"
        "- Avoid gendered pronouns.\n"
        "- Be direct and concise.\n"
        "- Only use information revealed in the conversation - do not assume "
        "anything about the resident.\n"
        "Use the similar examples for style guidance."
    )
    return (
        f"{instruction}\n\n"
        f"Conversation so far:\n{context_text}\n\n"
        f"Example utterances of the operator:\n{examples_text}\n\n"
        "Operator:"
    ).strip()


def _build_iql_rag_prompt(
    history: List[Dict],
    policy_name: str,
    rag_examples: List[Dict],
    max_context: int = 6,
) -> str:
    policy_id = policy_name
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
            example_lines.append(f"Resident: {r_line}\nOperator: {o_line}")
    ex_block = "\n\n".join(example_lines) if example_lines else None

    instruction = (
        "You are the OPERATOR talking to a RESIDENT during a wildfire "
        "evacuation call.\n"
        f"Use the operator policy style optimized for: {policy_id}.\n"
        "Read the conversation so far, then produce the next operator reply.\n"
        "CRITICAL RULES:\n"
        "- KEEP IT BRIEF: Maximum 1-2 short sentences (20-30 words total).\n"
        "- Get straight to the point - no elaboration.\n"
        "- Calm, professional, evacuation-focused.\n"
        "- If resident resists, emphasize urgency/danger; if cooperative, "
        "give clear next steps.\n"
        "- No role labels, no meta commentary.\n"
        "- Avoid gendered pronouns.\n"
        "- Be direct and concise.\n"
        "- Only use information revealed in the conversation - do not assume "
        "anything about the resident.\n"
        "Use the similar examples for style guidance."
    )
    return (
        f"{instruction}\n\n"
        f"Conversation so far:\n{context_text}\n\n"
        f"Similar example dialogues:\n{ex_block if ex_block else '(none)'}\n\n"
        "Operator:"
    ).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

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
    strategy     : "zero_shot" | "rag_successful" | "iql_rag"
    temperature  : Sampling temperature.
    max_tokens   : Max response length.
    policy_name  : Required for "iql_rag" strategy.
    rag_examples : Retrieved examples.  For "rag_successful" these are
                   dicts with "text".  For "iql_rag" they have
                   "resident_text" and "operator_text".

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

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'.  "
            "Use 'zero_shot', 'rag_successful', or 'iql_rag'."
        )

    reply = call_llm(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        fallback=FALLBACK_REPLY,
    )
    return reply
