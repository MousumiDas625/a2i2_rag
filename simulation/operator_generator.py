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
    7. RANDOM+LOCAL RAG — Random training policy each turn; few-shot examples
       retrieved only from that policy's per-policy index (no persona block).
    8. RANDOM+LOCAL RAG+PERSONA — Same random policy + local retrieval, plus
       the matching persona profile in the prompt.
    9. RANDOM+PERSONA+GLOBAL RAG — Random policy + persona; examples from the
       global successful-operator corpus (like IQL global RAG, but random match).

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


def _operator_instruction_no_persona() -> str:
    """Same intro + RULES as IQL operator prompts, without persona / selector."""
    return (
        "You are an emergency OPERATOR on a wildfire evacuation call.\n\n"
        "RULES:\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
        "- Only use information revealed in the conversation — adapt your "
        "response to what the resident has actually said."
    )


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

    instruction = _operator_instruction_no_persona()
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

    example_lines = []
    for ex in examples[:3]:
        r_line = ex.get("resident_text", "").strip()
        o_line = ex.get("operator_text", "").strip()
        if r_line and o_line:
            example_lines.append(f"  Resident: {r_line}\n  Operator: {o_line}")
        else:
            text = ex.get("text", "").strip()
            if text:
                example_lines.append(f"  Operator: {text}")
    ex_block = "\n\n".join(example_lines) if example_lines else None

    instruction = _operator_instruction_no_persona()
    prompt = f"{instruction}\n\nConversation so far:\n{context_text}\n\n"
    if ex_block:
        prompt += f"Reference style examples:\n{ex_block}\n\n"
    prompt += "Operator:"
    return prompt.strip()


def _get_persona_block(policy_name: str) -> str:
    """Build a persona summary from config/personas.py for the selected policy."""
    persona = PERSONA.get(policy_name)
    if persona and isinstance(persona, dict):
        desc = persona.get("description", "")
        info = persona.get("information", "")
        return f"{desc}\n{info}"
    return "general resident in need of evacuation assistance"


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
        "The IQL policy selector has identified that this resident most "
        f"closely matches the following profile:\n{persona_block}\n\n"
        "RULES:\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
        "- Only use information revealed in the conversation — adapt your "
        "response to what the resident has actually said."
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
        "The IQL policy selector has identified that this resident most "
        f"closely matches the following profile:\n{persona_block}\n\n"
        "RULES:\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
        "- Only use information revealed in the conversation — adapt your "
        "response to what the resident has actually said."
    )

    prompt = f"{instruction}\n\nConversation so far:\n{context_text}\n\n"
    if ex_block:
        prompt += f"Reference style examples:\n{ex_block}\n\n"
    prompt += "Operator:"
    return prompt.strip()


def _build_random_local_rag_persona_prompt(
    history: List[Dict],
    policy_name: str,
    rag_examples: List[Dict],
    max_context: int = 10,
) -> str:
    """Same structure as IQL+per-policy RAG, but policy is random (not IQL)."""
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
        "For this turn, one training operator policy profile was chosen "
        "uniformly at random (not from a learned selector). The profile below "
        "is for conditioning only; the resident on the line may or may not match "
        f"it closely.\n{persona_block}\n\n"
        "Reference examples below were retrieved from that policy's dialogue "
        "index.\n\n"
        "RULES:\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
        "- Only use information revealed in the conversation — adapt your "
        "response to what the resident has actually said."
    )

    prompt = f"{instruction}\n\nConversation so far:\n{context_text}\n\n"
    if ex_block:
        prompt += f"Reference style examples:\n{ex_block}\n\n"
    prompt += "Operator:"
    return prompt.strip()


def _build_random_persona_global_rag_prompt(
    history: List[Dict],
    policy_name: str,
    rag_examples: List[Dict],
    max_context: int = 6,
) -> str:
    """Like IQL+global RAG persona prompt, but policy selection is random."""
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
        "For this turn, one training operator policy profile was chosen "
        "uniformly at random (not from a learned selector). The profile below "
        "is for conditioning only; the resident on the line may or may not match "
        f"it closely.\n{persona_block}\n\n"
        "Reference examples below come from the global successful-operator "
        "corpus (not restricted to that policy's index).\n\n"
        "RULES:\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
        "- Only use information revealed in the conversation — adapt your "
        "response to what the resident has actually said."
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
        "The IQL policy selector has identified that this resident most "
        f"closely matches the following profile:\n{persona_block}\n\n"
        "RULES:\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
        "- Only use information revealed in the conversation — adapt your "
        "response to what the resident has actually said."
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
    "random_persona",
    "random_local_rag", "random_local_rag_persona", "random_persona_global_rag",
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
    policy_name  : Required for IQL / random (persona or RAG) strategies.
    rag_examples : Retrieved examples (not used by zero-shot / persona-only /
                   random_persona without RAG).

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

    elif strategy == "random_local_rag":
        prompt = _build_rag_successful_prompt(history, rag_examples or [])

    elif strategy == "random_local_rag_persona":
        if policy_name is None:
            raise ValueError("policy_name is required for 'random_local_rag_persona'.")
        prompt = _build_random_local_rag_persona_prompt(
            history, policy_name, rag_examples or []
        )

    elif strategy == "random_persona_global_rag":
        if policy_name is None:
            raise ValueError("policy_name is required for 'random_persona_global_rag'.")
        prompt = _build_random_persona_global_rag_prompt(
            history, policy_name, rag_examples or []
        )

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
