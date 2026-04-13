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
    3. IQL+RAG   — Uses the IQL-selected policy's per-policy FAISS
                   index for policy-specific few-shot examples.
    4. IQL+GLOBAL-RAG — Uses IQL policy selection (concern + resources)
                        but retrieves examples from the global
                        successful-operator corpus instead of the
                        per-policy index.  Baseline for ablation.

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


_POLICY_RESOURCES = {
    "bob": {
        "concern": "work commitments and not wanting to be interrupted",
        "resources": [
            "A data-backup crew will arrive within 20 minutes to secure your equipment and files",
            "The evacuation center on Main Street has Wi-Fi, power outlets, and desk space so you can resume work immediately",
            "We will transport your essential work equipment in a separate cargo vehicle",
        ],
    },
    "niki": {
        "concern": "uncertainty about the situation and needing clear direction",
        "resources": [
            "A patrol car will escort you and your husband directly to the evacuation center on Lincoln Road",
            "The evacuation route via Oak Avenue is fully clear and monitored by our units right now",
            "The shelter at Lincoln High School has food, water, cots, and medical staff ready for you",
        ],
    },
    "lindsay": {
        "concern": "children's safety and needing parental approval before leaving",
        "resources": [
            "We are contacting the children's parents right now and will relay their instructions to you",
            "A child-safe transport with car seats is being dispatched to your address",
            "The evacuation center has a dedicated children's area with trained childcare staff",
        ],
    },
    "michelle": {
        "concern": "protecting property and believing the house is prepared",
        "resources": [
            "A fire-protection crew will apply retardant spray to your house immediately after you leave",
            "We are stationing a monitoring team on your street to protect properties during the evacuation",
            "Your address is flagged for priority property-protection — a crew is assigned specifically to your block",
        ],
    },
    "ross": {
        "concern": "stranded passengers with mobility issues who cannot move on their own",
        "resources": [
            "A wheelchair-accessible van is being dispatched to your GPS location right now",
            "Two EMTs with stretchers and mobility equipment will arrive within 15 minutes",
            "We are clearing Route 5 as a dedicated evacuation corridor for your vehicle",
        ],
    },
}


def _build_iql_rag_prompt(
    history: List[Dict],
    policy_name: str,
    rag_examples: List[Dict],
    max_context: int = 10,
) -> str:
    policy_info = _POLICY_RESOURCES.get(policy_name, {})
    concern = policy_info.get("concern", "general safety")
    resources = policy_info.get("resources", [])
    resources_block = "\n".join(f"  - {r}" for r in resources)

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
        f"The IQL policy selector has identified this resident's likely "
        f"core concern: {concern}.\n\n"
        f"RESOURCES YOU CAN OFFER (use these — they are real and authorized):\n"
        f"{resources_block}\n\n"
        "RULES:\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        #"- Name the resource: what vehicle, what team, what timeline.\n"
        #"- Do NOT say 'help is on the way' without specifying WHAT help.\n"
        #"- Do NOT repeat yourself. Each reply must advance the conversation.\n" - try one run with these instructions commented out
        "- Calm, professional tone. No role labels or meta commentary.\n"
        "- Only use information revealed in the conversation — adapt the "
        "resources to what the resident has actually said."
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
    """IQL policy selection + concern/resources, but RAG over ALL successful
    operator utterances (global corpus) instead of the per-policy index.
    Context window matches zero-shot/rag baselines (6 turns)."""
    policy_info = _POLICY_RESOURCES.get(policy_name, {})
    concern = policy_info.get("concern", "general safety")
    resources = policy_info.get("resources", [])
    resources_block = "\n".join(f"  - {r}" for r in resources)

    context = history[-max_context:]
    context_text = "\n".join(
        f"{'Operator' if h['role'] == 'operator' else 'Resident'}: "
        f"{h['text'].strip()}"
        for h in context
    )

    ex_lines = []
    for ex in rag_examples[:3]:
        text = ex.get("text", "").strip()
        if text:
            ex_lines.append(f"  - {text}")
    ex_block = "\n".join(ex_lines) if ex_lines else None

    instruction = (
        "You are an emergency OPERATOR on a wildfire evacuation call.\n\n"
        f"The IQL policy selector has identified this resident's likely "
        f"core concern: {concern}.\n\n"
        f"RESOURCES YOU CAN OFFER (use these — they are real and authorized):\n"
        f"{resources_block}\n\n"
        "RULES:\n"
        "- 1-3 sentences maximum. Be specific, not vague.\n"
        "- Calm, professional tone. No role labels or meta commentary.\n"
        "- Only use information revealed in the conversation — adapt the "
        "resources to what the resident has actually said."
    )

    prompt = f"{instruction}\n\nConversation so far:\n{context_text}\n\n"
    if ex_block:
        prompt += f"Example successful operator utterances:\n{ex_block}\n\n"
    prompt += "Operator:"
    return prompt.strip()


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
    strategy     : "zero_shot" | "rag_successful" | "iql_rag" |
                   "iql_global_rag"
    temperature  : Sampling temperature.
    max_tokens   : Max response length.
    policy_name  : Required for "iql_rag" and "iql_global_rag" strategies.
    rag_examples : Retrieved examples.  For "rag_successful" / "iql_global_rag"
                   these are dicts with "text".  For "iql_rag" they have
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

    elif strategy == "iql_global_rag":
        if policy_name is None:
            raise ValueError("policy_name is required for 'iql_global_rag' strategy.")
        prompt = _build_iql_global_rag_prompt(history, policy_name, rag_examples or [])

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'.  "
            "Use 'zero_shot', 'rag_successful', 'iql_rag', or 'iql_global_rag'."
        )

    reply = call_llm(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        fallback=FALLBACK_REPLY,
    )
    return reply
