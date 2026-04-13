"""
conversation_loop.py — Core Conversation Simulation Loop
==========================================================

PURPOSE:
    Runs a single operator↔resident conversation from start to finish.
    This is the shared engine used by all three experiments and by the
    interactive mode.

HOW IT WORKS:
    1. Seed the conversation (either with an operator opener or a resident
       "Hello? Who is this?").
    2. Alternate turns:
       - OPERATOR: call the appropriate strategy from operator_generator.
       - RESIDENT: call resident_simulator (LLM) or accept typed input
                   if interactive_resident=True.
    3. After each resident turn, run the decision judge.
       - SUCCESS → append closing message and stop.
       - FAILURE → stop.
       - None    → continue.
    4. Return a result dict with full history, status, and metadata.

INTERACTIVE FLAGS:
    interactive_operator : bool — if True, the human types operator replies
                                  (IQL's recommendation is still shown).
    interactive_resident : bool — if True, the human types resident replies.

USAGE:
    from simulation.conversation_loop import run_conversation
    result = run_conversation(
        resident_name="ross",
        strategy="iql_rag",
    )
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MAX_TURNS, K_EXAMPLES, RUNS_DIR,
    DEFAULT_TEMPERATURE_OP, DEFAULT_TEMPERATURE_RES,
    DEFAULT_MAX_TOKENS_OP, DEFAULT_MAX_TOKENS_RES,
)
from simulation.resident_simulator import generate_resident_reply
from simulation.operator_generator import generate_operator_reply
from simulation.decision import is_successful_session


def run_conversation(
    resident_name: str = "ross",
    strategy: str = "iql_rag",
    seed_text: Optional[str] = None,
    max_turns: int = MAX_TURNS,
    k_examples: int = K_EXAMPLES,
    temperature_op: float = DEFAULT_TEMPERATURE_OP,
    temperature_res: float = DEFAULT_TEMPERATURE_RES,
    interactive_operator: bool = False,
    interactive_resident: bool = False,
    selector=None,
    run_id: Optional[str] = None,
) -> Dict:
    """
    Run one full conversation.

    Parameters
    ----------
    resident_name        : Persona key (e.g. "ross", "bob").
    strategy             : "zero_shot" | "rag_successful" | "iql_rag"
    seed_text            : Optional first operator line.
    max_turns            : Hard cap on total turns.
    k_examples           : Number of RAG examples per operator turn.
    temperature_op/res   : LLM temperatures.
    interactive_operator : Human types operator lines.
    interactive_resident : Human types resident lines.
    selector             : Pre-initialised IQLPolicySelector (for iql_rag).
    run_id               : Optional tag for output filenames.

    Returns
    -------
    {
        "status": "SUCCESS" | "FAILURE",
        "success": 0 | 1,
        "turns": int,
        "history": [...],
        "path": str,
    }
    """
    # Lazy imports to avoid circular / heavy loads when not needed
    if strategy == "iql_rag":
        from retrieval.policy_selector import IQLPolicySelector
        from retrieval.rag_retrieval import retrieve_topk_pairs
        if selector is None:
            selector = IQLPolicySelector()
    if strategy == "rag_successful":
        from retrieval.rag_retrieval import retrieve_from_successful

    ts = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_DIR / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Seed history ──────────────────────────────────────────────────────────
    history: List[Dict[str, str]] = []
    if seed_text and seed_text.strip():
        history.append({"role": "operator", "text": seed_text.strip()})
        operator_next = False
    else:
        history.append({"role": "resident", "text": "Hello? Who is this?"})
        operator_next = True

    print(f"\n[START] {strategy} | resident={resident_name} | run={ts}")
    print("=" * 60)

    for turn_idx in range(1, max_turns + 1):
        if operator_next:
            # Need at least one resident line before operator can respond
            if not any(h["role"] == "resident" for h in history):
                operator_next = False
                continue

            # ── OPERATOR TURN ──────────────────────────────────────────────
            if interactive_operator:
                # Show IQL recommendation if available
                if strategy == "iql_rag" and selector is not None:
                    best_policy, qvals = selector.select_policy(history)
                    q_str = ", ".join(f"{k}: {v:.3f}" for k, v in qvals.items())
                    print(f"  [IQL recommends: {best_policy}]  Q: {{{q_str}}}")
                op_reply = input("  Operator> ").strip()
                if not op_reply:
                    op_reply = "Please evacuate for your safety."
                history.append({"role": "operator", "text": op_reply})
            else:
                rag_examples: list = []
                policy_name: Optional[str] = None

                if strategy == "iql_rag":
                    best_policy, qvals = selector.select_policy(history)
                    policy_name = best_policy
                    last_res = next(
                        (h["text"] for h in reversed(history) if h["role"] == "resident"), ""
                    )
                    rag_examples = retrieve_topk_pairs(best_policy, last_res, k=k_examples)
                    q_str = ", ".join(f"{k}: {v:.3f}" for k, v in qvals.items())
                    print(f"[TURN {turn_idx}] Policy: {best_policy} | Q: {{{q_str}}}")

                elif strategy == "rag_successful":
                    last_res = next(
                        (h["text"] for h in reversed(history) if h["role"] == "resident"), ""
                    )
                    rag_examples = retrieve_from_successful(last_res, k=k_examples)

                op_reply = generate_operator_reply(
                    history,
                    strategy=strategy,
                    temperature=temperature_op,
                    max_tokens=DEFAULT_MAX_TOKENS_OP,
                    policy_name=policy_name,
                    rag_examples=rag_examples,
                )

                entry: dict = {"role": "operator", "text": op_reply}
                if policy_name:
                    entry["selected_policy"] = policy_name
                    entry["examples_used"] = rag_examples
                history.append(entry)

            print(f"Operator: {history[-1]['text']}\n")
            operator_next = False
            continue

        # ── RESIDENT TURN ─────────────────────────────────────────────────
        if interactive_resident:
            res_reply = input("  Resident> ").strip()
            if not res_reply:
                res_reply = "I'm not sure about that."
        else:
            res_reply = generate_resident_reply(
                history, resident_name,
                temperature=temperature_res,
                max_tokens=DEFAULT_MAX_TOKENS_RES,
            )

        history.append({"role": "resident", "text": res_reply})
        print(f"Resident: {res_reply}\n")

        # ── Decision check ────────────────────────────────────────────────
        decision, closing_msg = is_successful_session(
            history, max_turns=max_turns, allow_early_stop=True,
        )

        if decision is True:
            print("[OK] Resident agreed to evacuate.")
            # Generate a final operator turn that acknowledges the
            # resident's cooperation and answers any remaining questions
            rag_examples_close: list = []
            policy_close: Optional[str] = None
            if strategy == "iql_rag" and selector is not None:
                policy_close, _ = selector.select_policy(history)
                last_res = next(
                    (h["text"] for h in reversed(history) if h["role"] == "resident"), ""
                )
                rag_examples_close = retrieve_topk_pairs(policy_close, last_res, k=k_examples)
            elif strategy == "rag_successful":
                last_res = next(
                    (h["text"] for h in reversed(history) if h["role"] == "resident"), ""
                )
                rag_examples_close = retrieve_from_successful(last_res, k=k_examples)

            close_reply = generate_operator_reply(
                history,
                strategy=strategy,
                temperature=temperature_op,
                max_tokens=DEFAULT_MAX_TOKENS_OP,
                policy_name=policy_close,
                rag_examples=rag_examples_close,
            )
            history.append({"role": "operator", "text": close_reply})
            print(f"Operator (closing): {close_reply}\n")
            break
        elif decision is False:
            print("[X] Resident refused / turn limit reached.")
            break

        operator_next = True
        time.sleep(0.3)

    # ── Persist ───────────────────────────────────────────────────────────────
    out_file = out_dir / f"dialogue_{resident_name}_{ts}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for h in history:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")

    decision_final, _ = is_successful_session(history)
    success = bool(decision_final)
    status = "SUCCESS" if success else "FAILURE"
    print(f"[FINAL] {status} | {len(history)} turns | → {out_file.name}")

    return {
        "status": status,
        "success": int(success),
        "turns": len(history),
        "history": history,
        "path": str(out_file),
    }
