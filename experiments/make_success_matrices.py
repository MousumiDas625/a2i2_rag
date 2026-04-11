#!/usr/bin/env python3
"""
make_success_matrices.py — Per-Experiment Success Rate Matrices
===============================================================

PURPOSE:
    Reads the most recent summary.json produced by each experiment and
    generates three separate success-rate matrices — one per experiment.

    Each matrix has:
        Rows    → resident personas
        Columns → run1 … run5 (individual run outcomes: 1=success, 0=fail)
        + a final "Rate" column showing per-resident success rate

    Output (written to data/runs/matrices_<timestamp>/):
        - zero_shot_matrix.csv
        - rag_successful_matrix.csv
        - iql_policy_matrix.csv
        - zero_shot_matrix.png   (heatmap)
        - rag_successful_matrix.png
        - iql_policy_matrix.png

USAGE:
    python experiments/make_success_matrices.py
    python experiments/make_success_matrices.py \
        --exp1 path/to/exp1/summary.json \
        --exp2 path/to/exp2/summary.json \
        --exp3 path/to/exp3/summary.json
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import RUNS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_summary(prefix: str) -> Path:
    candidates = sorted(RUNS_DIR.glob(f"{prefix}_*/summary.json"), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No summary.json found for '{prefix}' under {RUNS_DIR}. "
            "Run the experiment first."
        )
    return candidates[0]


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_matrix(summary: dict) -> tuple[list, list, np.ndarray, np.ndarray]:
    """
    Returns (residents, run_labels, binary_matrix, rate_vector).

    binary_matrix shape: (n_residents, n_runs)  — values 0 or 1
    rate_vector   shape: (n_residents,)           — fraction in [0, 1]
    """
    results = summary.get("results", [])

    # Sort residents and determine max run index
    residents = sorted({r["resident"] for r in results})
    max_run = max(r["run"] for r in results) if results else 0
    run_labels = [f"Run {i}" for i in range(1, max_run + 1)]

    # Fill matrix: default -1 = missing
    matrix = np.full((len(residents), max_run), -1, dtype=float)
    for r in results:
        row = residents.index(r["resident"])
        col = r["run"] - 1
        matrix[row, col] = float(r["success"])

    # Per-resident success rate (ignoring missing cells)
    with np.errstate(invalid="ignore"):
        valid = matrix >= 0
        rate = np.where(
            valid.sum(axis=1) > 0,
            (matrix * valid).sum(axis=1) / valid.sum(axis=1),
            np.nan,
        )

    return residents, run_labels, matrix, rate


def _print_matrix(
    label: str,
    residents: list,
    run_labels: list,
    matrix: np.ndarray,
    rate: np.ndarray,
) -> None:
    col_w = 8
    name_w = 14
    header = f"  {'Resident':<{name_w}}" + "".join(
        f"{rl:>{col_w}}" for rl in run_labels
    ) + f"{'Rate':>{col_w}}"
    sep = "  " + "-" * (len(header) - 2)

    print(f"\n{'=' * 62}")
    print(f"  SUCCESS MATRIX — {label.upper()}")
    print(f"{'=' * 62}")
    print(header)
    print(sep)
    for i, res in enumerate(residents):
        row_str = f"  {res:<{name_w}}"
        for j in range(len(run_labels)):
            v = matrix[i, j]
            cell = "✓" if v == 1 else ("✗" if v == 0 else "?")
            row_str += f"{cell:>{col_w}}"
        r = rate[i]
        row_str += f"  {r:.0%}" if not np.isnan(r) else f"  {'N/A':>5}"
        print(row_str)
    overall = float(np.nanmean(rate))
    print(sep)
    print(f"  {'OVERALL':<{name_w}}" + " " * (col_w * len(run_labels)) + f"  {overall:.0%}")
    print(f"{'=' * 62}")


def _save_csv(
    out_path: Path,
    residents: list,
    run_labels: list,
    matrix: np.ndarray,
    rate: np.ndarray,
) -> None:
    fieldnames = ["resident"] + run_labels + ["success_rate"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, res in enumerate(residents):
            row = {"resident": res}
            for j, rl in enumerate(run_labels):
                v = matrix[i, j]
                row[rl] = int(v) if v >= 0 else "N/A"
            row["success_rate"] = (
                f"{rate[i]:.4f}" if not np.isnan(rate[i]) else "N/A"
            )
            writer.writerow(row)
    print(f"  CSV  → {out_path}")


def _save_heatmap(
    out_path: Path,
    label: str,
    residents: list,
    run_labels: list,
    matrix: np.ndarray,
    rate: np.ndarray,
) -> None:
    # Append rate column for display
    display = np.hstack([matrix, rate.reshape(-1, 1)])
    col_labels = run_labels + ["Rate"]

    fig, ax = plt.subplots(
        figsize=(max(6, len(col_labels) * 1.2), max(4, len(residents) * 0.7 + 1))
    )

    # Mask missing cells for colouring
    masked = np.ma.masked_where(display < 0, display)
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#cccccc")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(residents)))
    ax.set_yticklabels([r.capitalize() for r in residents], fontsize=10)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Annotate cells
    for i in range(len(residents)):
        for j in range(len(col_labels)):
            v = display[i, j]
            if v < 0:
                txt, colour = "?", "grey"
            elif j == len(col_labels) - 1:          # rate column
                txt, colour = f"{v:.0%}", "black"
            else:
                txt, colour = ("✓" if v == 1 else "✗"), "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=colour, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Success rate")
    ax.set_title(f"Success Matrix — {label}", fontsize=13, pad=18)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PNG  → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-experiment success-rate matrices"
    )
    parser.add_argument("--exp1", default=None, help="Path to exp1 summary.json")
    parser.add_argument("--exp2", default=None, help="Path to exp2 summary.json")
    parser.add_argument("--exp3", default=None, help="Path to exp3 summary.json")
    args = parser.parse_args()

    experiments = [
        ("exp1_zero_shot",      "Zero-Shot",      args.exp1),
        ("exp2_rag_successful", "RAG-Successful",  args.exp2),
        ("exp3_iql_policy",     "IQL+RAG",         args.exp3),
    ]

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = RUNS_DIR / f"matrices_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}\n")

    for prefix, label, path_arg in experiments:
        print(f"── {label} ──────────────────────────────")
        try:
            p = Path(path_arg) if path_arg else _find_latest_summary(prefix)
            summary = _load_summary(p)
            print(f"  Loaded: {p}")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}\n")
            continue

        residents, run_labels, matrix, rate = _build_matrix(summary)
        _print_matrix(label, residents, run_labels, matrix, rate)

        slug = prefix.split("_", 1)[1]          # e.g. "zero_shot"
        _save_csv(out_dir / f"{slug}_matrix.csv",
                  residents, run_labels, matrix, rate)
        _save_heatmap(out_dir / f"{slug}_matrix.png",
                      label, residents, run_labels, matrix, rate)
        print()

    print(f"[DONE] All matrices saved to {out_dir}")


if __name__ == "__main__":
    main()
