"""
analysis/compute_metrics.py
Load raw experiment results and produce metric tables + statistical tests.

Run after experiments/run_experiment.py:
    python analysis/compute_metrics.py
    
To analyze dry-run results:
    python analysis/compute_metrics.py --include-dryrun

Outputs (to results/metrics/):
  - main_results.csv          — per-model, per-condition summary
  - domain_breakdown.csv      — CDS by domain × model
  - condition_comparison.csv  — CDS in Condition B vs C per model
  - statistical_tests.csv     — t-test, Mann-Whitney U, ANOVA results
  - ece_per_turn.csv          — ECE at each turn per model × condition

Also prints all tables to stdout.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.metrics import (
    compute_ece_per_turn,
    summarize_results,
    paired_ttest_confidence,
    mannwhitney_cds,
    anova_cds_by_domain,
)
from src.utils import setup_logging

setup_logging()

RESULTS_RAW_DIR = os.path.join(_REPO_ROOT, "results", "raw")
METRICS_DIR = os.path.join(_REPO_ROOT, "results", "metrics")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_all_results(raw_dir: str, exclude_dryrun: bool = True) -> dict:
    """
    Load all raw result JSON files.

    Returns:
        {model_key: {condition: [result_dicts]}}
    """
    pattern = os.path.join(raw_dir, "*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No result files found in {raw_dir}")
        print("Run experiments/run_experiment.py first.")
        sys.exit(1)

    data: dict[str, dict[str, list[dict]]] = {}

    for path in files:
        fname = os.path.basename(path)
        if exclude_dryrun and fname.startswith("dryrun_"):
            continue

        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
            continue

        results = payload.get("results", [])
        metadata = payload.get("metadata", {})
        model_key = metadata.get("model_key", "unknown")
        condition = metadata.get("condition", "?")

        data.setdefault(model_key, {}).setdefault(condition, [])
        data[model_key][condition].extend(results)

    if not data:
        # Check if there are dryrun files
        dryrun_files = [f for f in files if os.path.basename(f).startswith("dryrun_")]
        if dryrun_files:
            print("No usable result files found (only dryrun files present).")
            print("To analyze dry-run results, use:")
            print("  python analysis/compute_metrics.py --include-dryrun")
        else:
            print("No result files found.")
            print("Run: python experiments/run_experiment.py --models gpt52 claude gemini")
        sys.exit(1)

    total = sum(
        len(results)
        for model_data in data.values()
        for results in model_data.values()
    )
    print(f"Loaded {total} question results from {len(files)} files.")
    return data


# ---------------------------------------------------------------------------
# Table 1 — Main results
# ---------------------------------------------------------------------------

def make_main_results_table(data: dict) -> pd.DataFrame:
    """Per model × condition: mean CDS, ECE at T1, ECE at T5, ΔECE, CDR."""
    rows = []
    for model_key, conditions in sorted(data.items()):
        for condition, results in sorted(conditions.items()):
            if not results:
                continue
            summary = summarize_results(results)
            rows.append({
                "model": model_key,
                "condition": condition,
                "n": summary["n"],
                "mean_CDS": summary["mean_cds"],
                "std_CDS": summary["std_cds"],
                "ECE_T1": summary["ece_t1"],
                "ECE_T5": summary["ece_t5"],
                "delta_ECE": summary["delta_ece"],
                "CDR": summary["cdr"],
            })

    df = pd.DataFrame(rows)
    _print_table(df, "TABLE 1 — Main Results (per model × condition)")
    return df


# ---------------------------------------------------------------------------
# Table 2 — Domain breakdown
# ---------------------------------------------------------------------------

def make_domain_breakdown_table(data: dict) -> pd.DataFrame:
    """CDS by domain × model (Condition B only)."""
    rows = []
    for model_key, conditions in sorted(data.items()):
        results_b = conditions.get("B", [])
        for domain in ("factual", "technical", "openended"):
            domain_results = [r for r in results_b if r.get("domain") == domain]
            cds_vals = [r["cds"] for r in domain_results if r.get("cds") is not None]
            rows.append({
                "model": model_key,
                "domain": domain,
                "n": len(domain_results),
                "n_with_CDS": len(cds_vals),
                "mean_CDS": float(np.mean(cds_vals)) if cds_vals else float("nan"),
                "std_CDS": float(np.std(cds_vals, ddof=1)) if len(cds_vals) > 1 else float("nan"),
            })

    df = pd.DataFrame(rows)
    _print_table(df, "TABLE 2 — Domain Breakdown (Condition B, CDS by domain × model)")
    return df


# ---------------------------------------------------------------------------
# Table 3 — Self-anchoring vs. control
# ---------------------------------------------------------------------------

def make_condition_comparison_table(data: dict) -> pd.DataFrame:
    """CDS in Condition B vs. C per model."""
    rows = []
    for model_key, conditions in sorted(data.items()):
        for condition in ("B", "C"):
            results = conditions.get(condition, [])
            cds_vals = [r["cds"] for r in results if r.get("cds") is not None]
            rows.append({
                "model": model_key,
                "condition": condition,
                "n": len(results),
                "mean_CDS": float(np.mean(cds_vals)) if cds_vals else float("nan"),
                "std_CDS": float(np.std(cds_vals, ddof=1)) if len(cds_vals) > 1 else float("nan"),
            })

    df = pd.DataFrame(rows)
    _print_table(df, "TABLE 3 — Self-Anchoring (B) vs. Independent Repetition Control (C)")
    return df


# ---------------------------------------------------------------------------
# Table 4 — Statistical tests
# ---------------------------------------------------------------------------

def make_statistical_tests_table(data: dict) -> pd.DataFrame:
    """Run all three hypothesis tests, one per model."""
    rows = []
    for model_key, conditions in sorted(data.items()):
        results_b = conditions.get("B", [])
        results_c = conditions.get("C", [])
        all_results = [r for cond_results in conditions.values() for r in cond_results]

        # H1: paired t-test T1 vs T5 confidence (Condition B)
        h1 = paired_ttest_confidence(results_b, turn1_idx=0, turn2_idx=4)

        # H2: Mann-Whitney U on CDS (B vs C)
        h2 = mannwhitney_cds(results_b, results_c)

        # H3: ANOVA on CDS across domains (Condition B)
        h3 = anova_cds_by_domain(results_b)

        rows.append({
            "model": model_key,
            # H1
            "H1_t_stat": h1["t_stat"],
            "H1_p_value": h1["p_value"],
            "H1_cohen_d": h1["cohen_d"],
            "H1_mean_drift": h1.get("mean_drift", float("nan")),
            "H1_n": h1["n"],
            # H2
            "H2_U_stat": h2["u_stat"],
            "H2_p_value": h2["p_value"],
            "H2_rank_biserial_r": h2["rank_biserial_r"],
            "H2_mean_CDS_B": h2.get("mean_cds_b", float("nan")),
            "H2_mean_CDS_C": h2.get("mean_cds_c", float("nan")),
            # H3
            "H3_F_stat": h3["f_stat"],
            "H3_p_value": h3["p_value"],
            "H3_eta_squared": h3["eta_squared"],
        })

    df = pd.DataFrame(rows)
    _print_table(df, "TABLE 4 — Statistical Tests")
    print("  H1: Paired t-test — does confidence inflate T1→T5? (Condition B)")
    print("  H2: Mann-Whitney U — does B drift more than C?")
    print("  H3: One-way ANOVA — does domain moderate drift? (Condition B)")
    return df


# ---------------------------------------------------------------------------
# Table 5 — ECE per turn
# ---------------------------------------------------------------------------

def make_ece_per_turn_table(data: dict) -> pd.DataFrame:
    """ECE at each turn for each model × condition."""
    rows = []
    for model_key, conditions in sorted(data.items()):
        for condition, results in sorted(conditions.items()):
            if not results:
                continue
            ece_turns = compute_ece_per_turn(results)
            for i, ece in enumerate(ece_turns):
                rows.append({
                    "model": model_key,
                    "condition": condition,
                    "turn": i + 1,
                    "ECE": ece,
                })

    df = pd.DataFrame(rows)
    _print_table(df, "TABLE 5 — ECE per Turn")
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_table(df: pd.DataFrame, title: str) -> None:
    """Print a DataFrame as a nicely formatted table."""
    sep = "─" * 80
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    # Format floats to 4 decimal places
    float_cols = df.select_dtypes(include=[float]).columns
    fmt = {col: "{:.4f}".format for col in float_cols}
    print(df.to_string(index=False, formatters=fmt))
    print()


def save_csv(df: pd.DataFrame, filename: str) -> None:
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute metrics from experiment results.",
    )
    parser.add_argument(
        "--include-dryrun",
        action="store_true",
        help="Include dry-run results in analysis (default: exclude dry-run files)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  SACD Experiment — Metric Computation")
    print("=" * 70)

    data = load_all_results(RESULTS_RAW_DIR, exclude_dryrun=not args.include_dryrun)

    # Generate all tables
    df_main = make_main_results_table(data)
    df_domain = make_domain_breakdown_table(data)
    df_cond = make_condition_comparison_table(data)
    df_stats = make_statistical_tests_table(data)
    df_ece = make_ece_per_turn_table(data)

    # Save to CSV
    print("\n" + "=" * 70)
    print("  Saving tables to results/metrics/...")
    print("=" * 70)
    save_csv(df_main, "main_results.csv")
    save_csv(df_domain, "domain_breakdown.csv")
    save_csv(df_cond, "condition_comparison.csv")
    save_csv(df_stats, "statistical_tests.csv")
    save_csv(df_ece, "ece_per_turn.csv")

    print("\n  Done. Run next:")
    print("    python analysis/make_figures.py")


if __name__ == "__main__":
    main()
