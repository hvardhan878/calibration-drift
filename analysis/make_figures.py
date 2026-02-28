"""
analysis/make_figures.py
Generate all publication-quality figures for the SACD paper.

Run after analysis/compute_metrics.py:
    python analysis/make_figures.py
    
To generate figures from dry-run results:
    python analysis/make_figures.py --include-dryrun

Output (to results/figures/):
  - fig1_confidence_across_turns.png / .pdf
  - fig2_ece_across_turns.png / .pdf
  - fig3_reliability_diagrams.png / .pdf
  - fig4_selfanchoring_vs_control.png / .pdf
  - fig5_domain_model_heatmap.png / .pdf
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Suppress matplotlib font warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.metrics import compute_ece, compute_ece_per_turn
from src.utils import setup_logging

setup_logging()

RESULTS_RAW_DIR = os.path.join(_REPO_ROOT, "results", "raw")
FIGURES_DIR = os.path.join(_REPO_ROOT, "results", "figures")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font="Arial", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

DOMAIN_COLORS = {
    "factual": "#2196F3",    # blue
    "technical": "#4CAF50",  # green
    "openended": "#FF9800",  # orange
}
DOMAIN_LABELS = {
    "factual": "Factual",
    "technical": "Technical",
    "openended": "Open-ended",
}
MODEL_DISPLAY = {
    "gpt52": "GPT-5.2",
    "claude": "Claude Sonnet 4.6",
    "gemini": "Gemini 3.1 Pro",
}
CONDITION_COLORS = {
    "A": "#9E9E9E",  # grey
    "B": "#E91E63",  # pink/red
    "C": "#3F51B5",  # indigo
}


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_all_results(raw_dir: str, exclude_dryrun: bool = True) -> dict:
    """Load raw results into {model_key: {condition: [results]}}."""
    pattern = os.path.join(raw_dir, "*.json")
    files = sorted(glob.glob(pattern))
    data: dict[str, dict[str, list[dict]]] = {}

    for path in files:
        fname = os.path.basename(path)
        if exclude_dryrun and fname.startswith("dryrun_"):
            continue
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception:
            continue

        results = payload.get("results", [])
        meta = payload.get("metadata", {})
        model_key = meta.get("model_key", "unknown")
        condition = meta.get("condition", "?")
        data.setdefault(model_key, {}).setdefault(condition, [])
        data[model_key][condition].extend(results)

    if not data:
        # Check if there are dryrun files
        dryrun_files = [f for f in files if os.path.basename(f).startswith("dryrun_")]
        if dryrun_files:
            print("No usable result files found (only dryrun files present).")
            print("To generate figures from dry-run results, use:")
            print("  python analysis/make_figures.py --include-dryrun")
        else:
            print("No result files found. Run the experiment first.")
    return data


def extract_confidence_series(
    results: list[dict],
    domain: str | None = None,
) -> dict[int, list[float]]:
    """
    Returns {turn_num: [confidence_values]} for all turns in results.
    Optionally filter by domain.
    """
    by_turn: dict[int, list[float]] = {}
    for r in results:
        if domain and r.get("domain") != domain:
            continue
        for turn in r.get("turns", []):
            t = turn.get("turn")
            c = turn.get("extracted_confidence")
            if t is not None and c is not None:
                by_turn.setdefault(t, []).append(c)
    return by_turn


def mean_ci_by_turn(
    results: list[dict],
    domain: str | None = None,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Returns (turns, means, ci_lo, ci_hi) for confidence across turns."""
    by_turn = extract_confidence_series(results, domain)
    turns = sorted(by_turn.keys())
    means, lo, hi = [], [], []
    for t in turns:
        vals = np.array(by_turn[t])
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        means.append(mean)
        lo.append(mean - 1.96 * se)
        hi.append(mean + 1.96 * se)
    return turns, means, lo, hi


def mean_accuracy_by_turn(
    results: list[dict],
    domain: str | None = None,
) -> float | None:
    """Overall mean accuracy (for dashed reference line)."""
    correct_vals = []
    for r in results:
        if domain and r.get("domain") != domain:
            continue
        for turn in r.get("turns", []):
            c = turn.get("correct")
            if c is not None:
                correct_vals.append(int(c))
    return float(np.mean(correct_vals)) if correct_vals else None


def _save_fig(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Confidence Across Turns
# ---------------------------------------------------------------------------

def make_figure1(data: dict) -> None:
    """
    Line plot: mean confidence by turn, per domain, per model (3 panels).
    Error bars: 95% CI. Dashed line: mean accuracy.
    """
    models = sorted(data.keys())
    n_models = len(models)
    if n_models == 0:
        print("  Skipping Fig 1 — no data")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_key in zip(axes, models):
        results_b = data[model_key].get("B", [])
        if not results_b:
            continue

        for domain in ("factual", "technical", "openended"):
            turns, means, lo, hi = mean_ci_by_turn(results_b, domain)
            if not turns:
                continue
            color = DOMAIN_COLORS[domain]
            label = DOMAIN_LABELS[domain]
            ax.plot(turns, means, "o-", color=color, label=label, linewidth=2, markersize=5)
            ax.fill_between(turns, lo, hi, alpha=0.15, color=color)

            # Dashed accuracy line
            acc = mean_accuracy_by_turn(results_b, domain)
            if acc is not None:
                ax.axhline(acc, linestyle="--", color=color, alpha=0.5, linewidth=1)

        ax.set_title(MODEL_DISPLAY.get(model_key, model_key), fontsize=12, fontweight="bold")
        ax.set_xlabel("Turn", fontsize=11)
        ax.set_xlim(0.5, 5.5)
        ax.set_xticks(range(1, 6))
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    axes[0].set_ylabel("Mean Confidence", fontsize=11)
    axes[0].legend(title="Domain", fontsize=9, title_fontsize=9, loc="lower left")

    fig.suptitle(
        "Figure 1 — Expressed Confidence Across Turns (Condition B: Self-Anchoring)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, "fig1_confidence_across_turns")


# ---------------------------------------------------------------------------
# Figure 2 — ECE Across Turns
# ---------------------------------------------------------------------------

def make_figure2(data: dict) -> None:
    """Same structure as Figure 1 but Y-axis = ECE."""
    models = sorted(data.keys())
    n_models = len(models)
    if n_models == 0:
        print("  Skipping Fig 2 — no data")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_key in zip(axes, models):
        results_b = data[model_key].get("B", [])
        if not results_b:
            continue

        for domain in ("factual", "technical", "openended"):
            # Compute ECE per turn for this domain
            domain_results = [r for r in results_b if r.get("domain") == domain]
            if not domain_results:
                continue
            ece_turns = compute_ece_per_turn(domain_results)
            turns = list(range(1, len(ece_turns) + 1))
            valid = [(t, e) for t, e in zip(turns, ece_turns) if not np.isnan(e)]
            if not valid:
                continue
            valid_turns, valid_ece = zip(*valid)
            color = DOMAIN_COLORS[domain]
            ax.plot(valid_turns, valid_ece, "o-", color=color,
                    label=DOMAIN_LABELS[domain], linewidth=2, markersize=5)

        ax.set_title(MODEL_DISPLAY.get(model_key, model_key), fontsize=12, fontweight="bold")
        ax.set_xlabel("Turn", fontsize=11)
        ax.set_xlim(0.5, 5.5)
        ax.set_xticks(range(1, 6))
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Expected Calibration Error (ECE)", fontsize=11)
    axes[0].legend(title="Domain", fontsize=9, title_fontsize=9, loc="upper left")

    fig.suptitle(
        "Figure 2 — Calibration Error Across Turns (Condition B: Self-Anchoring)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, "fig2_ece_across_turns")


# ---------------------------------------------------------------------------
# Figure 3 — Reliability Diagrams
# ---------------------------------------------------------------------------

def _reliability_diagram(
    ax: plt.Axes,
    results: list[dict],
    turn_idx: int,
    domain: str,
    color: str,
    label: str,
    n_bins: int = 10,
) -> None:
    """Draw a reliability diagram for one turn on the given axes."""
    confs, corrs = [], []
    for r in results:
        if r.get("domain") != domain:
            continue
        turns = r.get("turns", [])
        if turn_idx < len(turns):
            c = turns[turn_idx].get("extracted_confidence")
            ok = turns[turn_idx].get("correct")
            if c is not None and ok is not None:
                confs.append(c)
                corrs.append(int(ok))

    if not confs:
        return

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [lo <= c < hi for c in confs] if i < n_bins - 1 else [lo <= c <= hi for c in confs]
        if not any(mask):
            continue
        bc = (lo + hi) / 2
        bacc = np.mean([corrs[j] for j, m in enumerate(mask) if m])
        bin_centers.append(bc)
        bin_accs.append(bacc)

    ax.plot(bin_centers, bin_accs, "o-", color=color, label=label, linewidth=2, markersize=5)


def make_figure3(data: dict) -> None:
    """
    3×2 grid (3 domains × 2 turns: T1 vs T5).
    T1 in blue, T5 in red, perfect calibration diagonal.
    Uses first available model with Condition B data.
    """
    # Find first model with B data
    model_key = None
    for mk, conds in data.items():
        if conds.get("B"):
            model_key = mk
            break

    if model_key is None:
        print("  Skipping Fig 3 — no Condition B data")
        return

    results_b = data[model_key]["B"]
    domains = ["factual", "technical", "openended"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, sharex=True)

    for col, domain in enumerate(domains):
        ax = axes[col]
        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfect calibration")

        _reliability_diagram(ax, results_b, 0, domain, "#2196F3", "Turn 1")
        _reliability_diagram(ax, results_b, 4, domain, "#E91E63", "Turn 5")

        ax.set_title(DOMAIN_LABELS[domain], fontsize=12, fontweight="bold")
        ax.set_xlabel("Confidence", fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))

    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].legend(fontsize=9, loc="upper left")

    fig.suptitle(
        f"Figure 3 — Reliability Diagrams: T1 vs T5 Calibration "
        f"({MODEL_DISPLAY.get(model_key, model_key)}, Condition B)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, "fig3_reliability_diagrams")


# ---------------------------------------------------------------------------
# Figure 4 — Self-Anchoring vs. Control
# ---------------------------------------------------------------------------

def make_figure4(data: dict) -> None:
    """
    Grouped bar chart: CDS for Condition B vs C per model.
    Shows that self-anchoring produces more drift than repetition alone.
    """
    models = sorted(data.keys())
    if not models:
        print("  Skipping Fig 4 — no data")
        return

    rows = []
    for model_key in models:
        for condition in ("B", "C"):
            results = data[model_key].get(condition, [])
            cds_vals = [r["cds"] for r in results if r.get("cds") is not None]
            if cds_vals:
                rows.append({
                    "model": MODEL_DISPLAY.get(model_key, model_key),
                    "condition": f"Condition {condition}",
                    "mean_CDS": float(np.mean(cds_vals)),
                    "sem_CDS": float(np.std(cds_vals, ddof=1) / np.sqrt(len(cds_vals))),
                })

    if not rows:
        print("  Skipping Fig 4 — no CDS data for B or C")
        return

    df = pd.DataFrame(rows)
    n_models = df["model"].nunique()
    fig, ax = plt.subplots(figsize=(max(6, 2 * n_models + 2), 5))

    x = np.arange(n_models)
    width = 0.35
    model_names = df["model"].unique()

    cond_b = df[df["condition"] == "Condition B"].set_index("model")
    cond_c = df[df["condition"] == "Condition C"].set_index("model")

    bars_b = ax.bar(
        x - width / 2,
        [cond_b.loc[m, "mean_CDS"] if m in cond_b.index else 0 for m in model_names],
        width,
        yerr=[cond_b.loc[m, "sem_CDS"] if m in cond_b.index else 0 for m in model_names],
        label="Condition B (Self-Anchoring)",
        color=CONDITION_COLORS["B"],
        alpha=0.85,
        capsize=4,
    )
    bars_c = ax.bar(
        x + width / 2,
        [cond_c.loc[m, "mean_CDS"] if m in cond_c.index else 0 for m in model_names],
        width,
        yerr=[cond_c.loc[m, "sem_CDS"] if m in cond_c.index else 0 for m in model_names],
        label="Condition C (Independent Repetition)",
        color=CONDITION_COLORS["C"],
        alpha=0.85,
        capsize=4,
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Mean Confidence Drift Score (CDS = T5 − T1)", fontsize=11)
    ax.set_title(
        "Figure 4 — Self-Anchoring vs. Independent Repetition Control",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.3f}"))
    fig.tight_layout()
    _save_fig(fig, "fig4_selfanchoring_vs_control")


# ---------------------------------------------------------------------------
# Figure 5 — Domain × Model Heatmap
# ---------------------------------------------------------------------------

def make_figure5(data: dict) -> None:
    """
    Heatmap of mean CDS. Rows = models (3), Columns = domains (3).
    Color = CDS magnitude. Uses Condition B results.
    """
    models = sorted(data.keys())
    domains = ["factual", "technical", "openended"]

    matrix = np.full((len(models), len(domains)), float("nan"))

    for r_idx, model_key in enumerate(models):
        results_b = data[model_key].get("B", [])
        for c_idx, domain in enumerate(domains):
            cds_vals = [
                r["cds"] for r in results_b
                if r.get("domain") == domain and r.get("cds") is not None
            ]
            if cds_vals:
                matrix[r_idx, c_idx] = float(np.mean(cds_vals))

    if np.all(np.isnan(matrix)):
        print("  Skipping Fig 5 — no CDS data")
        return

    row_labels = [MODEL_DISPLAY.get(m, m) for m in models]
    col_labels = [DOMAIN_LABELS[d] for d in domains]

    # Symmetric color range centered at 0
    vabs = np.nanmax(np.abs(matrix))
    vabs = max(vabs, 0.01)

    fig, ax = plt.subplots(figsize=(6, max(3, len(models) * 1.2)))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap=cmap,
        center=0,
        vmin=-vabs,
        vmax=vabs,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean CDS (T5 − T1 confidence)", "shrink": 0.8},
    )
    ax.set_title(
        "Figure 5 — Calibration Drift by Model × Domain (Condition B)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Domain", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, "fig5_domain_model_heatmap")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from experiment results.",
    )
    parser.add_argument(
        "--include-dryrun",
        action="store_true",
        help="Include dry-run results in figures (default: exclude dry-run files)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  SACD Experiment — Figure Generation")
    print("=" * 70)

    data = load_all_results(RESULTS_RAW_DIR, exclude_dryrun=not args.include_dryrun)

    if not data:
        sys.exit(1)

    print(f"\nLoaded data for models: {sorted(data.keys())}")
    print(f"Saving figures to: {FIGURES_DIR}\n")

    print("Generating Figure 1 — Confidence Across Turns...")
    make_figure1(data)

    print("Generating Figure 2 — ECE Across Turns...")
    make_figure2(data)

    print("Generating Figure 3 — Reliability Diagrams...")
    make_figure3(data)

    print("Generating Figure 4 — Self-Anchoring vs. Control...")
    make_figure4(data)

    print("Generating Figure 5 — Domain × Model Heatmap...")
    make_figure5(data)

    print("\n  All figures saved to results/figures/")


if __name__ == "__main__":
    main()
