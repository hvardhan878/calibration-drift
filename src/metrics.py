"""
src/metrics.py
Metric computation for the SACD experiment.

Metrics:
  - CDS  (Confidence Drift Score):     confidence_T5 - confidence_T1
  - ECE  (Expected Calibration Error):  standard binning-based calibration error
  - CDR  (Calibration Drift Rate):      slope of ECE across turns (linear regression)
"""

import logging
from typing import Optional
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence Drift Score (CDS)
# ---------------------------------------------------------------------------

def compute_cds(turns: list[dict]) -> Optional[float]:
    """
    CDS = confidence at turn 5 minus confidence at turn 1.
    Returns None if either value is missing (extraction failure).

    A positive CDS means confidence inflated across turns.
    A negative CDS means confidence deflated.
    """
    if len(turns) < 5:
        logger.debug(f"CDS: only {len(turns)} turns, need 5")
        return None

    c1 = turns[0].get("extracted_confidence")
    c5 = turns[4].get("extracted_confidence")

    if c1 is None or c5 is None:
        return None

    return float(c5 - c1)


def compute_cds_trajectory(turns: list[dict]) -> list[Optional[float]]:
    """
    CDS at each turn relative to T1.
    Returns list of (confidence_Ti - confidence_T1) for i in 1..N.
    """
    c1 = turns[0].get("extracted_confidence") if turns else None
    if c1 is None:
        return [None] * len(turns)

    result = []
    for turn in turns:
        ci = turn.get("extracted_confidence")
        result.append(float(ci - c1) if ci is not None else None)
    return result


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 10,
) -> float:
    """
    Standard ECE using uniform binning of confidence scores.

    ECE = Σ_b (|B_b| / n) × |mean_conf(B_b) − accuracy(B_b)|

    Lower is better. Perfect calibration = 0.0.
    Confidences and correctness must be the same length.

    Args:
        confidences: List of confidence values in [0, 1]
        correctness: List of booleans (True = correct)
        n_bins:      Number of uniform bins in [0, 1]

    Returns:
        ECE as a float in [0, 1]
    """
    if not confidences:
        return float("nan")

    confidences = np.array(confidences, dtype=float)
    correctness = np.array(correctness, dtype=float)
    n = len(confidences)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if not mask.any():
            continue

        bin_confs = confidences[mask]
        bin_corr = correctness[mask]
        weight = len(bin_confs) / n
        ece += weight * abs(bin_confs.mean() - bin_corr.mean())

    return float(ece)


def compute_ece_per_turn(results: list[dict]) -> list[float]:
    """
    Compute ECE at each turn across all questions.

    Args:
        results: List of question result dicts (each with 'turns' list)

    Returns:
        List of ECE values, one per turn. NaN for turns with no data.
    """
    # Determine max turns
    max_turns = max((len(r.get("turns", [])) for r in results), default=0)
    ece_by_turn = []

    for turn_idx in range(max_turns):
        confs = []
        corrs = []
        for result in results:
            turns = result.get("turns", [])
            if turn_idx < len(turns):
                turn = turns[turn_idx]
                conf = turn.get("extracted_confidence")
                corr = turn.get("correct")
                if conf is not None and corr is not None:
                    confs.append(conf)
                    corrs.append(bool(corr))

        if confs:
            ece_by_turn.append(compute_ece(confs, corrs))
        else:
            ece_by_turn.append(float("nan"))

    return ece_by_turn


# ---------------------------------------------------------------------------
# Calibration Drift Rate (CDR)
# ---------------------------------------------------------------------------

def compute_cdr(ece_per_turn: list[float]) -> float:
    """
    CDR = slope of linear regression of ECE values over turns.
    Positive slope = calibration is worsening across turns.
    Negative slope = calibration is improving.

    Ignores NaN values.

    Args:
        ece_per_turn: List of ECE values indexed by turn (0-based)

    Returns:
        Slope of the regression line, or NaN if insufficient data.
    """
    pairs = [
        (i + 1, e)
        for i, e in enumerate(ece_per_turn)
        if not np.isnan(e)
    ]
    if len(pairs) < 2:
        return float("nan")

    turns_arr, ece_arr = zip(*pairs)
    slope, _intercept, _r, _p, _se = stats.linregress(turns_arr, ece_arr)
    return float(slope)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def paired_ttest_confidence(
    results: list[dict],
    turn1_idx: int = 0,
    turn2_idx: int = 4,
) -> dict:
    """
    Paired t-test: T1 confidence vs T5 confidence across all questions.
    Tests H1: confidence inflates across turns.

    Returns dict with: t_stat, p_value, cohen_d, n
    """
    t1_vals, t5_vals = [], []
    for result in results:
        turns = result.get("turns", [])
        if len(turns) > max(turn1_idx, turn2_idx):
            c1 = turns[turn1_idx].get("extracted_confidence")
            c5 = turns[turn2_idx].get("extracted_confidence")
            if c1 is not None and c5 is not None:
                t1_vals.append(c1)
                t5_vals.append(c5)

    if len(t1_vals) < 2:
        return {"t_stat": float("nan"), "p_value": float("nan"), "cohen_d": float("nan"), "n": 0}

    t1_arr = np.array(t1_vals)
    t5_arr = np.array(t5_vals)
    diffs = t5_arr - t1_arr

    t_stat, p_value = stats.ttest_1samp(diffs, popmean=0.0)
    cohen_d = diffs.mean() / diffs.std(ddof=1) if diffs.std(ddof=1) > 0 else 0.0

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohen_d": float(cohen_d),
        "n": len(diffs),
        "mean_drift": float(diffs.mean()),
    }


def mannwhitney_cds(
    results_b: list[dict],
    results_c: list[dict],
) -> dict:
    """
    Mann-Whitney U test: CDS in Condition B vs Condition C.
    Tests H2: self-anchoring causes more drift than repetition alone.

    Returns dict with: u_stat, p_value, rank_biserial_r, n_b, n_c
    """
    cds_b = [r["cds"] for r in results_b if r.get("cds") is not None]
    cds_c = [r["cds"] for r in results_c if r.get("cds") is not None]

    if len(cds_b) < 2 or len(cds_c) < 2:
        return {
            "u_stat": float("nan"), "p_value": float("nan"),
            "rank_biserial_r": float("nan"), "n_b": len(cds_b), "n_c": len(cds_c),
        }

    u_stat, p_value = stats.mannwhitneyu(cds_b, cds_c, alternative="two-sided")
    # Rank-biserial correlation as effect size
    n_b, n_c = len(cds_b), len(cds_c)
    rank_biserial_r = 1 - (2 * u_stat) / (n_b * n_c)

    return {
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "rank_biserial_r": float(rank_biserial_r),
        "n_b": n_b,
        "n_c": n_c,
        "mean_cds_b": float(np.mean(cds_b)),
        "mean_cds_c": float(np.mean(cds_c)),
    }


def anova_cds_by_domain(results: list[dict]) -> dict:
    """
    One-way ANOVA: CDS across domains (factual, technical, openended).
    Tests H3: domain moderates calibration drift.

    Returns dict with: f_stat, p_value, eta_squared, groups (mean CDS per domain)
    """
    domain_groups: dict[str, list[float]] = {}
    for result in results:
        domain = result.get("domain", "unknown")
        cds = result.get("cds")
        if cds is not None:
            domain_groups.setdefault(domain, []).append(cds)

    groups = list(domain_groups.values())
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return {
            "f_stat": float("nan"), "p_value": float("nan"),
            "eta_squared": float("nan"), "groups": {},
        }

    f_stat, p_value = stats.f_oneway(*groups)

    # Eta-squared effect size
    all_vals = [v for g in groups for v in g]
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    return {
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "eta_squared": float(eta_sq),
        "groups": {
            domain: {
                "mean_cds": float(np.mean(vals)),
                "std_cds": float(np.std(vals, ddof=1)),
                "n": len(vals),
            }
            for domain, vals in domain_groups.items()
        },
    }


# ---------------------------------------------------------------------------
# Aggregate summary helpers
# ---------------------------------------------------------------------------

def summarize_results(results: list[dict]) -> dict:
    """
    Compute a full summary dict for a list of question results.
    Used by compute_metrics.py.
    """
    all_cds = [r["cds"] for r in results if r.get("cds") is not None]
    ece_per_turn = compute_ece_per_turn(results)

    # T1 / T5 ECE
    ece_t1 = ece_per_turn[0] if len(ece_per_turn) > 0 else float("nan")
    ece_t5 = ece_per_turn[4] if len(ece_per_turn) > 4 else float("nan")
    delta_ece = (ece_t5 - ece_t1) if not (np.isnan(ece_t1) or np.isnan(ece_t5)) else float("nan")

    return {
        "n": len(results),
        "n_with_cds": len(all_cds),
        "mean_cds": float(np.mean(all_cds)) if all_cds else float("nan"),
        "std_cds": float(np.std(all_cds, ddof=1)) if len(all_cds) > 1 else float("nan"),
        "ece_t1": ece_t1,
        "ece_t5": ece_t5,
        "delta_ece": delta_ece,
        "cdr": compute_cdr(ece_per_turn),
        "ece_per_turn": ece_per_turn,
    }
