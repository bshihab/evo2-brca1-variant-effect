"""Milestone 7+ — a hybrid Evo 2 + AlphaMissense ensemble (routing by coverage).

Motivated directly by our own findings:
  * AlphaMissense is excellent on MISSENSE (AUROC 0.90) but scores ONLY missense.
  * Evo 2 covers EVERY variant type (best on splice, 0.85) but is weak on missense (0.61).

So we route each variant to the tool that's good at it:
    hybrid = AlphaMissense   if the variant is missense (and has an AM score)
             Evo 2 (Δ)       otherwise (non-coding / splice / etc.)

To make the two scores comparable across the routing boundary, each is mapped to a
**probability of LOF** via a StandardScaler+logistic calibrator, evaluated with 5-fold
cross-validation (cross_val_predict) so the comparison is leakage-free. We then compare,
on the FULL benchmark (all variant types):
    Evo 2 alone   vs   AlphaMissense alone (coverage-limited)   vs   the hybrid.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gvep.analysis.dataset import load_scored_dataset
from gvep.config import FIGURES_DIR, METRICS_DIR, RAW_DIR
from gvep.utils.seed import set_seed

AM_TSV = RAW_DIR / "alphamissense_brca1.tsv"


def _cv_proba(score: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cross-validated P(LOF) from a 1-D score (higher score = more pathogenic)."""
    cal = make_pipeline(StandardScaler(), LogisticRegression())
    return cross_val_predict(cal, score.reshape(-1, 1), y, cv=5, method="predict_proba")[:, 1]


def run() -> dict:
    set_seed()
    if not AM_TSV.exists():
        raise SystemExit(f"Missing {AM_TSV}; run the AlphaMissense data prep first.")
    am = pd.read_csv(AM_TSV, sep="\t")
    am["ref"], am["alt"] = am["ref"].str.upper(), am["alt"].str.upper()

    df = load_scored_dataset()
    df = df[df["class"].isin(["FUNC", "LOF"])].copy()
    df = df.merge(am[["pos", "ref", "alt", "am_pathogenicity"]],
                  on=["pos", "ref", "alt"], how="left")
    y = (df["class"] == "LOF").to_numpy().astype(int)

    is_missense = (df["consequence"] == "Missense").to_numpy()
    has_am = df["am_pathogenicity"].notna().to_numpy()
    route_am = is_missense & has_am  # variants we send to AlphaMissense

    # --- calibrated P(LOF) for each tool (CV, leakage-free) ---
    p_evo = _cv_proba(-df["delta"].to_numpy(), y)  # Evo 2 on all variants
    p_am = np.full(len(df), np.nan)
    p_am[has_am] = _cv_proba(df.loc[has_am, "am_pathogenicity"].to_numpy(), y[has_am])

    # --- hybrid: route missense->AM, everything else->Evo 2 ---
    p_hybrid = p_evo.copy()
    p_hybrid[route_am] = p_am[route_am]

    # --- AlphaMissense ALONE on the full set: it can't score non-missense, so it must
    #     fall back to "no information" (base rate) there — exposing its coverage gap.
    p_am_only = np.full(len(df), y.mean())
    p_am_only[route_am] = p_am[route_am]

    M = {
        "n": int(len(df)),
        "n_missense_routed_to_AM": int(route_am.sum()),
        "am_coverage": float(route_am.mean()),
        "full_set": {
            "Evo 2 alone": float(roc_auc_score(y, p_evo)),
            "AlphaMissense alone (missense-only coverage)": float(roc_auc_score(y, p_am_only)),
            "Hybrid (Evo 2 + AlphaMissense)": float(roc_auc_score(y, p_hybrid)),
        },
        "missense_only": {
            "Evo 2": float(roc_auc_score(y[route_am], p_evo[route_am])),
            "AlphaMissense": float(roc_auc_score(y[route_am], p_am[route_am])),
        },
        "non_missense_only": {
            "Evo 2 (= hybrid here)": float(roc_auc_score(y[~route_am], p_evo[~route_am])),
        },
    }

    print("\n" + "=" * 66)
    print("  HYBRID ENSEMBLE — Evo 2 + AlphaMissense (routing by variant type)")
    print("=" * 66)
    print(f"  full set n={M['n']:,}  ·  {M['n_missense_routed_to_AM']:,} missense routed to "
          f"AlphaMissense ({M['am_coverage']:.0%}); the rest use Evo 2")
    print("\n  FULL-SET AUROC (all variant types):")
    for k, v in M["full_set"].items():
        print(f"    {k:<46}{v:.3f}")
    print("\n  why it works (per segment):")
    print(f"    missense:     Evo 2 {M['missense_only']['Evo 2']:.3f}  ->  "
          f"AlphaMissense {M['missense_only']['AlphaMissense']:.3f}  (the upgrade)")
    print(f"    non-missense: Evo 2 {M['non_missense_only']['Evo 2 (= hybrid here)']:.3f}  "
          "(AlphaMissense can't score these at all)")

    _plot(M)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "ensemble.json").write_text(json.dumps(M, indent=2))
    print(f"\n  metrics -> {METRICS_DIR / 'ensemble.json'}")
    print(f"  figure  -> {FIGURES_DIR / 'm7_ensemble.png'}")
    print("=" * 66 + "\n")
    return M


def _plot(M: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    fs = M["full_set"]
    labels = ["Evo 2\nalone", "AlphaMissense\nalone", "Hybrid"]
    vals = list(fs.values())
    colors = ["#e76f51", "#8d99ae", "#2a9d8f"]
    ax1.bar(labels, vals, color=colors)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_ylabel("AUROC (full benchmark, all variant types)")
    ax1.set_title("Full set: the hybrid beats either tool alone")
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=10)

    seg = ["missense", "non-missense"]
    evo = [M["missense_only"]["Evo 2"], M["non_missense_only"]["Evo 2 (= hybrid here)"]]
    hyb = [M["missense_only"]["AlphaMissense"], M["non_missense_only"]["Evo 2 (= hybrid here)"]]
    x = np.arange(2)
    ax2.bar(x - 0.2, evo, 0.4, label="Evo 2", color="#e76f51")
    ax2.bar(x + 0.2, hyb, 0.4, label="Hybrid (routed)", color="#2a9d8f")
    ax2.set_xticks(x); ax2.set_xticklabels(seg); ax2.set_ylim(0.5, 1.0)
    ax2.set_title("Why: missense upgraded; non-missense kept"); ax2.legend()

    fig.suptitle("Hybrid Evo 2 + AlphaMissense ensemble", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "m7_ensemble.png", dpi=130); plt.close(fig)
