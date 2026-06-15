"""Milestone 2 sanity check: do disruptive variants get more negative deltas?

This is the first reality check on the scoring engine. If the method works at all, the
LOF (loss-of-function) variants should sit at MORE NEGATIVE delta values than the FUNC
(functional) ones — i.e. the model finds disruptive variants less plausible. We plot the
two distributions and compute a quick AUROC. (The full, honest evaluation is Milestone 3;
this is just "is it pointing the right way?")

Runs locally on CPU after the Modal scoring job writes data/cache/evo2_delta_scores.csv.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write files, don't open windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from gvep.config import CACHE_DIR, FIGURES_DIR

SCORES_CSV = CACHE_DIR / "evo2_delta_scores.csv"
FIG_OUT = FIGURES_DIR / "m2_delta_distributions.png"


def run() -> None:
    if not SCORES_CSV.exists():
        raise SystemExit(
            f"No scores found at {SCORES_CSV}.\n"
            "Run the Modal scoring job first:  modal run -m gvep.scoring.modal_app::main"
        )

    df = pd.read_csv(SCORES_CSV)
    df = df.dropna(subset=["delta"])

    # --- quick separation metric ---
    # Predictor is -delta (more positive = more likely LOF). Two framings:
    #   (a) LOF vs everything else; (b) the clean LOF vs FUNC contrast (drop INT).
    y_all = (df["class"] == "LOF").astype(int)
    auroc_all = roc_auc_score(y_all, -df["delta"])

    clean = df[df["class"].isin(["LOF", "FUNC"])]
    auroc_clean = roc_auc_score((clean["class"] == "LOF").astype(int), -clean["delta"])

    print("\n" + "=" * 60)
    print("  MILESTONE 2 SANITY CHECK — Evo 2 delta scores")
    print("=" * 60)
    print(f"  scored variants: {len(df):,}")
    print("\n  median delta by class (more negative = more disruptive):")
    for cls in ("FUNC", "INT", "LOF"):
        sub = df[df["class"] == cls]["delta"]
        if len(sub):
            print(f"    {cls:4s}: median={sub.median():+.4f}  mean={sub.mean():+.4f}  n={len(sub):,}")
    print(f"\n  quick AUROC (LOF vs rest):     {auroc_all:.3f}")
    print(f"  quick AUROC (LOF vs FUNC only): {auroc_clean:.3f}")
    print(f"  (published Evo 2 1B reference: ~0.73)")
    direction = "✅ correct" if df[df['class']=='LOF']['delta'].median() < \
        df[df['class']=='FUNC']['delta'].median() else "❌ BACKWARDS"
    print(f"\n  direction check: LOF more negative than FUNC?  {direction}")
    print("=" * 60)

    # --- plot ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    order = ["FUNC", "INT", "LOF"]
    palette = {"FUNC": "#2a9d8f", "INT": "#e9c46a", "LOF": "#e76f51"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for cls in order:
        sub = df[df["class"] == cls]["delta"]
        if len(sub):
            sns.kdeplot(sub, ax=ax1, label=f"{cls} (n={len(sub):,})",
                        color=palette[cls], fill=True, alpha=0.25, linewidth=2)
    ax1.set_title("Evo 2 delta-likelihood by functional class")
    ax1.set_xlabel("delta = var_logL - ref_logL  (more negative = more disruptive)")
    ax1.axvline(0, color="gray", ls="--", lw=1)
    ax1.legend()

    sns.boxplot(data=df[df["class"].isin(order)], x="class", y="delta",
                order=order, palette=palette, ax=ax2)
    ax2.set_title(f"AUROC LOF vs FUNC = {auroc_clean:.3f}  (ref ~0.73)")
    ax2.axhline(0, color="gray", ls="--", lw=1)

    fig.suptitle("Milestone 2 sanity check — BRCA1 zero-shot scoring", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=130)
    print(f"\n  saved plot: {FIG_OUT}\n")
