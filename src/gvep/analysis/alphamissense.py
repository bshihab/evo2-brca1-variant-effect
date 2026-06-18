"""Milestone 7 (stretch) — benchmark Evo 2 against AlphaMissense.

AlphaMissense (Google DeepMind, 2023) is a state-of-the-art **missense** pathogenicity
predictor. Here we add it to the Milestone 3 benchmark on the BRCA1 missense variants — the
category where the Evo 2 1B zero-shot score was weakest (~0.60). AlphaMissense only covers
missense variants, so this comparison is missense-only.

Data: AlphaMissense hg19 (Zenodo 8360242), filtered to the BRCA1 region. See
docs/ACCESS_PATH.md / the data prep step. No GPU needed.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from gvep.analysis.dataset import load_scored_dataset
from gvep.config import FIGURES_DIR, METRICS_DIR, RAW_DIR

AM_TSV = RAW_DIR / "alphamissense_brca1.tsv"

# predictor column -> sign that makes "higher = more pathogenic"
OTHER = {"CADD": ("CADD.score", 1), "phyloP": ("phylop", 1),
         "SIFT": ("sift", -1), "PolyPhen-2": ("polyphen2", 1)}


def run() -> dict:
    if not AM_TSV.exists():
        raise SystemExit(f"Missing {AM_TSV}. Run the AlphaMissense data prep first "
                         "(download + filter, see the M7 step).")
    am = pd.read_csv(AM_TSV, sep="\t")
    am["ref"] = am["ref"].astype(str).str.upper()
    am["alt"] = am["alt"].astype(str).str.upper()

    df = load_scored_dataset()
    mis = df[(df["consequence"] == "Missense") & (df["class"].isin(["FUNC", "LOF"]))].copy()
    merged = mis.merge(am[["pos", "ref", "alt", "am_pathogenicity"]],
                       on=["pos", "ref", "alt"], how="left")
    sub = merged.dropna(subset=["am_pathogenicity"]).copy()
    y = (sub["class"] == "LOF").astype(int).to_numpy()

    print("\n" + "=" * 64)
    print("  MILESTONE 7 — Evo 2 vs AlphaMissense (BRCA1 missense)")
    print("=" * 64)
    print(f"  missense variants with AlphaMissense scores: {len(sub):,} "
          f"(of {len(merged):,} missense)")
    print(f"  LOF in this subset: {int(y.sum())} ({y.mean():.0%})\n")

    results = {
        "Evo 2 1B (zero-shot)": float(roc_auc_score(y, -sub["delta"])),
        "AlphaMissense": float(roc_auc_score(y, sub["am_pathogenicity"])),
    }
    for name, (col, sign) in OTHER.items():
        s = sub.dropna(subset=[col])
        if len(s) > 20:
            results[name] = float(roc_auc_score(
                (s["class"] == "LOF").astype(int), sign * s[col]))

    ordered = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
    print(f"  {'predictor':<24}{'AUROC':>8}")
    for name, au in ordered:
        star = "  *" if name in ("AlphaMissense", "Evo 2 1B (zero-shot)") else ""
        print(f"  {name:<24}{au:>8.3f}{star}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "alphamissense_benchmark.json").write_text(
        json.dumps({"n": int(len(sub)), "auroc": results}, indent=2))
    _plot(ordered, len(sub))
    print(f"\n  metrics -> {METRICS_DIR / 'alphamissense_benchmark.json'}")
    print(f"  figure  -> {FIGURES_DIR / 'm7_alphamissense.png'}")
    print("=" * 64 + "\n")
    return results


def _plot(ordered, n) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names = [k for k, _ in ordered][::-1]
    vals = [v for _, v in ordered][::-1]
    colors = ["#e76f51" if k == "Evo 2 1B (zero-shot)"
              else "#2a9d8f" if k == "AlphaMissense" else "#8d99ae" for k in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.axvline(0.5, ls="--", color="gray")
    ax.set_xlim(0.4, 1.0)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_title(f"BRCA1 missense: Evo 2 vs AlphaMissense vs established tools (n={n:,})")
    ax.set_xlabel("AUROC (LOF vs FUNC)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "m7_alphamissense.png", dpi=130)
    plt.close(fig)
