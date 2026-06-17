"""Milestone 5 — per-variant explanation layer.

Turns the raw model output into a structured, plain-language, *honesty-aware* explanation
for a single variant. Crucially, it doesn't just report a prediction — it reports **how much
to trust that prediction for this variant's category**, using the Milestone 3 per-category
reliability numbers. A confident-sounding answer on a missense variant (where the model is
near chance) is exactly the kind of thing this layer guards against.

Inputs it combines:
  * Evo 2 delta score (M2) -> a calibrated P(disruptive).
  * Per-category reliability (M3 honesty metrics) -> a trust tier + caveat.
  * ClinVar context + the Findlay experimental label (benchmark ground truth).

No GPU/network needed; operates on the variants we've already scored.
The output is a plain dict, so an LLM could optionally render it as prose — but we render
it deterministically here to stay free and reproducible.
"""

from __future__ import annotations

import functools
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gvep.analysis.dataset import load_scored_dataset
from gvep.config import METRICS_DIR

HONESTY_JSON = METRICS_DIR / "honesty_metrics.json"
P_HIGH, P_LOW = 0.60, 0.20  # calibrated-probability cutoffs for the call


@functools.lru_cache(maxsize=1)
def _context():
    """Load the scored dataset, a delta->P(LOF) calibrator, and per-category trust."""
    df = load_scored_dataset()
    clean = df[df["class"].isin(["FUNC", "LOF"])]
    # StandardScaler is essential: raw delta is ~1e-3, so without scaling the logistic
    # coefficient stays tiny and predictions collapse to the base rate.
    calibrator = make_pipeline(StandardScaler(), LogisticRegression()).fit(
        (-clean["delta"]).to_numpy().reshape(-1, 1), clean["is_lof"].to_numpy()
    )
    trust = {}
    if HONESTY_JSON.exists():
        H = json.loads(HONESTY_JSON.read_text())
        trust = {r["consequence"]: r for r in H.get("by_consequence", [])}
    return df, calibrator, trust


def _trust_tier(auroc: float | None) -> tuple[str, str]:
    """Map a category's benchmark AUROC to a trust tier + plain-language caveat."""
    if auroc is None:
        return ("not assessable",
                "this category had too few of one class in the benchmark to estimate "
                "reliability — treat any prediction here as unvalidated")
    if auroc >= 0.80:
        return ("moderate–high",
                f"the model separates this category reasonably well (benchmark AUROC {auroc:.2f})")
    if auroc >= 0.70:
        return ("moderate", f"the model has moderate skill here (benchmark AUROC {auroc:.2f})")
    if auroc >= 0.60:
        return ("low",
                f"the model is weak here (benchmark AUROC {auroc:.2f} — not far above chance); "
                "do not rely on this prediction")
    return ("very low",
            f"the model is essentially at chance here (benchmark AUROC {auroc:.2f}); "
            "this prediction is not trustworthy")


def explain_variant(pos: int, ref: str, alt: str) -> dict | None:
    """Return a structured explanation dict for a variant, or None if not in our set."""
    df, calibrator, trust = _context()
    row = df[(df["pos"] == int(pos)) & (df["ref"] == ref.upper()) & (df["alt"] == alt.upper())]
    if row.empty:
        return None
    r = row.iloc[0]

    p = float(calibrator.predict_proba(np.array([[-r["delta"]]]))[0, 1])
    call = ("LIKELY DISRUPTIVE (pathogenic-leaning)" if p >= P_HIGH
            else "LIKELY TOLERATED (benign-leaning)" if p <= P_LOW
            else "UNCERTAIN")
    cons = r["consequence"]
    t = trust.get(cons, {})
    tier, caveat = _trust_tier(t.get("auroc"))
    prot = r.get("protein_variant")
    clinvar = r.get("clinvar")

    return {
        "variant": f"chr17:{int(r['pos'])} {r['ref']}>{r['alt']}",
        "gene": "BRCA1",
        "consequence": cons,
        "protein_variant": prot if pd.notna(prot) else None,
        "evo2_delta": round(float(r["delta"]), 6),
        "prob_disruptive": round(p, 3),
        "prediction": call,
        "trust_tier": tier,
        "trust_caveat": caveat,
        "category_auroc": t.get("auroc"),
        "category_fpr_at_90_sens": t.get("fpr"),
        "clinvar": clinvar if pd.notna(clinvar) else "not in ClinVar slice",
        "findlay_experimental_class": r["class"],
    }


def format_explanation(e: dict) -> str:
    """Render the structured explanation as a readable, honest text block."""
    prot = f", {e['protein_variant']}" if e["protein_variant"] else ""
    warn = "⚠ " if e["trust_tier"] in ("low", "very low", "not assessable") else ""
    fpr = e["category_fpr_at_90_sens"]
    fpr_txt = (f" Category false-alarm rate at high sensitivity is {fpr:.0%}."
               if isinstance(fpr, (int, float)) else "")
    lines = [
        "─" * 66,
        f" VARIANT  {e['variant']}   ({e['gene']}, {e['consequence']}{prot})",
        "─" * 66,
        f" Evo 2 zero-shot delta   : {e['evo2_delta']:+.6f}",
        f" Estimated P(disruptive) : {e['prob_disruptive']:.0%}",
        f" PREDICTION              : {e['prediction']}",
        "",
        f" {warn}CONFIDENCE: {e['trust_tier'].upper()}",
        f"   This is a {e['consequence'].upper()} variant — {e['trust_caveat']}.{fpr_txt}",
        "",
        f" ClinVar                 : {e['clinvar']}",
        f" Findlay experiment      : {e['findlay_experimental_class']}  (benchmark ground truth)",
        "",
        " NOTE: research/triage prototype — NOT a clinical diagnostic.",
        "─" * 66,
    ]
    return "\n".join(lines)


def prioritize(records: list[tuple[int, str, str]] | None = None, top: int = 25) -> list[dict]:
    """Rank variants by predicted disruptiveness — the VUS-triage view.

    With no `records`, ranks the real Variants of Uncertain Significance from ClinVar
    that we also have Evo 2 scores for (the actual triage use case). Returns the top-N,
    each with its explanation fields, sorted most-disruptive first.
    """
    df, _, _ = _context()
    if records is None:
        sub = df[df["clinvar"].astype(str).str.contains("ncertain", na=False)]
        keys = list(zip(sub["pos"], sub["ref"], sub["alt"]))
    else:
        keys = [(int(p), r, a) for p, r, a in records]

    out = []
    for pos, ref, alt in keys:
        e = explain_variant(pos, ref, alt)
        if e:
            out.append(e)
    out.sort(key=lambda e: e["prob_disruptive"], reverse=True)
    return out[:top]


def _demo_variants(df) -> list[tuple[int, str, str]]:
    """Pick a few variants that showcase the range of trust tiers."""
    picks = []
    # a missense the model gets WRONG (truth LOF, low score) -> low-trust caveat
    mis = df[(df["consequence"] == "Missense") & (df["class"] == "LOF")].sort_values("delta")
    if len(mis):
        r = mis.iloc[len(mis) // 2]; picks.append((int(r.pos), r.ref, r.alt))
    # a splice-region LOF (higher trust category)
    sp = df[(df["consequence"] == "Splice region") & (df["class"] == "LOF")].sort_values("delta")
    if len(sp):
        r = sp.iloc[0]; picks.append((int(r.pos), r.ref, r.alt))
    # a benign synonymous
    syn = df[(df["consequence"] == "Synonymous") & (df["class"] == "FUNC")]
    if len(syn):
        r = syn.iloc[0]; picks.append((int(r.pos), r.ref, r.alt))
    return picks


def run_demo() -> None:
    """Print explanations for a few illustrative variants and save them."""
    df, _, _ = _context()
    from gvep.config import RESULTS_DIR

    blocks = []
    print("\n  MILESTONE 5 — per-variant explanations (illustrative)\n")
    for pos, ref, alt in _demo_variants(df):
        e = explain_variant(pos, ref, alt)
        if e:
            block = format_explanation(e)
            print(block + "\n")
            blocks.append(block)
    (RESULTS_DIR / "example_explanations.txt").write_text("\n\n".join(blocks))
    print(f"  saved -> {RESULTS_DIR / 'example_explanations.txt'}")
