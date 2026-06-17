"""Milestone 3 — Validation & the HONESTY layer (the project's centerpiece).

Goes beyond a single headline AUROC to ask the questions that decide whether a tool
like this is trustworthy:
  1. Headline metrics: AUROC + AUPRC (with bootstrap confidence intervals).
  2. Stratified performance by variant category (consequence, coding/non-coding, SNV type).
  3. False-positive rate per category at a fixed, clinically-motivated sensitivity.
  4. Calibration: are scores meaningful as probabilities? (reliability diagram, Brier).
  5. Severity failure mode: does it do worse on MILD loss-of-function than SEVERE?
  6. Class-imbalance honesty: how a good AUROC can hide poor precision on the rare class.
  7. Benchmark vs established tools (CADD, phyloP, SIFT, PolyPhen-2).

Everything runs locally on the scores we already computed. Figures -> results/figures/,
numbers -> results/metrics/honesty_metrics.json, and a summary is printed for RESULTS.md.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict

from gvep.analysis.dataset import PREDICTORS, load_scored_dataset
from gvep.config import FIGURES_DIR, METRICS_DIR
from gvep.utils.seed import set_seed

# Clinically-motivated operating point: a triage tool should catch most harmful
# variants, so we fix sensitivity (recall of LOF) high and look at the cost (false alarms).
TARGET_SENSITIVITY = 0.90
MIN_PER_CLASS = 8  # don't report an AUROC for a stratum with fewer than this per class


# ---------------------------------------------------------------------------
# small metric helpers
# ---------------------------------------------------------------------------
def _oriented(df: pd.DataFrame, col: str, sign: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_is_lof, score) with NaNs dropped and score oriented so higher = LOF."""
    sub = df[["is_lof", col]].dropna()
    return sub["is_lof"].to_numpy(), sign * sub[col].to_numpy()


def _auroc_ci(y: np.ndarray, s: np.ndarray, n_boot: int = 1000) -> tuple[float, float, float]:
    """AUROC with a 95% bootstrap confidence interval."""
    rng = np.random.default_rng(0)
    auroc = roc_auc_score(y, s)
    boots = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        if y[b].sum() == 0 or y[b].sum() == len(b):  # need both classes
            continue
        boots.append(roc_auc_score(y[b], s[b]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auroc), float(lo), float(hi)


# ---------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------
def run() -> dict:
    set_seed()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_scored_dataset()
    M: dict = {}

    # "clean" = the unambiguous separation problem (drop the intermediate class).
    clean = df[df["class"].isin(["FUNC", "LOF"])].copy()
    y_clean = clean["is_lof"].to_numpy()
    s_clean = -clean["delta"].to_numpy()  # higher = more likely LOF

    print("\n" + "=" * 70)
    print("  MILESTONE 3 — VALIDATION & HONESTY LAYER")
    print("=" * 70)

    # --- 1. Headline metrics --------------------------------------------------
    auroc, lo, hi = _auroc_ci(y_clean, s_clean)
    auprc = average_precision_score(y_clean, s_clean)
    prevalence = y_clean.mean()
    # LOF vs everything-else (FUNC+INT) — the messier, more realistic framing
    y_all = df["is_lof"].to_numpy()
    s_all = -df["delta"].to_numpy()
    auroc_all = roc_auc_score(y_all, s_all)
    M["headline"] = {
        "auroc_lof_vs_func": auroc, "auroc_ci": [lo, hi],
        "auprc_lof_vs_func": float(auprc), "auprc_baseline": float(prevalence),
        "auroc_lof_vs_rest": float(auroc_all), "n_clean": int(len(clean)),
    }
    print(f"\n[1] HEADLINE (LOF vs FUNC, n={len(clean):,})")
    print(f"    AUROC = {auroc:.3f}  (95% CI {lo:.3f}–{hi:.3f})")
    print(f"    AUPRC = {auprc:.3f}  (baseline = prevalence = {prevalence:.3f})")
    print(f"    AUROC (LOF vs FUNC+INT) = {auroc_all:.3f}")

    # --- 2/3. Stratified AUROC + false-positive rate by consequence ----------
    # Operating threshold: the score cutoff that yields TARGET_SENSITIVITY overall.
    fpr_curve, tpr_curve, thr = roc_curve(y_clean, s_clean)
    k = int(np.argmin(np.abs(tpr_curve - TARGET_SENSITIVITY)))
    threshold = thr[k]
    overall_fpr = fpr_curve[k]
    pred_pos = s_clean >= threshold
    precision_at = (y_clean[pred_pos].sum() / max(pred_pos.sum(), 1))
    M["operating_point"] = {
        "target_sensitivity": TARGET_SENSITIVITY, "threshold": float(threshold),
        "overall_fpr": float(overall_fpr), "precision": float(precision_at),
    }
    print(f"\n[2/3] STRATIFIED by consequence  (at {TARGET_SENSITIVITY:.0%} sensitivity)")
    print(f"    overall FPR={overall_fpr:.1%}, precision={precision_at:.1%}")
    print(f"    {'consequence':<17}{'n':>6}{'%LOF':>7}{'AUROC':>8}{'FPR':>8}")
    strat = []
    for cons, g in clean.groupby("consequence"):
        n, n_lof, n_func = len(g), int(g["is_lof"].sum()), int((g["is_lof"] == 0).sum())
        yg, sg = g["is_lof"].to_numpy(), -g["delta"].to_numpy()
        au = (roc_auc_score(yg, sg)
              if n_lof >= MIN_PER_CLASS and n_func >= MIN_PER_CLASS else np.nan)
        func = g[g["is_lof"] == 0]
        fpr = float((-func["delta"] >= threshold).mean()) if len(func) else np.nan
        strat.append({"consequence": cons, "n": n, "pct_lof": n_lof / n,
                      "auroc": None if np.isnan(au) else float(au),
                      "fpr": None if np.isnan(fpr) else fpr})
        au_s = "  n/a" if np.isnan(au) else f"{au:.3f}"
        fpr_s = "  n/a" if np.isnan(fpr) else f"{fpr:.1%}"
        print(f"    {cons:<17}{n:>6}{n_lof / n:>6.0%}{au_s:>8}{fpr_s:>8}")
    M["by_consequence"] = strat

    # by region (coding vs non-coding) and SNV type
    M["by_region"], M["by_snv"] = {}, {}
    for key, col in [("by_region", "region"), ("by_snv", "snv_type")]:
        for val, g in clean.groupby(col):
            yg, sg = g["is_lof"].to_numpy(), -g["delta"].to_numpy()
            if yg.sum() >= MIN_PER_CLASS and (yg == 0).sum() >= MIN_PER_CLASS:
                M[key][val] = {"n": int(len(g)), "auroc": float(roc_auc_score(yg, sg))}

    # --- 4. Calibration (cross-validated logistic -> reliability) ------------
    lr = LogisticRegression()
    p = cross_val_predict(lr, s_clean.reshape(-1, 1), y_clean, cv=5, method="predict_proba")[:, 1]
    brier = brier_score_loss(y_clean, p)
    M["calibration"] = {"brier": float(brier)}
    print(f"\n[4] CALIBRATION (after CV logistic recalibration): Brier={brier:.3f}")

    # --- 5. Severity failure mode -------------------------------------------
    func = df[df["class"] == "FUNC"]
    print("\n[5] SEVERITY failure mode (separating LOF from FUNC):")
    sev_res = {}
    for level in ("severe", "mild"):
        lof_lvl = df[df["severity"] == level]
        yy = np.r_[np.ones(len(lof_lvl)), np.zeros(len(func))]
        ss = np.r_[-lof_lvl["delta"].to_numpy(), -func["delta"].to_numpy()]
        au = float(roc_auc_score(yy, ss))
        sev_res[level] = {"n_lof": int(len(lof_lvl)), "auroc": au}
        print(f"    {level:<7} LOF (n={len(lof_lvl):>3}) vs FUNC:  AUROC = {au:.3f}")
    M["severity"] = sev_res
    print(f"    -> degradation (severe - mild) = "
          f"{sev_res['severe']['auroc'] - sev_res['mild']['auroc']:+.3f}")

    # --- 6. Class-imbalance honesty -----------------------------------------
    print("\n[6] CLASS-IMBALANCE honesty:")
    print(f"    LOF prevalence (vs FUNC) = {prevalence:.1%}  -> AUPRC baseline {prevalence:.3f}")
    print(f"    AUROC {auroc:.3f} looks strong, but at {TARGET_SENSITIVITY:.0%} sensitivity")
    print(f"    precision is only {precision_at:.1%} -> ~{(1-precision_at)/precision_at:.1f} "
          "false alarms per true LOF caught.")

    # --- 7. Benchmark vs established predictors ------------------------------
    print("\n[7] BENCHMARK vs established tools (AUROC, LOF vs FUNC):")
    bench_all, bench_mis = {}, {}
    missense = clean[clean["consequence"] == "Missense"]
    for name, (col, sign) in PREDICTORS.items():
        y, s = _oriented(clean, col, sign)
        if y.sum() >= MIN_PER_CLASS and (y == 0).sum() >= MIN_PER_CLASS:
            bench_all[name] = float(roc_auc_score(y, s))
        ym, sm = _oriented(missense, col, sign)
        if ym.sum() >= MIN_PER_CLASS and (ym == 0).sum() >= MIN_PER_CLASS:
            bench_mis[name] = float(roc_auc_score(ym, sm))
    M["benchmark_all"], M["benchmark_missense"] = bench_all, bench_mis
    print(f"    {'predictor':<22}{'all variants':>14}{'missense only':>15}")
    for name in PREDICTORS:
        a = f"{bench_all[name]:.3f}" if name in bench_all else "  n/a"
        m = f"{bench_mis[name]:.3f}" if name in bench_mis else "  n/a"
        print(f"    {name:<22}{a:>14}{m:>15}")

    _make_figures(df, clean, y_clean, s_clean, p, threshold, strat, bench_all, bench_mis)

    (METRICS_DIR / "honesty_metrics.json").write_text(json.dumps(M, indent=2))
    print(f"\n  metrics -> {METRICS_DIR / 'honesty_metrics.json'}")
    print(f"  figures -> {FIGURES_DIR}/m3_*.png")
    print("=" * 70 + "\n")
    return M


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _make_figures(df, clean, y, s, p, threshold, strat, bench_all, bench_mis) -> None:
    # ROC + PR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y, s)
    ax1.plot(fpr, tpr, lw=2, color="#264653")
    ax1.plot([0, 1], [0, 1], "--", color="gray")
    ax1.set(title=f"ROC (AUROC={roc_auc_score(y, s):.3f})",
            xlabel="false positive rate", ylabel="true positive rate (sensitivity)")
    prec, rec, _ = precision_recall_curve(y, s)
    ax2.plot(rec, prec, lw=2, color="#e76f51")
    ax2.axhline(y.mean(), ls="--", color="gray", label=f"baseline={y.mean():.2f}")
    ax2.set(title=f"Precision–Recall (AUPRC={average_precision_score(y, s):.3f})",
            xlabel="recall (sensitivity)", ylabel="precision")
    ax2.legend()
    fig.suptitle("M3 — headline curves (LOF vs FUNC)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "m3_roc_pr.png", dpi=130); plt.close(fig)

    # per-consequence AUROC + FPR
    sd = pd.DataFrame(strat).sort_values("n", ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    labels = [f"{c}\n(n={n}, {p:.0%} LOF)" for c, n, p in
              zip(sd["consequence"], sd["n"], sd["pct_lof"])]
    ax1.bar(range(len(sd)), [a if a is not None else 0 for a in sd["auroc"]],
            color=["#2a9d8f" if a is not None else "#cccccc" for a in sd["auroc"]])
    ax1.axhline(0.5, ls="--", color="gray"); ax1.set_ylim(0, 1)
    ax1.set_xticks(range(len(sd))); ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax1.set(title="AUROC by consequence (grey = not computable)", ylabel="AUROC")
    ax2.bar(range(len(sd)), [f if f is not None else 0 for f in sd["fpr"]], color="#e76f51")
    ax2.set_xticks(range(len(sd))); ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax2.set(title="False-positive rate by consequence @90% sensitivity", ylabel="FPR")
    fig.suptitle("M3 — performance is NOT uniform across variant types", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "m3_by_consequence.png", dpi=130); plt.close(fig)

    # calibration reliability
    fig, ax = plt.subplots(figsize=(6, 6))
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p, bins) - 1
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#264653", label="Evo 2 (recalibrated)")
    ax.set(title=f"Calibration (Brier={brier_score_loss(y, p):.3f})",
           xlabel="predicted P(LOF)", ylabel="observed LOF fraction")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "m3_calibration.png", dpi=130); plt.close(fig)

    # severity + benchmark
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    func = df[df["class"] == "FUNC"]
    for level, color in [("severe", "#9b2226"), ("mild", "#e9c46a")]:
        lof = df[df["severity"] == level]
        yy = np.r_[np.ones(len(lof)), np.zeros(len(func))]
        ss = np.r_[-lof["delta"], -func["delta"]]
        fpr2, tpr2, _ = roc_curve(yy, ss)
        ax1.plot(fpr2, tpr2, lw=2, color=color,
                 label=f"{level} LOF (AUROC={roc_auc_score(yy, ss):.2f})")
    ax1.plot([0, 1], [0, 1], "--", color="gray")
    ax1.set(title="Severity failure mode: severe vs mild LOF",
            xlabel="false positive rate", ylabel="sensitivity"); ax1.legend()
    names = list(bench_all)
    ax2.barh(range(len(names)), [bench_all[n] for n in names], color="#2a9d8f")
    ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names)
    ax2.axvline(0.5, ls="--", color="gray"); ax2.set_xlim(0.4, 1.0)
    ax2.set(title="Benchmark vs established tools (all variants)", xlabel="AUROC")
    fig.suptitle("M3 — failure mode + benchmark", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "m3_severity_benchmark.png", dpi=130); plt.close(fig)
