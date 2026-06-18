"""Milestone 4 — push past zero-shot with an embedding-based classifier.

Instead of using Evo 2's delta *score*, we use its internal *embedding* (the 1920-dim
representation it computes at the variant position) and train a lightweight supervised
classifier on the (variant - reference) difference vector. The question: can a small
model on top of Evo 2's features beat the zero-shot AUROC (~0.74)?

Rigor choices (to avoid fooling ourselves):
  * **Grouped cross-validation by genomic position** (StratifiedGroupKFold on `pos`): all
    alternate alleles at a position stay in the same fold, so the test fold contains
    positions never seen in training. This measures generalization to NEW positions, not
    memorization of position-specific signal — a real leakage risk otherwise.
  * Compare against the zero-shot baseline **on the exact same samples**.
  * Report the **train-vs-CV gap** as an overfitting check.

Generalization caveat: this is still all within ONE gene (BRCA1). Cross-gene
generalization is explicitly NOT tested here (that's Milestone 7).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gvep.analysis.dataset import load_scored_dataset
from gvep.config import CACHE_DIR, FIGURES_DIR, METRICS_DIR
from gvep.utils.seed import set_seed

EMB_NPZ = CACHE_DIR / "evo2_embeddings.npz"
N_SPLITS = 5


def _cv_auroc(make_est, X, y, groups) -> tuple[float, float, float, np.ndarray]:
    """Grouped-CV out-of-fold AUROC/AUPRC + train AUROC (overfitting check)."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    oof = cross_val_predict(make_est(), X, y, groups=groups, cv=cv,
                            method="predict_proba")[:, 1]
    cv_auroc = roc_auc_score(y, oof)
    cv_auprc = average_precision_score(y, oof)
    est = make_est().fit(X, y)  # fit on all -> train-set AUROC (optimistic)
    train_auroc = roc_auc_score(y, est.predict_proba(X)[:, 1])
    return cv_auroc, cv_auprc, train_auroc, oof


def run() -> dict:
    set_seed()
    if not EMB_NPZ.exists():
        raise SystemExit(
            f"No embeddings at {EMB_NPZ}. Extract them first:\n"
            "  modal run -m gvep.scoring.modal_app::embed"
        )
    npz = np.load(EMB_NPZ)
    # Materialize the arrays ONCE — indexing npz[...] re-reads the whole archive each time.
    diff_arr, idx_arr = npz["diff"], npz["idx"]
    emb = {int(i): diff_arr[k] for k, i in enumerate(idx_arr)}

    df = load_scored_dataset().reset_index(drop=True)
    df["idx"] = df.index
    df = df[df["class"].isin(["FUNC", "LOF"])].copy()           # clean problem
    df = df[df["idx"].isin(emb)].copy()
    X = np.stack([emb[i] for i in df["idx"]])
    y = df["is_lof"].to_numpy()
    groups = df["pos"].to_numpy()
    zs = -df["delta"].to_numpy()  # zero-shot predictor on the SAME samples

    print("\n" + "=" * 70)
    print("  MILESTONE 4 — EMBEDDING-BASED CLASSIFIER (vs zero-shot)")
    print("=" * 70)
    print(f"  samples={len(df):,}  features={X.shape[1]}  positions(groups)={len(set(groups)):,}")

    M: dict = {"n": int(len(df)), "n_features": int(X.shape[1])}

    # zero-shot baseline on these samples
    zs_auroc = roc_auc_score(y, zs)
    zs_auprc = average_precision_score(y, zs)
    M["zero_shot"] = {"auroc": float(zs_auroc), "auprc": float(zs_auprc)}
    print(f"\n  zero-shot (Δ score):        AUROC={zs_auroc:.3f}  AUPRC={zs_auprc:.3f}")

    # PCA(50) compresses the 1,920-dim embedding diff: far less memory + compute, and
    # fewer parameters to overfit. Both models share this front-end.
    estimators = {
        "logreg": lambda: make_pipeline(
            StandardScaler(), PCA(n_components=50, random_state=0),
            LogisticRegression(C=0.5, max_iter=2000)),
        "mlp": lambda: make_pipeline(
            StandardScaler(), PCA(n_components=50, random_state=0),
            MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2,
                          max_iter=300, random_state=0)),
    }
    oof_by_model = {}
    for name, make_est in estimators.items():
        cv_auroc, cv_auprc, train_auroc, oof = _cv_auroc(make_est, X, y, groups)
        oof_by_model[name] = oof
        M[name] = {"cv_auroc": cv_auroc, "cv_auprc": cv_auprc,
                   "train_auroc": train_auroc, "gain_vs_zeroshot": cv_auroc - zs_auroc}
        print(f"\n  {name:<8} (grouped {N_SPLITS}-fold CV):")
        print(f"      CV  AUROC={cv_auroc:.3f}  AUPRC={cv_auprc:.3f}  "
              f"(gain vs zero-shot {cv_auroc - zs_auroc:+.3f})")
        print(f"      train AUROC={train_auroc:.3f}  -> overfit gap "
              f"{train_auroc - cv_auroc:+.3f}")

    # Where does it help? The category zero-shot was weakest on: missense.
    print("\n  missense-only (the hard category zero-shot scored ~0.60):")
    mis = (df["consequence"] == "Missense").to_numpy()
    M["missense"] = {}
    if mis.sum() > 50:
        zs_mis = roc_auc_score(y[mis], zs[mis])
        emb_mis = roc_auc_score(y[mis], oof_by_model["logreg"][mis])  # reuse OOF
        M["missense"] = {"zero_shot": float(zs_mis), "logreg": float(emb_mis),
                         "gain": float(emb_mis - zs_mis)}
        print(f"      zero-shot={zs_mis:.3f}   logreg(emb)={emb_mis:.3f}   "
              f"gain {emb_mis - zs_mis:+.3f}")

    _plot(M)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "classifier_metrics.json").write_text(json.dumps(M, indent=2))
    print(f"\n  metrics -> {METRICS_DIR / 'classifier_metrics.json'}")
    print(f"  figure  -> {FIGURES_DIR / 'm4_classifier.png'}")
    print("=" * 70 + "\n")
    return M


def _plot(M: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    names = ["zero-shot Δ", "logreg(emb)", "MLP(emb)"]
    vals = [M["zero_shot"]["auroc"], M["logreg"]["cv_auroc"], M["mlp"]["cv_auroc"]]
    colors = ["#8d99ae", "#2a9d8f", "#264653"]
    ax1.bar(names, vals, color=colors)
    ax1.axhline(M["zero_shot"]["auroc"], ls="--", color="gray")
    ax1.set_ylim(0.5, 1.0); ax1.set_ylabel("AUROC (grouped CV)")
    ax1.set_title("Embedding classifier vs zero-shot (all clean variants)")
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    # train vs CV (overfitting) for each model
    mods = ["logreg", "mlp"]
    cvv = [M[m]["cv_auroc"] for m in mods]
    trv = [M[m]["train_auroc"] for m in mods]
    x = np.arange(len(mods))
    ax2.bar(x - 0.2, trv, 0.4, label="train (optimistic)", color="#e9c46a")
    ax2.bar(x + 0.2, cvv, 0.4, label="grouped CV (honest)", color="#2a9d8f")
    ax2.set_xticks(x); ax2.set_xticklabels(mods); ax2.set_ylim(0.5, 1.0)
    ax2.set_title("Overfitting check: train vs CV AUROC"); ax2.legend()

    fig.suptitle("M4 — supervised classifier on Evo 2 embeddings", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "m4_classifier.png", dpi=130); plt.close(fig)
