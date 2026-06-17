"""Build the enriched analysis table for the Milestone 3 honesty layer.

Merges our Evo 2 delta scores with the rich Findlay annotations (variant consequence,
established predictor scores, allele frequency) and derives the stratification columns
the honesty layer needs. Everything downstream reads from load_scored_dataset().
"""

from __future__ import annotations

import pandas as pd

from gvep.config import CACHE_DIR
from gvep.data.findlay import FINDLAY_XLSX, download_findlay

SCORES_CSV = CACHE_DIR / "evo2_delta_scores.csv"

# Findlay's `consequence` categories grouped into coding vs non-coding/regulatory.
CODING = {"Missense", "Synonymous", "Nonsense"}
NONCODING = {"Intronic", "Splice region", "Canonical splice", "5' UTR"}

# Established predictors we benchmark Evo 2 against, and the SIGN that makes each
# "higher = more likely loss-of-function" (so roc_auc_score reads them consistently).
# Evo 2: more-negative delta = more disruptive, so the LOF-predictor is -delta.
PREDICTORS = {
    "Evo 2 (Δ, zero-shot)": ("delta", -1),
    "CADD": ("CADD.score", +1),
    "phyloP": ("phylop", +1),
    "SIFT": ("sift", -1),          # lower SIFT = more damaging (missense only)
    "PolyPhen-2": ("polyphen2", +1),  # higher = more damaging (missense only)
}


def _snv_type(ref: str, alt: str) -> str:
    """Transition (purine<->purine or pyrimidine<->pyrimidine) vs transversion."""
    transitions = ({"A", "G"}, {"C", "T"})
    return "transition" if {ref, alt} in transitions else "transversion"


def load_scored_dataset() -> pd.DataFrame:
    """Return one tidy row per variant: Evo 2 scores + annotations + derived columns."""
    if not SCORES_CSV.exists():
        raise SystemExit(
            f"No scores at {SCORES_CSV}. Run the scoring job first:\n"
            "  modal run --detach -m gvep.scoring.modal_app::main"
        )
    scores = pd.read_csv(SCORES_CSV)  # pos, ref, alt, score, class, delta, ...

    download_findlay()
    ann = pd.read_excel(FINDLAY_XLSX, header=2, engine="openpyxl")[
        ["position (hg19)", "reference", "alt", "consequence", "protein_variant",
         "CADD.score", "phyloP (mammalian)", "sift", "polyphen2",
         "gnomAD_AF", "clinvar_simple"]
    ].rename(columns={
        "position (hg19)": "pos",
        "reference": "ref",
        "alt": "alt",
        "phyloP (mammalian)": "phylop",
        "clinvar_simple": "clinvar",
    })
    ann["ref"] = ann["ref"].astype(str).str.upper().str.strip()
    ann["alt"] = ann["alt"].astype(str).str.upper().str.strip()

    df = scores.merge(ann, on=["pos", "ref", "alt"], how="left")

    # --- derived stratification columns ---
    df["snv_type"] = [_snv_type(r, a) for r, a in zip(df["ref"], df["alt"])]
    df["region"] = df["consequence"].map(
        lambda c: "coding" if c in CODING else "non-coding"
    )
    df["is_lof"] = (df["class"] == "LOF").astype(int)
    # Severity: within LOF, split by experimental function score at the LOF median
    # (more negative score = more severe loss of function).
    lof_median = df.loc[df["class"] == "LOF", "score"].median()
    df["severity"] = "n/a"
    df.loc[df["class"] == "LOF", "severity"] = [
        "severe" if s <= lof_median else "mild"
        for s in df.loc[df["class"] == "LOF", "score"]
    ]
    return df
