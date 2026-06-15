"""Findlay et al. (2018) BRCA1 saturation genome editing dataset — our ground truth.

~3,900 single-nucleotide variants across 13 BRCA1 exons, each with an experimentally
measured function score and a functional class:
  * FUNC = functional (variant tolerated)        -> treated as benign-like
  * LOF  = loss of function (variant disruptive)  -> treated as pathogenic-like
  * INT  = intermediate                           -> the ambiguous middle

We use the exact supplementary file bundled in the Evo 2 repo so coordinates/columns
match the canonical pipeline. Coordinates are hg19/GRCh37 (see reference.py).
"""

from __future__ import annotations

import pandas as pd

from gvep.config import RAW_DIR
from gvep.utils.io import download_file

FINDLAY_URL = (
    "https://raw.githubusercontent.com/ArcInstitute/evo2/main/"
    "notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx"
)
FINDLAY_BYTES = 2_268_761
FINDLAY_XLSX = RAW_DIR / "findlay2018_brca1_sge.xlsx"

_VALID_BASES = set("ACGT")
_VALID_CLASSES = {"FUNC", "INT", "LOF"}


def download_findlay() -> "object":
    """Fetch the Findlay supplementary .xlsx into data/raw/ (cached)."""
    return download_file(FINDLAY_URL, FINDLAY_XLSX, expected_bytes=FINDLAY_BYTES)


def load_findlay() -> pd.DataFrame:
    """Load and clean the Findlay dataset into a tidy per-variant table.

    Returns columns: chrom, pos (hg19, 1-based), ref, alt, score, class.
    Cleaning keeps only well-formed SNVs on chr17 with a valid class and score.
    """
    download_findlay()
    # header=2: the real column names are on the third row of the sheet.
    df = pd.read_excel(FINDLAY_XLSX, header=2, engine="openpyxl")

    df = df[
        ["chromosome", "position (hg19)", "reference", "alt",
         "function.score.mean", "func.class"]
    ].rename(
        columns={
            "chromosome": "chrom",
            "position (hg19)": "pos",
            "reference": "ref",
            "alt": "alt",
            "function.score.mean": "score",
            "func.class": "class",
        }
    )

    n_raw = len(df)

    # --- cleaning / sanity filters ---
    df["ref"] = df["ref"].astype(str).str.upper().str.strip()
    df["alt"] = df["alt"].astype(str).str.upper().str.strip()
    df["class"] = df["class"].astype(str).str.upper().str.strip()
    df["chrom"] = df["chrom"].astype(str).str.replace("chr", "", case=False).str.strip()

    df = df[df["chrom"] == "17"]                              # BRCA1 is on chr17
    df = df[df["ref"].isin(_VALID_BASES) & df["alt"].isin(_VALID_BASES)]  # SNVs only
    df = df[df["ref"] != df["alt"]]                           # must be a real change
    df = df[df["class"].isin(_VALID_CLASSES)]                 # known functional class
    df = df.dropna(subset=["pos", "score"])
    df["pos"] = df["pos"].astype(int)

    df = df.reset_index(drop=True)
    print(f"[gvep] Findlay: {len(df):,} clean SNVs (from {n_raw:,} raw rows)")
    return df
