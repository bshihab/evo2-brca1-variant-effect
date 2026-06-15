"""Build the processed data layer and report a summary.

Orchestrates Milestone 1 end to end:
  1. download GRCh37 chr17 reference
  2. load + clean the Findlay dataset
  3. validate integrity: build a window per variant and check ref allele == genome
  4. fetch the ClinVar BRCA1 slice (supplementary)
  5. save processed tables to data/processed/ and print a summary

Run via `python -m gvep.cli data` (or `make data`).
"""

from __future__ import annotations

import pandas as pd

from gvep.config import GENE, PROCESSED_DIR, WINDOW_BP
from gvep.data.clinvar import fetch_clinvar_brca1
from gvep.data.findlay import load_findlay
from gvep.data.reference import load_chr17_sequence
from gvep.data.windows import build_window

FINDLAY_OUT = PROCESSED_DIR / "findlay_brca1.csv"
CLINVAR_OUT = PROCESSED_DIR / "clinvar_brca1.csv"
EXAMPLES_OUT = PROCESSED_DIR / "example_windows.txt"


def validate_findlay(df: pd.DataFrame, seq: str) -> pd.DataFrame:
    """Build a window per variant and flag whether the ref allele matches the genome.

    Adds a `ref_match` column. This is the key data-integrity gate: if our coordinates,
    build, or strand assumptions were wrong, ref alleles would not match the genome.
    """
    matches, genome_bases = [], []
    for row in df.itertuples(index=False):
        w = build_window(seq, row.pos, row.ref, row.alt, WINDOW_BP)
        matches.append(w.ref_matches)
        genome_bases.append(w.genome_base)
    out = df.copy()
    out["ref_match"] = matches
    out["genome_base"] = genome_bases
    return out


def _save_examples(df: pd.DataFrame, seq: str, n: int = 2) -> None:
    """Write a couple of full ref/var windows to disk so they can be eyeballed."""
    lines = []
    for cls in ("FUNC", "LOF"):
        sub = df[df["class"] == cls]
        if sub.empty:
            continue
        row = sub.iloc[0]
        w = build_window(seq, int(row.pos), row.ref, row.alt, WINDOW_BP)
        flank = 15
        c = w.center
        lines += [
            f"=== {cls} example: chr17:{row.pos} {row.ref}>{row.alt} "
            f"(score={row.score:.3f}) ===",
            f"window length : {len(w.ref_seq)} bp (target {WINDOW_BP})",
            f"variant index : {w.center}",
            f"ref ...{w.ref_seq[c-flank:c]}[{w.ref_seq[c]}]{w.ref_seq[c+1:c+1+flank]}...",
            f"var ...{w.var_seq[c-flank:c]}[{w.var_seq[c]}]{w.var_seq[c+1:c+1+flank]}...",
            "",
        ]
    EXAMPLES_OUT.write_text("\n".join(lines))


def _summary(findlay: pd.DataFrame, clinvar: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print(f"  MILESTONE 1 DATA SUMMARY — {GENE}")
    print("=" * 64)

    n = len(findlay)
    print(f"\nFindlay 2018 (ground truth): {n:,} clean SNVs")
    print("  class balance:")
    counts = findlay["class"].value_counts()
    for cls in ("FUNC", "INT", "LOF"):
        c = int(counts.get(cls, 0))
        print(f"    {cls:4s}: {c:5,d}  ({c / n:5.1%})")
    n_lof = int(counts.get("LOF", 0))
    print(f"  LOF (minority/pathogenic-like) fraction: {n_lof / n:.1%}  "
          "<- note this imbalance for Milestone 3")

    matched = int(findlay["ref_match"].sum())
    print(f"\n  integrity: ref allele matches genome for "
          f"{matched:,}/{n:,} ({matched / n:.2%})")
    if matched != n:
        bad = findlay[~findlay["ref_match"]][["pos", "ref", "genome_base", "class"]]
        print(f"  !! {n - matched} mismatches (showing up to 5):")
        print(bad.head().to_string(index=False))

    print(f"\n  score range: [{findlay['score'].min():.3f}, "
          f"{findlay['score'].max():.3f}]  (lower = more disruptive)")
    print("\n  example rows:")
    print(findlay[["pos", "ref", "alt", "score", "class", "ref_match"]]
          .head(4).to_string(index=False))

    print(f"\nClinVar BRCA1 slice: {len(clinvar):,} SNVs")
    if not clinvar.empty:
        print(f"  VUS (uncertain significance): {int(clinvar['is_vus'].sum()):,}")
        print("  clinical significance breakdown (top 6):")
        for sig, c in clinvar["clin_sig"].value_counts().head(6).items():
            print(f"    {c:5,d}  {sig}")

    print("\n  saved:")
    print(f"    {FINDLAY_OUT}")
    print(f"    {CLINVAR_OUT}")
    print(f"    {EXAMPLES_OUT}")
    print("=" * 64 + "\n")


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[gvep] loading GRCh37 chr17 reference...")
    seq = load_chr17_sequence()
    print(f"[gvep] chr17 length: {len(seq):,} bp")

    findlay = validate_findlay(load_findlay(), seq)
    findlay.to_csv(FINDLAY_OUT, index=False)
    _save_examples(findlay, seq)

    clinvar = fetch_clinvar_brca1()
    clinvar.to_csv(CLINVAR_OUT, index=False)

    _summary(findlay, clinvar)
