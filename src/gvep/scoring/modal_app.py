"""Evo 2 delta-likelihood scoring on Modal (cloud GPU).

This is the Milestone 2 scoring engine. It runs the Evo 2 1B model on a Modal GPU and
computes, for every BRCA1 variant, the zero-shot pathogenicity signal:

    delta = score_sequences(variant_window) - score_sequences(reference_window)

A more negative delta means the variant makes the sequence look less like real DNA, i.e.
more likely disruptive. (See PRIMER.md.) The published 1B benchmark is AUROC ≈ 0.73 — our
sanity target for Milestone 3.

WHY MODAL: the evo2 package requires Transformer Engine + FP8, which needs an Ada/Hopper
GPU; Apple Silicon can't run it. Modal gives us a cheap FP8-capable L4 on demand.

RUN IT (from the repo root, after `modal token new`):
    # cheap smoke test first (validates the image + GPU + FP8 on ~4 sequences):
    .venv/bin/modal run -m gvep.scoring.modal_app::smoke
    # then the full dataset:
    .venv/bin/modal run -m gvep.scoring.modal_app::main
    # to use an H100 instead of the default L4 (if FP8 needs Hopper):
    GVEP_GPU=H100 .venv/bin/modal run -m gvep.scoring.modal_app::main

The full run writes data/cache/evo2_delta_scores.csv locally.
"""

from __future__ import annotations

import os

import modal

from gvep.config import EVO2_MODEL, MODAL_GPU, WINDOW_BP

GPU = os.environ.get("GVEP_GPU", MODAL_GPU)  # "L4" (default) or "H100"
SCORE_BATCH = int(os.environ.get("GVEP_BATCH", "8"))  # sequences per score call

# --- Container image -------------------------------------------------------
# Mirrors the evo2 README "full install" (required for the 1B model). The TE +
# flash-attn build is the fragile part; if it fails we iterate on this block.
image = (
    modal.Image.micromamba(python_version="3.12")
    .micromamba_install("cuda-nvcc", "cuda-cudart-dev", channels=["nvidia"])
    .micromamba_install("transformer-engine-torch=2.3.0", channels=["conda-forge"])
    .pip_install("flash-attn==2.8.0.post2", extra_options="--no-build-isolation")
    .pip_install("evo2", "biopython", "requests", "tqdm", "numpy", "pandas")
    .env({"HF_HOME": "/cache/hf"})  # cache model weights on the Volume
    .add_local_python_source("gvep")  # reuse our window-building code remotely
)

app = modal.App("evo2-brca1-scoring", image=image)
# Persisted across runs so the ~3.4 GB of weights + reference download once.
cache = modal.Volume.from_name("evo2-cache", create_if_missing=True)


# --- Remote helpers (run inside the GPU container) -------------------------
def _load_chr17() -> str:
    """Download + load GRCh37 chr17 into the container (cached on the Volume)."""
    import gzip
    from pathlib import Path

    from Bio import SeqIO

    from gvep.data.reference import CHR17_BYTES, CHR17_URL
    from gvep.utils.io import download_file

    dest = Path("/cache/ref/GRCh37.p13_chr17.fna.gz")
    download_file(CHR17_URL, dest, expected_bytes=CHR17_BYTES)
    with gzip.open(dest, "rt") as fh:
        return str(next(SeqIO.parse(fh, "fasta")).seq).upper()


def _score_all(model, seqs: list[str]) -> list[float]:
    """Score a list of sequences in batches; return one log-likelihood per sequence."""
    out: list[float] = []
    for i in range(0, len(seqs), SCORE_BATCH):
        chunk = seqs[i : i + SCORE_BATCH]
        scores = model.score_sequences(chunk)
        out.extend(float(s) for s in scores)
    return out


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=7200)
def score_variants(records: list[dict]) -> list[dict]:
    """Score every variant. `records` = [{idx, pos, ref, alt}, ...] (1-based pos)."""
    from evo2.models import Evo2

    from gvep.data.windows import build_window

    print(f"[remote] GPU={GPU}, model={EVO2_MODEL}, variants={len(records):,}")
    seq = _load_chr17()
    model = Evo2(EVO2_MODEL)

    # Build windows. Reference windows depend only on position, so dedup them:
    # ~3 alts share each position → ~1/3 the reference forward passes.
    ref_by_pos: dict[int, str] = {}
    var_seqs: list[str] = []
    for r in records:
        w = build_window(seq, r["pos"], r["ref"], r["alt"], WINDOW_BP)
        ref_by_pos.setdefault(r["pos"], w.ref_seq)
        var_seqs.append(w.var_seq)

    uniq_pos = list(ref_by_pos)
    print(f"[remote] scoring {len(uniq_pos):,} unique ref windows + "
          f"{len(var_seqs):,} variant windows")
    ref_scores = dict(zip(uniq_pos, _score_all(model, [ref_by_pos[p] for p in uniq_pos])))
    var_scores = _score_all(model, var_seqs)

    results = []
    for r, vs in zip(records, var_scores):
        rs = ref_scores[r["pos"]]
        results.append({**r, "ref_score": rs, "var_score": vs, "delta": vs - rs})
    return results


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=1800)
def smoke_score() -> dict:
    """Cheap validation: load the model and score 2 ref/var pairs on the chosen GPU."""
    from evo2.models import Evo2

    print(f"[remote] smoke test on GPU={GPU}")
    model = Evo2(EVO2_MODEL)
    ref = ["ACGTACGTACGTACGTACGT", "TTTTGGGGCCCCAAAATTTT"]
    var = ["ACGTACGTACGAACGTACGT", "TTTTGGGGCACCAAAATTTT"]
    rs, vs = model.score_sequences(ref), model.score_sequences(var)
    deltas = [float(v) - float(r) for r, v in zip(rs, vs)]
    print(f"[remote] ref={[float(x) for x in rs]} var={[float(x) for x in vs]}")
    return {"gpu": GPU, "model": EVO2_MODEL, "deltas": deltas}


# --- Local entrypoints (run on your Mac, orchestrate the remote work) ------
@app.local_entrypoint()
def smoke():
    """`modal run -m gvep.scoring.modal_app::smoke` — validate image + GPU + FP8."""
    result = smoke_score.remote()
    print(f"\n✅ Evo 2 loaded and scored on {result['gpu']}.")
    print(f"   deltas (sanity, should be finite numbers): {result['deltas']}")


@app.local_entrypoint()
def main():
    """`modal run -m gvep.scoring.modal_app::main` — score the full Findlay dataset."""
    import pandas as pd

    from gvep.config import CACHE_DIR
    from gvep.data.build import FINDLAY_OUT

    df = pd.read_csv(FINDLAY_OUT)
    records = [
        {"idx": int(i), "pos": int(row.pos), "ref": row.ref, "alt": row.alt}
        for i, row in df.iterrows()
    ]
    print(f"Scoring {len(records):,} variants on {GPU} via Modal...")
    results = score_variants.remote(records)

    scores = pd.DataFrame(results).set_index("idx")
    out = df.join(scores[["ref_score", "var_score", "delta"]])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / "evo2_delta_scores.csv"
    out.to_csv(dest, index=False)
    print(f"\n✅ wrote {dest}  ({len(out):,} rows)")
    print(out[["pos", "ref", "alt", "class", "delta"]].head().to_string(index=False))
