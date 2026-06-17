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
    # Python 3.12: transformer-engine-torch 2.3.0 on conda-forge ships only for
    # python 3.10 or 3.12 (the build solver confirmed 3.11 is unavailable).
    modal.Image.micromamba(python_version="3.12")
    # Image builds run on a CPU-only machine, so conda can't "see" a GPU and refuses
    # to select CUDA-enabled PyTorch. This tells it to assume CUDA 12.9 is present.
    .env({"CONDA_OVERRIDE_CUDA": "12.9"})
    # One combined solve so cuda-nvcc, the CUDA runtime, PyTorch, and Transformer
    # Engine all land on a mutually-compatible set. TE 2.3.0 specifically requires
    # CUDA 12.9 (cuda-cudart >=12.9.37,<13), so we pin cuda-version to match.
    .micromamba_install(
        "cuda-nvcc", "cuda-cudart-dev", "cuda-version=12.9",
        "transformer-engine-torch=2.3.0",
        channels=["conda-forge", "nvidia"],
    )
    .pip_install("flash-attn==2.8.0.post2", extra_options="--no-build-isolation")
    .pip_install("evo2", "biopython", "requests", "tqdm", "numpy", "pandas")
    # Separate, cheap layer (keeps the heavy layers above cached): CA bundle so
    # HTTPS to huggingface.co verifies inside the fresh container.
    .micromamba_install("ca-certificates", channels=["conda-forge"])
    .env({
        "HF_HOME": "/cache/hf",  # cache model weights on the Volume
        # Point Python/httpx/requests at the conda CA bundle (the fresh container
        # has no system CA path configured, which broke the HF model download).
        "SSL_CERT_FILE": "/opt/conda/ssl/cacert.pem",
        "REQUESTS_CA_BUNDLE": "/opt/conda/ssl/cacert.pem",
    })
    .add_local_python_source("gvep")  # reuse our window-building code remotely
)

app = modal.App("evo2-brca1-scoring", image=image)
# Persisted across runs so the ~3.4 GB of weights + reference download once.
cache = modal.Volume.from_name("evo2-cache", create_if_missing=True)

# Results are written HERE on the Volume (server-side) so they survive a dropped
# local connection (e.g. the Mac sleeping). We then fetch them in a quick call.
RESULTS_PATH = "/cache/results/raw_scores.csv"

# Milestone 4: embeddings. We extract a late-middle layer's representation at the
# variant position and store the (variant - reference) difference vector per variant.
EMB_LAYER = os.environ.get("GVEP_EMB_LAYER", "blocks.20.mlp.l3")
EMB_PATH = "/cache/results/embeddings.npz"


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


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=14400)
def score_variants(records: list[dict]) -> int:
    """Score every variant and persist results to the Volume. Returns row count.

    `records` = [{idx, pos, ref, alt}, ...] (1-based pos). Results are written to
    RESULTS_PATH on the Volume and committed BEFORE returning, so they're safe even
    if the local connection drops and the return value never gets delivered.
    """
    import csv
    import os

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

    # Persist to the Volume and commit so the results survive a dropped connection.
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["idx", "pos", "ref", "alt", "ref_score", "var_score", "delta"]
        )
        writer.writeheader()
        writer.writerows(results)
    cache.commit()
    print(f"[remote] ✅ wrote {len(results):,} rows to {RESULTS_PATH} on the Volume")
    return len(results)


@app.function(volumes={"/cache": cache})
def get_results() -> str:
    """Return the raw scores CSV text from the Volume (recovery / fetch path)."""
    cache.reload()
    with open(RESULTS_PATH) as fh:
        return fh.read()


# --- Milestone 4: embedding extraction -------------------------------------
def _embed_center(model, seqs: list[str], centers: list[int]) -> list:
    """Return the layer-EMB_LAYER embedding vector at each sequence's center position."""
    import torch

    out = []
    for seq, c in zip(seqs, centers):
        ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int).unsqueeze(0).to("cuda:0")
        with torch.no_grad():
            _, emb = model(ids, return_embeddings=True, layer_names=[EMB_LAYER])
        out.append(emb[EMB_LAYER][0, c, :].float().cpu().numpy())
    return out


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=14400)
def extract_embeddings(records: list[dict]) -> int:
    """Per variant, store the (variant - reference) embedding difference to the Volume."""
    import numpy as np

    from evo2.models import Evo2

    from gvep.data.windows import build_window

    print(f"[remote] embeddings: GPU={GPU}, layer={EMB_LAYER}, variants={len(records):,}")
    seq = _load_chr17()
    model = Evo2(EVO2_MODEL)

    ref_by_pos: dict[int, tuple[str, int]] = {}
    var_items = []  # (idx, pos, var_seq, center)
    for r in records:
        w = build_window(seq, r["pos"], r["ref"], r["alt"], WINDOW_BP)
        ref_by_pos[r["pos"]] = (w.ref_seq, w.center)
        var_items.append((r["idx"], r["pos"], w.var_seq, w.center))

    pos_list = list(ref_by_pos)
    print(f"[remote] embedding {len(pos_list):,} ref + {len(var_items):,} var windows")
    ref_vecs = _embed_center(model, [ref_by_pos[p][0] for p in pos_list],
                             [ref_by_pos[p][1] for p in pos_list])
    ref_emb = dict(zip(pos_list, ref_vecs))
    var_vecs = _embed_center(model, [it[2] for it in var_items], [it[3] for it in var_items])

    idxs = np.array([it[0] for it in var_items])
    diff = np.stack([v - ref_emb[it[1]] for v, it in zip(var_vecs, var_items)]).astype("float32")
    import os

    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    np.savez_compressed(EMB_PATH, idx=idxs, diff=diff)
    cache.commit()
    print(f"[remote] ✅ wrote embeddings {diff.shape} to {EMB_PATH}")
    return int(len(idxs))


@app.function(gpu=GPU, volumes={"/cache": cache}, timeout=1800)
def smoke_embed() -> dict:
    """Validate the embedding layer name on one short sequence (cheap)."""
    import torch

    from evo2.models import Evo2

    model = Evo2(EVO2_MODEL)
    ids = torch.tensor(model.tokenizer.tokenize("ACGTACGTACGTACGT"), dtype=torch.int).unsqueeze(0).to("cuda:0")
    _, emb = model(ids, return_embeddings=True, layer_names=[EMB_LAYER])
    return {"layer": EMB_LAYER, "shape": tuple(emb[EMB_LAYER].shape)}


@app.function(volumes={"/cache": cache})
def get_embeddings() -> bytes:
    """Return the embeddings .npz bytes from the Volume."""
    cache.reload()
    with open(EMB_PATH, "rb") as fh:
        return fh.read()


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


def _save_from_volume() -> None:
    """Fetch raw scores from the Volume, join with the Findlay table, save locally.

    Used by both `main` (after scoring) and `fetch` (recovery, no re-scoring). The
    fetch itself is a tiny, fast call, so it's robust even on a flaky connection.
    """
    import io

    import pandas as pd

    from gvep.config import CACHE_DIR
    from gvep.data.build import FINDLAY_OUT

    raw = get_results.remote()
    scores = pd.read_csv(io.StringIO(raw)).set_index("idx")
    df = pd.read_csv(FINDLAY_OUT)
    out = df.join(scores[["ref_score", "var_score", "delta"]])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / "evo2_delta_scores.csv"
    out.to_csv(dest, index=False)
    print(f"\n✅ wrote {dest}  ({len(out):,} rows)")
    print(out[["pos", "ref", "alt", "class", "delta"]].head().to_string(index=False))


@app.local_entrypoint()
def main():
    """Score the full dataset. Run DETACHED so a sleeping Mac can't kill it:
        modal run --detach -m gvep.scoring.modal_app::main
    Results are persisted to the Volume; if your connection drops, recover with:
        modal run -m gvep.scoring.modal_app::fetch
    """
    import pandas as pd

    from gvep.data.build import FINDLAY_OUT

    df = pd.read_csv(FINDLAY_OUT)
    records = [
        {"idx": int(i), "pos": int(row.pos), "ref": row.ref, "alt": row.alt}
        for i, row in df.iterrows()
    ]
    print(f"Scoring {len(records):,} variants on {GPU} via Modal (detached-safe)...")
    score_variants.remote(records)  # computes + persists to the Volume
    _save_from_volume()             # then pull results down + join + save locally


@app.local_entrypoint()
def fetch():
    """`modal run -m gvep.scoring.modal_app::fetch` — recover results from the Volume
    (e.g. if the scoring run completed server-side but the local connection dropped)."""
    _save_from_volume()


# --- Milestone 4 embedding entrypoints -------------------------------------
def _save_embeddings_from_volume() -> None:
    import io

    import numpy as np

    from gvep.config import CACHE_DIR

    raw = get_embeddings.remote()
    arr = np.load(io.BytesIO(raw))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / "evo2_embeddings.npz"
    dest.write_bytes(raw)
    print(f"\n✅ wrote {dest}  (diff shape {arr['diff'].shape})")


@app.local_entrypoint()
def embed_smoke():
    """`modal run -m gvep.scoring.modal_app::embed_smoke` — validate the embedding layer."""
    r = smoke_embed.remote()
    print(f"\n✅ embeddings work. layer={r['layer']} shape={r['shape']} "
          f"(batch, length, hidden)")


@app.local_entrypoint()
def embed():
    """Extract embeddings for all variants (detached-safe). Recover with ::embed_fetch."""
    import pandas as pd

    from gvep.data.build import FINDLAY_OUT

    df = pd.read_csv(FINDLAY_OUT)
    records = [
        {"idx": int(i), "pos": int(row.pos), "ref": row.ref, "alt": row.alt}
        for i, row in df.iterrows()
    ]
    print(f"Extracting embeddings for {len(records):,} variants on {GPU}...")
    extract_embeddings.remote(records)
    _save_embeddings_from_volume()


@app.local_entrypoint()
def embed_fetch():
    """`modal run -m gvep.scoring.modal_app::embed_fetch` — recover embeddings from Volume."""
    _save_embeddings_from_volume()
