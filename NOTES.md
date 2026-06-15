# NOTES — Running Biology + ML Learning Log

This is the project's lab notebook. Each milestone adds an entry explaining the **biology**,
the **ML reasoning**, and any **decisions/gotchas**. It doubles as a learning log and as
portfolio documentation. Newest entries at the bottom of each milestone section.

---

## Milestone 0 — Scaffolding & orientation

**Date:** 2026-06-15

### What this milestone is
Set up the repo, environment, and conceptual + access-path groundwork before writing any
model or data code. Deliverables: structure, venv, deps, `PRIMER.md`, this log, and a
documented Evo 2 access decision (`docs/ACCESS_PATH.md`).

### Biology recap (see PRIMER.md for the full version)
- We're studying **BRCA1**, a tumor-suppressor gene on **chromosome 17, minus strand**.
  Loss-of-function variants raise hereditary breast/ovarian cancer risk.
- Ground truth comes from **Findlay et al. (2018)** saturation genome editing: experimental
  function scores for ~3,893 SNVs, splitting cleanly into **LOF / INT / FUNC**.

### ML reasoning
- **Evo 2** is a DNA language model; it scores how "biologically plausible" a sequence is.
- **Zero-shot delta-likelihood scoring:** `delta = var_log_prob - ref_log_prob`. Strongly
  negative delta ⇒ the variant makes the sequence look unnatural ⇒ likely disruptive. No
  variant labels are used to make this prediction — that's the "zero-shot" part.
- The project's real contribution is **Milestone 3's honesty layer** — not the headline
  AUROC, but the per-category, calibration, and failure-mode analysis around it.

### Key decisions & gotchas
- **Apple Silicon can't run Evo 2** (CUDA-only, StripedHyena 2 kernels). ⇒ inference on
  **Modal** cloud GPU; Mac handles data/analysis/UI.
- **Surprise:** the **1B model officially wants an H100 (FP8)**, while the **7B runs on any
  GPU in bf16**. NVIDIA's BRCA1 tutorial nonetheless runs the 1B on a non-Hopper A6000 in
  bf16, so we'll do the same and *measure* any accuracy gap vs. the published number.
- **Chosen path:** Modal + `evo2_1b_base` + **bf16, no FP8/Transformer Engine** (cheapest,
  lightest install, exposes the logits delta-scoring needs). Weights are **Apache-2.0**.
- **Reproducibility:** single global seed in `gvep.config`, applied via
  `gvep.utils.seed.set_seed()`. Deps pinned; `make lock` freezes exact versions.
- Local Python is 3.13 (fine for data/analysis/UI); the **Modal container pins 3.11/3.12**
  for Evo 2.

### Open items carried forward
- Confirm exact `evo2` install + log-likelihood API inside the Modal image at Milestone 2.
- Verify whether NVIDIA's hosted API exposes likelihoods (fallback path only).

### How to run what exists now
```bash
make setup   # venv + deps + editable install
make lock    # freeze versions
make help    # list milestone entry points
```

---

## Milestone 1 — Data layer

**Date:** 2026-06-15

### What this milestone is
Build the data foundation: fetch the BRCA1 reference region, the Findlay 2018 ground-truth
dataset, and a ClinVar slice; produce per-variant ref/variant sequence windows with
coordinates and labels; and *prove the data is internally consistent* before any modeling.

### Sources confirmed (research before coding)
The cleanest, canonical sources are the files bundled in the Evo 2 repo (so our pipeline
matches the published one exactly):
- **Findlay 2018:** `41586_2018_461_MOESM3_ESM.xlsx` (read with `header=2`); columns
  `chromosome, position (hg19), reference, alt, function.score.mean, func.class`.
- **Reference:** `GRCh37.p13_chr17.fna.gz`.
- **Critical fact:** coordinates are **hg19/GRCh37**, *not* GRCh38. Using the wrong build
  would have silently shifted every position. This is why the ref-allele check exists.
- **ClinVar:** NCBI E-utilities (`esearch`+`esummary`, JSON), term
  `BRCA1[gene] AND "single nucleotide variant"[Type of variation]`.

### Biology / ML reasoning
- Findlay function scores are **bimodal**: FUNC (tolerated) vs LOF (disruptive), with a thin
  INT band between. We'll treat LOF as "pathogenic-like" and FUNC as "benign-like" for the
  Milestone 3 evaluation; INT is genuinely ambiguous and we'll handle it explicitly.
- Each variant becomes two 8192-bp windows (ref vs alt at the center) — the exact inputs
  Evo 2 will score next milestone.

### Results
- **3,893 clean SNVs** (every raw row survived cleaning — the bundled file is already tidy).
- Class balance: **FUNC 72.5% / INT 6.4% / LOF 21.1%**. The 21% LOF minority is important:
  Milestone 3's honesty layer must report minority-class performance, not just aggregate AUROC.
- **Integrity: ref allele matches the genome for 3,893/3,893 (100%).** This is the headline —
  it validates coordinates, genome build, and the forward-strand assumption all at once.
- Score range `[-5.65, 1.31]` (lower = more disruptive), matching the published distribution.
- **ClinVar:** 4,999 BRCA1 SNVs, **1,410 VUS** (the variants a triage tool most wants to rank).

### Decisions & gotchas
- **Strand:** BRCA1 is minus-strand, but Findlay ref/alt are on the **forward** genomic strand
  (proven by the 100% ref match against plus-strand chr17). Evo 2 sees both strands in
  training, so we feed forward-strand windows directly — no reverse-complement.
- **Storage:** processed tables are small CSVs (coords + score + class + `ref_match`); we do
  NOT persist the 64 MB of full window strings — they're rebuilt on demand from the cached
  reference via `build_window()`. Keeps the repo light and the data regenerable.
- **0-based vs 1-based:** dataset positions are 1-based; we convert with `p = pos - 1` before
  string indexing. Off-by-one here would have broken the ref check — it didn't.
- **Known limitation (ClinVar, for later):** ~1,242 ClinVar records came back with an empty
  significance string (API stores some classifications under different keys). ClinVar is only
  used in Milestone 5, so we note this and will refine the parser when we actually consume it.

### How to run
```bash
make data        # or: python -m gvep.cli data
# outputs: data/processed/{findlay_brca1.csv, clinvar_brca1.csv, example_windows.txt}
```

---

## Milestone 2 — Core scoring engine
*(not started)*
