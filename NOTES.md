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

**Date:** 2026-06-15 (engine built; awaiting first Modal run for results)

### What this milestone is
Stand up Evo 2 1B on a cloud GPU and compute the zero-shot signal for every variant:
`delta = score_sequences(var_window) - score_sequences(ref_window)`. Then the first sanity
check: are LOF deltas more negative than FUNC deltas?

### Research before coding (and a correction to Milestone 0)
- **Scoring API confirmed** (from the official BRCA1 notebook): `model = Evo2('evo2_1b_base')`,
  then `model.score_sequences(list_of_seqs)` → one log-likelihood per sequence. Delta is the
  difference. **Published 1B benchmark: AUROC ≈ 0.73** — our sanity target.
- **CORRECTION:** Milestone 0 planned "1B in bf16, no FP8." That is impossible — the `evo2`
  package hard-requires Transformer Engine + FP8 to load the 1B model (GitHub issue #208
  errors without TE). FP8 hardware = Ada/Hopper/Blackwell only (NOT Ampere; A100/A10G are out).
- **Resolution:** embrace FP8 on an FP8-capable GPU. Modal's **L4 is Ada (compute 8.9) and
  supports FP8** — cheap *and* correct. H100 is the fallback. GPU is a one-line parameter, so
  switching needs no image rebuild.

### Engineering decisions
- **Modal micromamba image** mirrors the evo2 README "full install" (cuda-nvcc, TE 2.3.0 from
  conda-forge, flash-attn 2.8.0.post2, then `pip install evo2`). The TE + flash-attn build is
  the fragile part and may need iteration — that's expected.
- **Weights + reference cached on a Modal Volume** (`evo2-cache`) so the ~3.4 GB downloads once.
- **Ref-window dedup:** reference windows depend only on position, and ~3 alts share each
  position, so we score ~1/3 as many reference windows (≈1,300 unique refs + 3,893 vars instead
  of 7,786). Pure compute/cost saving with no effect on results.
- **Data transfer kept tiny:** only the small variant table (pos/ref/alt) is sent to the
  container; the 64 MB of window strings are rebuilt remotely via our `build_window()`.
- **Two entrypoints:** `smoke` (load model + score 4 seqs — cheap GPU/FP8 validation) and
  `main` (full dataset → `data/cache/evo2_delta_scores.csv`).

### Results (run completed)
- **Scored all 3,893 variants** on a Modal **L4** (FP8). ~75 min, ~$1.
- **Quick AUROC (LOF vs FUNC) = 0.737; LOF vs rest = 0.729.** The published Evo 2 1B
  benchmark is **~0.73 — we reproduced it almost exactly.** Strong end-to-end validation
  that the data layer, coordinates/build, windows, FP8-on-L4, and delta scoring are all correct.
- Median delta steps down by class exactly as expected: **FUNC −0.0001 > INT −0.0003 >
  LOF −0.0006** (more negative = more disruptive). Direction check ✅.
- Plot: `results/figures/m2_delta_distributions.png` — FUNC tightly peaked at 0, LOF shifted
  left with a heavy disruptive tail, INT in between.

### Note on the tiny absolute delta values
`score_sequences` returns the **mean** log-likelihood per token (averaged over 8,192 tokens),
so a single-base change moves the average by a tiny amount (~1e-3). The *absolute* numbers look
small, but what matters for classification is the **separation** between classes — and AUROC
0.73 confirms that separation is real and on par with the published result.

### Gotchas encountered (worth remembering)
- The `evo2` package **requires FP8/Transformer Engine**; the Modal image needed Python 3.12 +
  **CUDA 12.9** + `transformer-engine-torch=2.3.0` (the solver dictated the exact versions), plus
  `ca-certificates` for the HuggingFace download to verify SSL.
- **L4 (Ada) runs the 1B fine in FP8** — no Hopper needed.
- **Resilience matters:** the first full run died when the Modal **free-tier budget** was hit
  mid-run (not the Mac sleeping, as first assumed). Fixed by (a) persisting results to a Modal
  Volume + committing, and (b) running with `--detach` so a dropped local connection can't kill
  it. Recovery path: `modal run -m gvep.scoring.modal_app::fetch`.

### How to run
```bash
modal run --detach -m gvep.scoring.modal_app::main   # full scoring (writes to Volume + local)
python -m gvep.cli sanity                            # plot + quick AUROC
```

---

## Milestone 3 — Validation & honesty layer

**Date:** 2026-06-17

### What this milestone is
The centerpiece: go *beyond* the headline AUROC and honestly characterize where the 1B
zero-shot model works and fails. Full writeup in **RESULTS.md**; code in
`analysis/honesty.py` + `analysis/dataset.py`; run via `python -m gvep.cli validate`.

### Data win
The raw Findlay xlsx has 62 columns, including `consequence` (missense/splice/intronic/…),
and bundled scores for **CADD, phyloP, SIFT, PolyPhen-2** — so we could stratify by category
*and* benchmark Evo 2 against established tools for free.

### Headline findings (honest, not flattering)
- **AUROC 0.737 (CI 0.715–0.757)** reproduces the published 1B number; AUPRC 0.564 (baseline 0.226).
- **The aggregate is inflated by between-category structure.** Within **missense** (the bulk of
  real VUS) AUROC collapses to **0.604**; the model's relative strength is **splice region (0.852)**.
  Several categories aren't even evaluable (one class absent — e.g. all Nonsense are LOF).
- **At 90% sensitivity: FPR 77%, precision 25% → ~2.9 false alarms per true LOF.** Class-imbalance
  honesty in one number.
- **Severity failure mode:** severe LOF 0.775 vs mild LOF 0.698 (degrades 0.077 on subtle variants).
- **Benchmark (humbling):** Evo 2 1B (0.737) is **beaten by CADD (0.817) and phyloP (0.793)**, and on
  missense trails every classic tool (0.604 vs CADD 0.756 / phyloP 0.737 / SIFT 0.708 / PolyPhen 0.669).

### Caveats kept honest
- Used the **1B budget model**; published 40B is ~0.87+ on BRCA1 — much of the gap is size.
- Evo 2 is **zero-shot**; CADD/SIFT/PolyPhen are supervised (possible circularity). But phyloP is
  unsupervised conservation and still beats the 1B — the most sobering comparison.

### ML concepts exercised
AUROC vs AUPRC, bootstrap CIs, operating point at fixed sensitivity, FPR/precision under
imbalance, calibration (CV logistic recalibration + Brier + reliability diagram), stratified
evaluation, and "AUROC not computable when a stratum is single-class."

### Figures (committed for the portfolio)
`m3_roc_pr.png`, `m3_by_consequence.png`, `m3_calibration.png`, `m3_severity_benchmark.png`.

---

## Milestone 4 — Embedding-based classifier
*(not started)*
