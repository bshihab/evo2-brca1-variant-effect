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
*(not started)*
