# Genomic Variant Effect Prediction — Hereditary Cancer Risk (BRCA1)

> **⚠️ Not a clinical diagnostic.** This is a research / triage-prioritization
> proof-of-concept. It is intended to help *prioritize* uncertain genetic variants
> for expert review and to study a genomic foundation model's real-world behavior —
> **not** to diagnose disease or guide any medical decision. Every output is an
> AI prediction with documented, category-specific limitations.

A portfolio-quality POC that uses the **Evo 2** genomic foundation model (Arc Institute)
to score the pathogenicity of single-nucleotide variants in **BRCA1** via **zero-shot
delta-likelihood scoring**, then builds a validation + honesty layer on top that is rigorous
about where the model succeeds and fails.

New to this field? Start with **[PRIMER.md](PRIMER.md)** — a plain-language explanation of
genomic foundation models, zero-shot variant effect prediction, delta-likelihood scoring,
and why BRCA1 is the canonical test case.

---

## Problem statement

Most variants found in cancer-risk genes like BRCA1 are **Variants of Uncertain Significance
(VUS)** — we don't yet know if they're harmful. Experimentally testing each one is slow and
expensive. Can a foundation model trained on raw DNA *predict* which variants disrupt gene
function, well enough to help **triage** which VUS deserve expert attention first?

## Approach (high level)

1. **Data layer** — reference BRCA1 region (chr17), the Findlay et al. (2018) saturation
   mutagenesis dataset (~3,893 SNVs with experimental function scores), and a slice of ClinVar.
2. **Zero-shot scoring** — for each variant, score the reference vs. variant sequence window
   with Evo 2 and compute `delta = var_log_prob - ref_log_prob`. More negative = more disruptive.
3. **Validation + honesty layer** *(the centerpiece)* — headline AUROC/AUPRC, then go further:
   per-category performance, false-positive rates, calibration, severity-dependent failure
   modes, and class-imbalance honesty.
4. **Push past zero-shot** — a lightweight supervised classifier on Evo 2 embeddings.
5. **Explanation layer** — plain-language, category-aware, uncertainty-honest per-variant output.
6. **Demo** — FastAPI + Streamlit VUS-prioritization view.

## Model access path

Evo 2 runs on a **cloud NVIDIA GPU via Modal** (the model is CUDA-only; Apple Silicon cannot
run it locally). We use the **1B model in bfloat16** to stay within free-tier budget. Full
rationale, hardware findings, and licensing are documented in
**[docs/ACCESS_PATH.md](docs/ACCESS_PATH.md)**.

## Status

| Milestone | Description | Status |
|---|---|---|
| 0 | Scaffolding, primer, access-path research | ✅ done |
| 1 | Data layer (Findlay + ClinVar + reference) | ✅ done |
| 2 | Core zero-shot scoring engine | ✅ done (AUROC 0.74, matches published ~0.73) |
| 3 | Validation + honesty layer | ✅ done (see [RESULTS.md](RESULTS.md)) |
| 4 | Embedding-based classifier | ⏸️ deferred (engine ready; paused on cloud budget) |
| 5 | Explanation layer | ✅ done (trust-aware per-variant explanations) |
| 6 | Demo app + packaging | ⬜ next |

## Quickstart

```bash
make setup     # create venv + install local deps
make lock      # freeze exact versions for reproducibility
make help      # see all milestone entry points
```

(Data/scoring/validation targets come online as each milestone lands.)

## Repository layout

```
src/gvep/          installable package
  data/            dataset fetchers + loaders (Milestone 1)
  scoring/         Evo 2 / Modal delta-likelihood engine (Milestone 2)
  analysis/        metrics, calibration, honesty layer (Milestone 3+)
  utils/           seeding, config helpers
data/              raw/ processed/ cache/  (gitignored; regenerable)
results/           figures/ metrics/       (gitignored; regenerable)
docs/              ACCESS_PATH.md and other design notes
NOTES.md           running biology + ML learning log
PRIMER.md          newcomer's conceptual primer
```

## Attribution

- **Evo 2** — Arc Institute, NVIDIA, and collaborators. Model + weights are Apache-2.0.
  Brixi et al., *"Genome modeling and design across all domains of life with Evo 2"*, Nature (2026).
- **Findlay et al. (2018)** — *"Accurate classification of BRCA1 variants with saturation
  genome editing"*, Nature. Data via MaveDB `urn:mavedb:00000045-b`.
- **ClinVar** — NCBI.

## License

MIT (this project's code). See [LICENSE](LICENSE) for third-party component licenses.
