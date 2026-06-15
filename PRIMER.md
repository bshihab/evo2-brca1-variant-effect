# PRIMER — Concepts for a Newcomer

This primer explains, in plain language, the four ideas this project rests on. It's written
for someone new to computational genomics (the author included). Where a claim is specific,
sources are listed at the bottom.

---

## 1. What is a genomic foundation model?

A **foundation model** is a large neural network trained on a huge amount of unlabeled data
with a simple self-supervised objective, producing a general-purpose model you can then apply
to many downstream tasks. In language, that objective is usually "predict the next token."

A **genomic** foundation model does the same thing, but the "language" is **DNA**. Instead of
words, the tokens are nucleotides — the letters **A, C, G, T**. The model reads enormous
amounts of real genomic sequence and learns to predict nucleotides from their surrounding
context. In doing so it implicitly learns the "grammar" of biology: what real genes, regulatory
regions, and protein-coding sequences tend to look like.

**Evo 2** (the model we use) is one of the largest such models. It was trained on ~9 trillion
nucleotides spanning all domains of life (bacteria, archaea, eukaryotes), at **single-nucleotide
resolution**, with a context window up to ~1 million bases in its largest configuration. It uses
a custom architecture called **StripedHyena 2** (not a standard Transformer) to handle those very
long sequences efficiently. It comes in 1B, 7B, 20B, and 40B parameter sizes.

The key intuition: **a model that's good at predicting "what DNA looks like" has implicitly
learned which sequences are biologically plausible.** That's the lever we pull for variant effect
prediction.

---

## 2. What is zero-shot variant effect prediction?

A **variant** is a change to the DNA sequence relative to a reference — e.g. position 41,276,045
on chromosome 17 is normally a `G`, but in some people it's an `A`. A **single-nucleotide variant
(SNV)** changes exactly one letter.

The **effect** we care about: does this change disrupt the gene's function (potentially
**pathogenic**), or is it harmless (**benign**)? Most discovered variants are **Variants of
Uncertain Significance (VUS)** — we genuinely don't know yet.

**Zero-shot** means we make this prediction **without training the model on any
variant-effect labels.** We don't show Evo 2 examples of "harmful" vs "harmless" variants and let
it learn the mapping. Instead, we use the model exactly as it was pre-trained — as a pure DNA
likelihood estimator — and read the effect out of its predictions. "Zero-shot" = "zero
task-specific training examples."

Why is this remarkable? Because if it works, a model that *never saw a single clinical label*
can still flag likely-harmful variants, just from having learned what healthy DNA looks like.

(Later, in Milestone 4, we relax this: we train a small classifier on the model's internal
**embeddings**. That's no longer zero-shot — it's a supervised upgrade we compare *against* the
zero-shot baseline to measure the gain.)

---

## 3. What is delta-likelihood scoring?

This is the actual mechanism. It's simpler than it sounds.

A trained DNA language model can assign a **likelihood** (really, a **log-likelihood**) to any
sequence — a number for "how plausible is this stretch of DNA, according to everything I learned?"
Real, functional sequences tend to score high; sequences that look broken or unnatural score low.

For a given variant, we:

1. Take a window of the **reference** sequence around the variant position → ask the model for its
   log-likelihood → call it `ref_log_prob`.
2. Take the **same window but with the single variant nucleotide substituted in** → ask the model
   → call it `var_log_prob`.
3. Compute the **delta**:

   ```
   delta = var_log_prob - ref_log_prob
   ```

Interpretation:

- **delta ≈ 0** → the variant barely changes how plausible the sequence looks → probably tolerated
  → likely benign.
- **delta strongly negative** → swapping in this nucleotide makes the sequence look much *less*
  like real, functional DNA → the model "expected" something else there → likely disruptive /
  pathogenic.

So the model never outputs "pathogenic" directly. We *infer* pathogenicity from how much the
single-letter change lowers the sequence's likelihood. This is the same logic behind protein
language-model variant scoring (e.g. ESM) and missense predictors — applied here at the DNA level,
which lets it also reason about non-coding and splicing effects that protein-only models miss.

A practical detail we'll handle in Milestone 1–2: BRCA1 is on the **minus strand** of chromosome
17, and windows must be built carefully so the reference allele actually matches the genome.
Getting coordinates and strand right is half the battle in this field.

---

## 4. Why is BRCA1 the canonical test case?

**BRCA1** is a tumor-suppressor gene; certain loss-of-function variants substantially raise the
risk of hereditary breast and ovarian cancer. It matters clinically *and* it happens to be the
best benchmark available, for one specific reason:

In 2018, **Findlay et al.** used **saturation genome editing** to experimentally measure the
functional effect of **~4,000 single-nucleotide variants** across the critical regions of BRCA1.
"Saturation" means they didn't cherry-pick — they tested essentially *every possible* SNV in those
regions. Each variant got a **function score** from a real cell-viability assay, and these scores
fall into a clean bimodal split: **loss-of-function (LOF)** vs **functional (FUNC)**, with some
**intermediate (INT)** in between.

That gives us a rare thing: a large, unbiased, experimentally-grounded **ground-truth** label set.
We can run Evo 2's zero-shot deltas across the same ~3,893 variants and ask: *do the model's scores
separate the experimentally-confirmed LOF variants from the functional ones?* It's the closest
thing the field has to a clean exam with an answer key — which is exactly why both the Evo 2 paper
and NVIDIA's own tutorial use BRCA1 as the demonstration.

The honest caveat (which Milestone 3 is built around): doing well on this one well-characterized
gene does **not** prove the method generalizes to other genes, to non-coding regions, or to the
messy reality of the clinic. Measuring *where it breaks* is the point of this project.

---

## Sources

- Brixi, G., et al. *"Genome modeling and design across all domains of life with Evo 2."*
  Nature (2026). Preprint: bioRxiv 2025.02.18.638918.
  <https://www.nature.com/articles/s41586-026-10176-5>
- Arc Institute — Evo 2 overview and model cards.
  <https://arcinstitute.org/tools/evo> · <https://github.com/ArcInstitute/evo2>
- Findlay, G.M., et al. *"Accurate classification of BRCA1 variants with saturation genome
  editing."* Nature 562, 217–222 (2018). Data: MaveDB `urn:mavedb:00000045-b`.
  <https://www.biorxiv.org/content/10.1101/294520v1>
- NVIDIA BioNeMo — *"Zero-shot prediction of BRCA1 variant effects with Evo 2"* tutorial.
  <https://docs.nvidia.com/bionemo-framework/2.5/user-guide/examples/bionemo-evo2/zeroshot_brca1/>
