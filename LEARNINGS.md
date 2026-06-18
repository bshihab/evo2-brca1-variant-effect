# LEARNINGS

A plain-language reflection on what this project taught me — biology, ML, and engineering.
Written as a learning log; see [PRIMER.md](PRIMER.md) for the gentler conceptual intro and
[RESULTS.md](RESULTS.md) for the findings.

## The domain (genomics)

- **DNA is a 4-letter language (A/C/G/T)**, and a *variant* is a one-letter change. The clinical
  problem is that most variants found in cancer-risk genes like **BRCA1** are **Variants of
  Uncertain Significance (VUS)** — nobody yet knows if they're harmful.
- **BRCA1 is the canonical benchmark** because Findlay et al. (2018) experimentally measured the
  effect of ~3,900 of its variants — a rare, large, unbiased answer key.
- **Coordinates are treacherous.** Genome builds (hg19 vs GRCh38) shift positions; BRCA1 is on the
  minus strand. A single integrity check — "does the reference allele match the genome?" — caught
  all of this at once (it came back 100%, which validated everything downstream).

## The method (foundation models)

- **Evo 2 is a "language model for DNA"** — a next-letter predictor that learned what real DNA
  looks like. We never trained it on disease labels (that's **zero-shot**).
- **Delta-likelihood scoring:** score the reference vs. the mutated sequence; the difference
  (`delta`) estimates how disruptive the change is. The same trick protein language models use,
  applied at DNA scale (so it can reason about non-coding/splice variants too).
- **"Looks unusual" is only a proxy for "is harmful."** That gap is the root of every error: the
  model misses harmful-but-normal-looking variants and false-alarms on harmless-but-weird ones —
  exactly like a spell-checker missing "their/there" but flagging "colour".

## The evaluation (this is the real skill)

- **AUROC ≠ accuracy.** AUROC is a *ranking* score: given a harmful and a harmless variant, how
  often does the model score the harmful one worse? (~74% for us.) It says nothing about a single
  variant — it's a group-level grade.
- **Aggregate numbers lie.** Our 0.74 was inflated by easy between-category structure; within the
  clinically-central **missense** category it was only **0.60**. Always stratify.
- **Class imbalance hides failure.** With 23% positives, a good AUROC coexisted with ~3 false
  alarms per true hit at usable sensitivity. AUPRC and precision-at-sensitivity tell that story.
- **Calibration is separate from ranking** — and easy to get wrong. Our first calibrator collapsed
  to the base rate because we forgot to standardize a tiny-magnitude feature (its Brier was exactly
  the no-skill value — a useful tell). A model can rank well yet output meaningless probabilities.
- **Benchmark honestly.** The 1B zero-shot score was beaten by CADD and even plain conservation
  (phyloP) on this dataset. Reporting that — with the fair caveats (budget model; zero-shot vs.
  supervised) — is more valuable than a flattering cherry-pick.
- **Separate *prediction* from *confidence*.** The prediction comes from the variant's score; the
  confidence comes from the model's measured track record on that *category*. A responsible tool
  refuses to claim a confidence the data can't support ("not assessable").

## The engineering

- **Apple Silicon can't run Evo 2** (CUDA-only). Real inference goes on a cloud GPU (Modal).
- **GPU environments are a version puzzle.** The working image needed Python 3.12 + CUDA 12.9 +
  Transformer-Engine 2.3.0 + flash-attn + CA certs — each constraint discovered by reading the
  solver's error messages. **FP8 runs fine on a cheap L4 (Ada)** — no H100 needed.
- **Make long jobs resilient.** Runs died to a free-tier **budget cap** mid-compute (not, as first
  assumed, the laptop sleeping). The fix: **persist results server-side + `--detach` + checkpoint
  in chunks**, so nothing is lost and work resumes. A real lesson in designing for failure.
- **Smoke-test before you spend.** A 4-sequence dry run caught an SSL bug and validated the layer
  name *before* committing to a ~75-minute, real-money run.

## If I did it again

- Extract **embeddings during the scoring pass** (one GPU run, not two).
- **Checkpoint from day one**, not after losing work to a budget cap.
- Try the **7B model** — likely a large accuracy gain over 1B, the dominant limitation here.
