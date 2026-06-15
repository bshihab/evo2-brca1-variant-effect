"""Central configuration: paths, constants, and the global random seed.

Keeping these in one place makes runs reproducible and avoids hard-coded paths
scattered across modules. Biological constants (gene coordinates, window size)
live here too so there's a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility -------------------------------------------------------
# A single seed used everywhere (numpy, python random, and any torch on the GPU
# side). See gvep.utils.seed.set_seed().
SEED: int = 42

# --- Paths -----------------------------------------------------------------
# Resolve relative to the repo root so things work regardless of CWD.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CACHE_DIR: Path = DATA_DIR / "cache"
RESULTS_DIR: Path = REPO_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
METRICS_DIR: Path = RESULTS_DIR / "metrics"

# --- Biology constants -----------------------------------------------------
# BRCA1 lives on chromosome 17, minus strand (GRCh38). Exact coordinates and the
# reference assembly are pinned in Milestone 1 once we fetch the genome region.
GENE: str = "BRCA1"
CHROMOSOME: str = "chr17"
STRAND: str = "-"

# Sequence window size (bp) centered on each variant, fed to Evo 2.
# Matches Evo 2 1B's 8192-token training context and NVIDIA's BRCA1 tutorial.
WINDOW_BP: int = 8192

# --- Model / inference -----------------------------------------------------
EVO2_MODEL: str = "evo2_1b_base"   # Apache-2.0; runs in bf16 (see docs/ACCESS_PATH.md)
EVO2_DTYPE: str = "bfloat16"       # no FP8 / Transformer Engine
