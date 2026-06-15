"""Reference genome: GRCh37 (hg19) chromosome 17.

BRCA1 is on chr17. The Findlay 2018 dataset reports coordinates in **hg19 (GRCh37)**,
so we MUST use a GRCh37 reference — using GRCh38 would silently shift every position and
break the ref-allele check. We reuse the exact chr17 FASTA bundled with the Evo 2 repo
(`GRCh37.p13_chr17.fna.gz`) so our windows match the canonical Evo 2 / BioNeMo pipeline.

See PRIMER.md (delta-likelihood) and docs/ACCESS_PATH.md for context.
"""

from __future__ import annotations

import functools
import gzip

from Bio import SeqIO

from gvep.config import RAW_DIR
from gvep.utils.io import download_file

# Bundled in the Evo 2 repo; verified directly downloadable (≈23 MB).
CHR17_URL = (
    "https://raw.githubusercontent.com/ArcInstitute/evo2/main/"
    "notebooks/brca1/GRCh37.p13_chr17.fna.gz"
)
CHR17_BYTES = 23_195_609  # exact size, used as a cheap integrity gate
CHR17_GZ = RAW_DIR / "GRCh37.p13_chr17.fna.gz"


def download_reference() -> "object":
    """Fetch the gzipped GRCh37 chr17 FASTA into data/raw/ (cached)."""
    return download_file(CHR17_URL, CHR17_GZ, expected_bytes=CHR17_BYTES)


@functools.lru_cache(maxsize=1)
def load_chr17_sequence() -> str:
    """Return the GRCh37 chr17 sequence as an uppercase string (cached in-process).

    Uppercasing normalizes any soft-masked (lowercase) bases so ref-allele comparisons
    are case-insensitive and consistent.
    """
    download_reference()
    with gzip.open(CHR17_GZ, "rt") as handle:
        record = next(SeqIO.parse(handle, "fasta"))
        seq = str(record.seq).upper()
    return seq
