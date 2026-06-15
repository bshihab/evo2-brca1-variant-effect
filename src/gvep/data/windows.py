"""Sequence-window construction — the heart of the data layer.

For each variant we build two equal-length DNA windows centered on the variant:
  * the REFERENCE window (genome as-is)
  * the VARIANT window (single base swapped to the alternate allele)

These are what Evo 2 scores in Milestone 2 (delta = var_log_prob - ref_log_prob).

This mirrors the Evo 2 BRCA1 notebook exactly so our results are comparable:
  p = pos - 1                      # 1-based genomic coord -> 0-based string index
  window = seq[p-4096 : p+4096]    # 8192 bp centered on the variant
  center = min(4096, p)            # where the variant sits within the window
  var = window[:center] + alt + window[center+1:]

Strand note: BRCA1 is on the minus strand, but the Findlay reference/alt alleles are
given on the FORWARD genomic strand (they match the plus-strand chr17 sequence). Evo 2
was trained on both strands, so we feed forward-strand windows directly — no reverse
complement needed. The `ref == window[center]` check below is exactly what guarantees
our coordinates/alleles line up with the genome.
"""

from __future__ import annotations

from dataclasses import dataclass

from gvep.config import WINDOW_BP


@dataclass(frozen=True)
class Window:
    ref_seq: str          # reference window
    var_seq: str          # variant window (alt substituted)
    center: int           # index of the variant within the window
    ref_matches: bool     # did genome[center] == reported ref allele?
    genome_base: str      # what the genome actually has at center (for debugging)


def build_window(
    seq: str, pos: int, ref: str, alt: str, window: int = WINDOW_BP
) -> Window:
    """Build ref/variant windows for a 1-based SNV at `pos` on sequence `seq`.

    Does NOT raise on a ref mismatch — it records `ref_matches=False` so the caller
    can audit how many variants fail the integrity check rather than crashing on one.
    """
    p = pos - 1  # to 0-based
    start = max(0, p - window // 2)
    end = min(len(seq), p + window // 2)
    ref_seq = seq[start:end]

    center = min(window // 2, p)  # handles variants near the chromosome start
    genome_base = ref_seq[center] if 0 <= center < len(ref_seq) else ""
    var_seq = ref_seq[:center] + alt + ref_seq[center + 1 :]

    return Window(
        ref_seq=ref_seq,
        var_seq=var_seq,
        center=center,
        ref_matches=genome_base.upper() == ref.upper(),
        genome_base=genome_base,
    )
