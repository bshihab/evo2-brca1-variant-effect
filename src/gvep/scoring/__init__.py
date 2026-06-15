"""Scoring engine (Milestone 2): Evo 2 delta-likelihood scoring on Modal.

Implemented in Milestone 2. Loads evo2_1b_base (bf16) on a cloud GPU, scores ref vs
variant windows, and computes delta = var_log_prob - ref_log_prob. See docs/ACCESS_PATH.md.
"""
