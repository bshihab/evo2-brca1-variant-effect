"""Command-line entry point: one subcommand per milestone step.

Usage:
    python -m gvep.cli <command>

Commands are scaffolded now and implemented as each milestone lands. This keeps the
Makefile targets and the package in sync from day one.
"""

from __future__ import annotations

import argparse
import sys

from gvep import __version__
from gvep.utils.seed import set_seed


def _not_yet(name: str, milestone: str) -> int:
    print(f"[gvep] '{name}' is scaffolded but not implemented yet ({milestone}).")
    return 0


def cmd_data(_args: argparse.Namespace) -> int:
    """[Milestone 1] Fetch + build the variant datasets."""
    from gvep.data.build import run

    run()
    return 0


def cmd_score(_args: argparse.Namespace) -> int:
    """[Milestone 2] Run Evo 2 delta-likelihood scoring on Modal."""
    print(
        "Scoring runs on a Modal GPU, not locally. After `modal token new`, run:\n"
        "  modal run -m gvep.scoring.modal_app::smoke   # cheap validation first\n"
        "  modal run -m gvep.scoring.modal_app::main    # full dataset\n"
        "Then: python -m gvep.cli sanity                # plot the distributions"
    )
    return 0


def cmd_sanity(_args: argparse.Namespace) -> int:
    """[Milestone 2] Plot delta distributions + quick AUROC (after scoring)."""
    from gvep.analysis.sanity import run

    run()
    return 0


def cmd_validate(_args: argparse.Namespace) -> int:
    """[Milestone 3] Compute metrics + honesty/calibration analysis."""
    from gvep.analysis.honesty import run

    run()
    return 0


def cmd_classify(_args: argparse.Namespace) -> int:
    """[Milestone 4] Train embedding classifier; compare to zero-shot."""
    from gvep.analysis.classifier import run

    run()
    return 0


def cmd_benchmark(_args: argparse.Namespace) -> int:
    """[Milestone 7] Benchmark Evo 2 vs AlphaMissense on BRCA1 missense."""
    from gvep.analysis.alphamissense import run

    run()
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """[Milestone 5] Plain-language, trust-aware explanation for a variant."""
    from gvep.explain import explain_variant, format_explanation, run_demo

    if args.pos and args.ref and args.alt:
        e = explain_variant(args.pos, args.ref, args.alt)
        if e is None:
            print(f"Variant chr17:{args.pos} {args.ref}>{args.alt} not in the scored set.")
            return 1
        print(format_explanation(e))
    else:
        run_demo()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gvep",
        description="Genomic Variant Effect Prediction (BRCA1) — research/triage POC, "
        "NOT a clinical diagnostic.",
    )
    parser.add_argument("--version", action="version", version=f"gvep {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("data", help="[M1] fetch + build datasets").set_defaults(func=cmd_data)
    sub.add_parser("score", help="[M2] how to run Evo 2 scoring on Modal").set_defaults(func=cmd_score)
    sub.add_parser("sanity", help="[M2] plot delta distributions + quick AUROC").set_defaults(func=cmd_sanity)
    sub.add_parser("validate", help="[M3] metrics + honesty layer").set_defaults(func=cmd_validate)
    sub.add_parser("classify", help="[M4] embedding classifier vs zero-shot").set_defaults(func=cmd_classify)
    sub.add_parser("benchmark", help="[M7] Evo 2 vs AlphaMissense (missense)").set_defaults(func=cmd_benchmark)
    p_ex = sub.add_parser("explain", help="[M5] trust-aware explanation for a variant")
    p_ex.add_argument("--pos", type=int, help="hg19 position on chr17")
    p_ex.add_argument("--ref", type=str, help="reference allele")
    p_ex.add_argument("--alt", type=str, help="alternate allele")
    p_ex.set_defaults(func=cmd_explain)
    return parser


def main(argv: list[str] | None = None) -> int:
    set_seed()  # deterministic from the very first command
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
