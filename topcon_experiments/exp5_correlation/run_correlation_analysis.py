"""Exp5 entry: physical Pearson + LLM semantic similarity via local Ollama embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.config import EXP5_OUT, OLLAMA_EMBEDDING_MODEL
from topcon_experiments.exp5_correlation.llm_texts import collect_all_text_items
from topcon_experiments.exp5_correlation.ollama_embedding import check_ollama_embedding
from topcon_experiments.exp5_correlation.physical_correlation import run_physical_correlation
from topcon_experiments.exp5_correlation.semantic_similarity import run_semantic_panels


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Exp5: Pearson correlation + LLM semantic similarity (Ollama qwen3-embedding).",
    )
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Only compute physical Pearson matrices (no Ollama).",
    )
    parser.add_argument(
        "--skip-physical",
        action="store_true",
        help="Only compute LLM semantic similarity.",
    )
    parser.add_argument(
        "--topk-within",
        type=int,
        default=5,
        help="Top-K formulas within iv_Eff for intra-target similarity panel.",
    )
    args = parser.parse_args(argv)

    setup_runtime()
    apply_plot_style()
    EXP5_OUT.mkdir(parents=True, exist_ok=True)
    log(f"Exp5 correlation analysis -> {EXP5_OUT}")

    if not args.skip_physical:
        log("Computing physical Pearson correlation matrices...")
        phys = run_physical_correlation()
        log(f"  physical outputs: {len(phys)} files")

    if not args.skip_semantic:
        log(f"Checking Ollama embedding model ({OLLAMA_EMBEDDING_MODEL})...")
        check_ollama_embedding()
        items = collect_all_text_items(topk_within=args.topk_within)
        log(f"Collected {len(items)} LLM texts across panels")
        sem = run_semantic_panels(items)
        log(f"  semantic outputs: {len(sem)} files")

    log("Exp5 done.")


if __name__ == "__main__":
    main()
