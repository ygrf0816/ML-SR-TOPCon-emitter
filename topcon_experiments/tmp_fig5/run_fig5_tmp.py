"""Run paper-ready Fig.5 plots (joint SR + LLM interpretation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.config import EXP5_OUT
from topcon_experiments.tmp_fig5.feature_participation import run as run_participation
from topcon_experiments.tmp_fig5.paper_fig5 import plot_composite_paper
from topcon_experiments.tmp_fig5.semantic_fig5_panels import run as run_semantic
from topcon_experiments.tmp_fig5.plot_physics_meaning_map import run as run_physics_meaning
from topcon_experiments.tmp_fig5.plot_paper_sr_insights import run as run_sr_insights
from topcon_experiments.tmp_fig5.plot_accuracy_comparison import run as run_accuracy_compare
from topcon_experiments.tmp_fig5.sr_llm_comparison import run as run_sr_llm
from topcon_experiments.tmp_fig5.utils import TMP_OUT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Paper Fig.5: joint SR vs LLM + feature roles + semantic/physical panels.",
    )
    parser.add_argument("--skip-semantic", action="store_true", help="Skip supplementary semantic copies")
    parser.add_argument("--skip-sr-llm", action="store_true")
    parser.add_argument("--skip-participation", action="store_true")
    args = parser.parse_args(argv)

    setup_runtime()
    apply_plot_style()
    TMP_OUT.mkdir(parents=True, exist_ok=True)
    log(f"Paper Fig.5 outputs -> {TMP_OUT}")

    if not args.skip_sr_llm:
        log("SR vs LLM-simplified (joint best per task)...")
        run_sr_llm(TMP_OUT)
        log("PySR / LLM / AutoGluon accuracy comparison...")
        run_accuracy_compare(TMP_OUT)
        log("SR paper insight figures (dumbbell, landscape, rose, ...)...")
        run_sr_insights(TMP_OUT)
        log("SR physical-meaning map (variables + keyword evidence)...")
        run_physics_meaning(TMP_OUT)

    if not args.skip_participation:
        log("Feature participation in best IV formulas...")
        run_participation(TMP_OUT)

    if not args.skip_semantic:
        if not (EXP5_OUT / "data" / "semantic_iv_athena_physics.csv").exists():
            log("exp5 semantic data missing — run: python3 -m topcon_experiments.run_exp5_correlation")
        else:
            log("Rendering supplementary semantic panels from exp5...")
            run_semantic(TMP_OUT)

    comp = plot_composite_paper(TMP_OUT)
    if comp:
        log(f"Composite figure -> {comp}")
    else:
        log("Composite skipped (run SR/LLM panels first)")

    log("Standalone panels for paper insertion:")
    for name in (
        "panel_sr_llm_iv_athena.png",
        "panel_complexity_reduction_iv.png",
        "panel_feature_participation_iv.png",
        "panel_sr_llm_all_tasks.png",
        "accuracy_sr_llm_autogluon_all.png",
        "sr_physics_meaning_map.png",
        "sr_insight_composite.png",
        "fig5_paper_composite.png",
    ):
        p = TMP_OUT / "plots" / name
        if p.exists():
            log(f"  {p}")
    log("Done.")


if __name__ == "__main__":
    main()
