"""Run DE for a single IV metric (default: iv_Eff / PCE)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import get_athena_bounds, load_raw_dataframe
from topcon_experiments.common.forward import load_forward_chain
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import plot_trace, save_csv
from topcon_experiments.config import (
    BOUNDS_QUANTILE_HIGH,
    BOUNDS_QUANTILE_LOW,
    IV_TARGET_LABELS,
    META_JSON,
    OUTPUT_ROOT,
)
from topcon_experiments.exp2_inverse.de_single import run_de_for_metric, save_best_result

EXP2_OUT = OUTPUT_ROOT / "exp2_inverse"


def main(metric: str = "iv_Eff") -> None:
    setup_runtime()
    EXP2_OUT.mkdir(parents=True, exist_ok=True)
    log("Loading forward chain...")
    chain = load_forward_chain(META_JSON)
    bounds = get_athena_bounds(load_raw_dataframe(), BOUNDS_QUANTILE_LOW, BOUNDS_QUANTILE_HIGH)
    label = IV_TARGET_LABELS.get(metric, metric)
    log(f"DE maximize {label} ({metric})")
    x_best, trace_df = run_de_for_metric(metric, chain, bounds)
    plot_trace(trace_df, label, EXP2_OUT / f"de_trace_{metric}")
    save_csv(save_best_result(x_best, chain, metric), EXP2_OUT / f"de_best_{metric}.csv")
    log(f"Done -> {EXP2_OUT}")


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "iv_Eff"
    main(m)
