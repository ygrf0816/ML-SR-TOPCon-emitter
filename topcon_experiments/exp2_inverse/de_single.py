"""Single-objective differential evolution inverse optimization."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import get_athena_bounds, load_raw_dataframe
from topcon_experiments.common.forward import load_forward_chain
from topcon_experiments.common.inverse_utils import (
    get_de_iv_scale_factors,
    multistart_refine_efficiency,
    scale_iv_for_de,
)
from topcon_experiments.common.log_utils import log, log_step, setup_runtime
from topcon_experiments.common.plot_utils import plot_trace, save_csv
from topcon_experiments.config import (
    ATHENA_FEATURES,
    BOUNDS_QUANTILE_HIGH,
    BOUNDS_QUANTILE_LOW,
    DE_ATOL,
    DE_IV_SCALE_ENABLED,
    DE_LOCAL_REFINE,
    DE_LOG_EVERY_EVALS,
    DE_MAXITER,
    DE_MUTATION,
    DE_POPSIZE,
    DE_RECOMBINATION,
    DE_SEED,
    DE_TOL,
    IV_TARGET_LABELS,
    IV_TARGETS,
    META_JSON,
    OUTPUT_ROOT,
)


EXP2_OUT = OUTPUT_ROOT / "exp2_inverse"


def _estimate_eval_budget(popsize: int, maxiter: int, n_params: int) -> int:
    """SciPy DE: ~(maxiter+1) * popsize * n_params evaluations."""
    return popsize * n_params * (maxiter + 1)


def _scaled_metric(chain, x: np.ndarray, metric: str) -> float:
    raw = float(chain.predict_iv_single(x, metric))
    return scale_iv_for_de(metric, raw)


def run_de_for_metric(
    metric: str,
    chain,
    bounds: list[tuple[float, float]],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run DE and record per-generation best/mean/std (no custom initial population)."""
    popsize = DE_POPSIZE
    trace_rows: list[dict] = []
    gen_vals: list[float] = []
    best_so_far = -np.inf
    last_flush = 0
    n_eval = 0
    t0 = time.perf_counter()
    budget = _estimate_eval_budget(popsize, DE_MAXITER, len(ATHENA_FEATURES))
    scale = get_de_iv_scale_factors().get(metric, 1.0)

    log(
        f"  DE config: maxiter={DE_MAXITER}, popsize={popsize}, "
        f"tol={DE_TOL}, atol={DE_ATOL}; ~{budget} objective evals expected"
    )
    if DE_IV_SCALE_ENABLED:
        log(f"  IV scale for {metric}: {scale:.6f} (scaled = surrogate * scale)")
    log(f"  Each eval runs model1(9) + model2(1) for {metric}")

    def _flush_generation(gen: int) -> None:
        # NOTE on sampling口径 (sample definition), three facts that matter:
        #   (a) `best_so_far` is the CUMULATIVE running max over EVERY objective
        #       evaluation so far -> monotone non-decreasing across generations.
        #       It is NOT a statistic of `chunk`.
        #   (b) `chunk` holds only the individuals evaluated *within the current
        #       generation*. SciPy evaluates NP = popsize * n_params trial vectors
        #       per generation; the very first chunk additionally contains the NP
        #       initial-population evaluations, so gen 0 has 2*NP samples. The
        #       `n_evals` column makes this explicit.
        #   (c) `mean +/- 1 std` is a DISPERSION measure, NOT a range. For the
        #       strongly left-skewed populations of a converged DE (most members
        #       near the optimum, a few stragglers far below) it is mathematically
        #       possible and in fact common that mean + std > best. Do not use it
        #       as an interval: use the quantiles / gen_best / gen_worst below.
        nonlocal last_flush
        chunk = gen_vals[last_flush:]
        if not chunk:
            return
        arr = np.asarray(chunk, dtype=float)
        q05, q25, q50, q75, q95 = (
            float(v) for v in np.percentile(arr, [5, 25, 50, 75, 95])
        )
        trace_rows.append({
            "gen": gen,
            "best": float(best_so_far),      # cumulative running max (all evals)
            "n_evals": int(arr.size),        # samples behind the stats below
            "gen_best": float(np.max(arr)),  # best within THIS generation
            "gen_worst": float(np.min(arr)),  # worst within THIS generation
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),  # dispersion, NOT a range bound
            "median": q50,
            "p05": q05,
            "p25": q25,
            "p75": q75,
            "p95": q95,
        })
        last_flush = len(gen_vals)

    def tracked_objective(x: np.ndarray) -> float:
        nonlocal best_so_far, n_eval
        n_eval += 1
        val = _scaled_metric(chain, x, metric)
        gen_vals.append(val)
        best_so_far = max(best_so_far, val)

        if n_eval == 1:
            log("  first objective eval done (initial population starting)...")
        elif DE_LOG_EVERY_EVALS > 0 and n_eval % DE_LOG_EVERY_EVALS == 0:
            elapsed = time.perf_counter() - t0
            rate = n_eval / max(elapsed, 1e-6)
            eta = (budget - n_eval) / max(rate, 1e-6)
            log(
                f"  eval {n_eval}/{budget}: current={val:.4f}, "
                f"best={best_so_far:.4f}, {elapsed:.0f}s elapsed, ETA ~{eta:.0f}s"
            )
        return -val

    def callback(_xk, convergence=0.0) -> bool:
        _flush_generation(len(trace_rows))
        if trace_rows:
            row = trace_rows[-1]
            elapsed = time.perf_counter() - t0
            log(
                f"  gen {row['gen']:3d}: best={row['best']:.4f}, "
                f"mean={row['mean']:.4f}, std={row['std']:.4f}, "
                f"evals={len(gen_vals)}, {elapsed:.0f}s, convergence={convergence:.2e}"
            )
        return False

    log("  launching differential_evolution...")
    result = differential_evolution(
        tracked_objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=DE_MAXITER,
        popsize=popsize,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        seed=DE_SEED,
        tol=DE_TOL,
        atol=DE_ATOL,
        polish=False,
        disp=False,
        callback=callback,
    )
    _flush_generation(len(trace_rows))
    x_best = result.x
    de_elapsed = time.perf_counter() - t0
    log(
        f"  DE finished: {result.nfev} evals, {len(trace_rows)} generations, "
        f"success={result.success}, message={result.message!r}, {de_elapsed:.0f}s"
    )

    if DE_LOCAL_REFINE:
        log("  local L-BFGS-B polish...")
        t1 = time.perf_counter()

        def neg_obj(x):
            return -_scaled_metric(chain, x, metric)

        res_local = minimize(
            neg_obj,
            x_best,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200},
        )
        if -res_local.fun > -neg_obj(x_best):
            x_best = res_local.x
            log(f"  local refine improved to {-res_local.fun:.4f} ({time.perf_counter()-t1:.0f}s)")
        else:
            log(f"  local refine kept DE solution ({time.perf_counter()-t1:.0f}s)")

    trace_df = pd.DataFrame(trace_rows)
    if not trace_df.empty:
        trace_df["gen"] = np.arange(len(trace_df), dtype=int)
    return x_best, trace_df


def save_best_result(x: np.ndarray, chain, metric: str) -> pd.DataFrame:
    params = {col: float(x[i]) for i, col in enumerate(ATHENA_FEATURES)}
    pred = chain.predict_all(x)
    row = {**params, **pred}
    if DE_IV_SCALE_ENABLED:
        for m in IV_TARGETS:
            if m in row:
                row[f"{m}_surrogate"] = row[m]
                row[m] = scale_iv_for_de(m, float(row[m]))
    row["optimized_metric"] = metric
    row["optimized_metric_label"] = IV_TARGET_LABELS.get(metric, metric)
    return pd.DataFrame([row])


def optimize_iv_eff(
    chain,
    df: pd.DataFrame,
    bounds: list[tuple[float, float]],
) -> tuple[np.ndarray, pd.DataFrame]:
    """DE (random init) + optional multistart local refine for iv_Eff."""
    x_de, trace_df = run_de_for_metric("iv_Eff", chain, bounds)
    de_val = _scaled_metric(chain, x_de, "iv_Eff")

    log("  multistart local refine from top-10 dataset samples...")
    t0 = time.perf_counter()
    x_ms, ms_val = multistart_refine_efficiency(chain, df, bounds, n_starts=10)
    log(f"  multistart done in {time.perf_counter()-t0:.0f}s: best={ms_val:.4f}%")
    log(f"iv_Eff DE={de_val:.4f}%, multistart={ms_val:.4f}%")
    x_best = x_ms if ms_val >= de_val else x_de
    return x_best, trace_df


def main() -> None:
    setup_runtime()
    EXP2_OUT.mkdir(parents=True, exist_ok=True)
    log("Loading forward chain (preloading all models)...")
    chain = load_forward_chain(META_JSON)
    log("Forward chain ready.")
    df = load_raw_dataframe()

    global_bounds = get_athena_bounds(df, BOUNDS_QUANTILE_LOW, BOUNDS_QUANTILE_HIGH)
    if DE_IV_SCALE_ENABLED:
        scales = get_de_iv_scale_factors()
        log("IV calibration scales (surrogate -> adjusted simulation):")
        for m in IV_TARGETS:
            log(f"  {IV_TARGET_LABELS.get(m, m)}: x{scales[m]:.6f}")
    log(
        f"Dataset iv_Eff max={df['iv_Eff'].max():.4f}%; "
        f"DE maxiter={DE_MAXITER}, popsize={DE_POPSIZE}, random init (no Eff seeding)"
    )

    for mi, metric in enumerate(IV_TARGETS, start=1):
        label = IV_TARGET_LABELS.get(metric, metric)
        log_step(mi, len(IV_TARGETS), f"DE maximize {label} ({metric})")
        t_task = time.perf_counter()
        if metric == "iv_Eff":
            x_best, trace_df = optimize_iv_eff(chain, df, global_bounds)
        else:
            x_best, trace_df = run_de_for_metric(metric, chain, global_bounds)
        plot_trace(trace_df, label, EXP2_OUT / f"de_trace_{metric}")
        best_df = save_best_result(x_best, chain, metric)
        save_csv(best_df, EXP2_OUT / f"de_best_{metric}.csv")
        log(
            f"  {len(trace_df)} generations recorded; best {label}={best_df[metric].iloc[0]:.4f}; "
            f"task time {time.perf_counter()-t_task:.0f}s"
        )
        log(best_df.to_string(index=False))

    from topcon_experiments.exp2_inverse.oracle_report import main as oracle_main
    oracle_main()
    log(f"DE results saved to {EXP2_OUT}")


if __name__ == "__main__":
    main()
