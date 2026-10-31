"""Dump the raw per-individual DE population fitness, generation by generation.

Why: the archived `de_trace_*.csv` only stores per-generation summary statistics
(mean/std/quantiles), so the population *distribution* cannot be re-plotted or
audited afterwards.  This script replays the differential-evolution run that
produced those traces and writes one row per evaluated individual:

    gen, idx, fitness

Two implementation details matter.

1. **Vectorised evaluation.**  `differential_evolution(..., vectorized=True)`
   predicts a whole generation (120 individuals) in one batch.  With a scalar
   objective each call costs ~1.6 s and a 31-generation run takes ~100 min; in
   batch mode the same run takes ~75 s.  Vectorised mode forces
   `updating="deferred"`, which is the variant the archived traces were produced
   with - the replay therefore has to use the same setting, otherwise the
   trajectory legitimately differs.

2. **Verification.**  After the replay the per-generation statistics are
   recomputed from the raw values and compared against `de_trace_<case>.csv`.
   Agreement to the stored precision is what makes the raw dump trustworthy.

Run on the analysis workstation:

    source ~/anaconda3/bin/activate autogluon
    cd ~/code/tcadproject/topcon_experiments
    python response_inverse_experiment/dump_de_population.py --case before
    python response_inverse_experiment/dump_de_population.py --case after
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from topcon_experiments.config import (  # noqa: E402
    ATHENA_FEATURES,
    BOUNDS_QUANTILE_HIGH,
    BOUNDS_QUANTILE_LOW,
    CURVE_DESCRIPTORS,
    DE_ATOL,
    DE_MAXITER,
    DE_MUTATION,
    DE_POPSIZE,
    DE_RECOMBINATION,
    DE_SEED,
    DE_TOL,
    LOG_FEATURES,
    META_JSON,
    MODEL2_FEATURES,
)
from topcon_experiments.common.autogluon_utils import safe_predict  # noqa: E402
from topcon_experiments.common.data import (  # noqa: E402
    get_athena_bounds,
    inverse_transform_target,
    load_raw_dataframe,
)
from topcon_experiments.common.forward import ForwardChain  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _as_rows(X: np.ndarray) -> np.ndarray:
    """Return an (N, 8) view of the array scipy hands to a vectorised objective.

    With `vectorized=True` scipy passes the whole population as `(S, N)` - one
    row per *parameter*, one column per individual - which is the transpose of
    what pandas wants.  A single recipe arrives as a flat length-8 vector.
    """
    arr = np.asarray(X, float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.shape[1] != len(ATHENA_FEATURES):
        arr = arr.T
    return arr


def predict_eff_batch(chain: ForwardChain, X: np.ndarray) -> np.ndarray:
    """Batched replica of `ForwardChain.predict_iv_single(x, 'iv_Eff')`.

    Same transform order as `_prepare_athena_frame` -> model 1 ->
    `inverse_transform_target` -> `_prepare_model2_frame` -> model 2.
    """
    raw = pd.DataFrame(_as_rows(X), columns=ATHENA_FEATURES)

    m1 = raw.copy()
    for col in LOG_FEATURES:
        if col in m1.columns:
            m1[col] = np.log(m1[col].astype(float))
    desc = {t: inverse_transform_target(t, safe_predict(chain._get_predictor1(t), m1))
            for t in CURVE_DESCRIPTORS}

    m2 = pd.concat([raw, pd.DataFrame(desc)], axis=1)
    for col in LOG_FEATURES:
        if col in m2.columns:
            m2[col] = np.log(m2[col].astype(float))
    return np.asarray(safe_predict(chain._get_predictor2("iv_Eff"),
                                   m2[MODEL2_FEATURES]), float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["before", "after"], required=True)
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()

    out = Path(args.out).resolve()
    meta = META_JSON if args.case == "before" else out / "meta_retrained.json"
    log(f"case={args.case}  meta={meta}")

    df = load_raw_dataframe()
    bounds = get_athena_bounds(df, BOUNDS_QUANTILE_LOW, BOUNDS_QUANTILE_HIGH)
    chain = ForwardChain(meta)

    # ---- sanity check: batched path must reproduce the scalar path ----------
    rng = np.random.default_rng(0)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    probe = lo + rng.random((5, len(lo))) * (hi - lo)
    vb = predict_eff_batch(chain, probe)
    vs = np.array([chain.predict_iv_single(x, "iv_Eff") for x in probe])
    log(f"batch vs scalar on 5 probes: max |d| = {np.abs(vb - vs).max():.3e}")

    gen_vals: list[float] = []
    raw: list[tuple[int, int, float]] = []
    state = {"last": 0, "gen": 0, "best": -np.inf}
    t0 = time.perf_counter()

    def objective(X):
        v = predict_eff_batch(chain, X)
        gen_vals.extend(v.tolist())
        state["best"] = max(state["best"], float(v.max()))
        return -v

    def callback(_xk, convergence=0.0):
        chunk = np.asarray(gen_vals[state["last"]:], float)
        if chunk.size:
            for i, v in enumerate(chunk):
                raw.append((state["gen"], i, float(v)))
            state["last"] = len(gen_vals)
            log(f"  gen {state['gen']:3d}: n={chunk.size} mean={chunk.mean():.4f} "
                f"best={chunk.max():.4f} p05={np.percentile(chunk,5):.4f} "
                f"p95={np.percentile(chunk,95):.4f} evals={len(gen_vals)} "
                f"({time.perf_counter()-t0:.0f}s)")
            state["gen"] += 1
        return False

    from scipy.optimize import differential_evolution

    log(f"replaying DE: maxiter={DE_MAXITER} popsize={DE_POPSIZE} seed={DE_SEED} "
        f"vectorized=True updating=deferred")
    res = differential_evolution(objective, bounds=bounds, strategy="best1bin",
                                 maxiter=DE_MAXITER, popsize=DE_POPSIZE,
                                 mutation=DE_MUTATION, recombination=DE_RECOMBINATION,
                                 seed=DE_SEED, tol=DE_TOL, atol=DE_ATOL,
                                 polish=False, disp=False, callback=callback,
                                 vectorized=True)
    log(f"done: {res.nfev} evals, best={state['best']:.4f}%, "
        f"{time.perf_counter()-t0:.0f}s")

    pop = pd.DataFrame(raw, columns=["gen", "idx", "fitness"])
    f_pop = out / f"de_population_{args.case}.csv"
    pop.to_csv(f_pop, index=False, float_format="%.6g")
    log(f"raw population -> {f_pop}  ({len(pop)} rows, {pop.gen.nunique()} generations)")

    rows, cum = [], -np.inf
    for g, sub in pop.groupby("gen", sort=True):
        v = sub.fitness.values
        cum = max(cum, v.max())
        rows.append({"gen": int(g), "n_evals": len(v), "best": cum,
                     "gen_best": v.max(), "gen_worst": v.min(),
                     "mean": v.mean(), "std": v.std(ddof=0),
                     "median": np.median(v),
                     "p05": np.percentile(v, 5), "p25": np.percentile(v, 25),
                     "p75": np.percentile(v, 75), "p95": np.percentile(v, 95)})
    stat = pd.DataFrame(rows)
    stat.to_csv(out / f"de_trace_from_raw_{args.case}.csv", index=False,
                float_format="%.6g")

    old_p = out / f"de_trace_{args.case}.csv"
    if old_p.exists():
        old = pd.read_csv(old_p)
        old = old[(old["gen"].notna()) & (old["n_evals"] > 0)].dropna(subset=["mean"])
        m = min(len(old), len(stat))
        dd = {k: float(np.abs(old[k].values[:m] - stat[k].values[:m]).max())
              for k in ("best", "mean", "p05", "p95", "median")}
        dd["n_evals"] = int(np.abs(old.n_evals.values[:m] - stat.n_evals.values[:m]).max())
        dd["gens_compared"] = m
        log(f"replay vs archived trace: {json.dumps(dd)}")
        (out / f"de_replay_check_{args.case}.json").write_text(
            json.dumps(dd, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
