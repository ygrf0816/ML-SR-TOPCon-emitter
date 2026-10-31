"""Multi-seed inverse design, run for both the original and the retrained Model 2.

Purpose (reviewer point 1 and 4): show that different optimisation starts land on
the same objective value - i.e. the surrogate is internally consistent - and then
test that value against TCAD for every recipe.

    python run_multiseed.py --case before --seeds 0-9
    python run_multiseed.py --case after  --seeds 0-9

Writes, under `out/multiseed/`:
    ms_recipes_<case>.csv       8 process parameters + all surrogate outputs
    ms_trace_<case>_s<seed>.csv  per-generation trace for each seed
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
sys.path.insert(0, str(HERE))

import config as C
from fwd import ForwardChain, get_athena_bounds
from autogluon.tabular import TabularPredictor  # noqa: F401  (import side effects)

F1 = list(C.ATHENA_FEATURES)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def de(chain: ForwardChain, bounds, seed: int, out: Path, tag: str,
       maxiter: int, popsize: int) -> tuple[np.ndarray, pd.DataFrame]:
    from scipy.optimize import differential_evolution

    vals: list[float] = []
    trace: list[dict] = []
    last, best = 0, -np.inf

    def score(X: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(X, columns=F1)
        D = chain.predict_curve_frame(frame)
        return chain.predict_iv_frame(frame, D)["iv_Eff"].to_numpy()

    def obj(x):
        nonlocal best
        X = np.asarray(x, float)
        if X.ndim == 1:
            v = float(score(X.reshape(1, -1))[0])
            vals.append(v)
            best = max(best, v)
            return -v
        if X.shape[1] != len(F1) and X.shape[0] == len(F1):
            X = X.T
        v = score(X)
        vals.extend(v.tolist())
        best = max(best, float(v.max()))
        return -v

    def cb(_xk, convergence=0.0):
        nonlocal last
        chunk = np.asarray(vals[last:], float)
        if chunk.size:
            trace.append({"gen": len(trace), "best": float(best),
                          "n_evals": int(chunk.size), "gen_best": float(chunk.max()),
                          "mean": float(chunk.mean()), "std": float(chunk.std(ddof=0))})
            last = len(vals)
        return False

    res = differential_evolution(obj, bounds=bounds, strategy="best1bin",
                                 maxiter=maxiter, popsize=popsize, mutation=C.DE_MUTATION,
                                 recombination=C.DE_RECOMBINATION, seed=seed,
                                 tol=C.DE_TOL, atol=C.DE_ATOL, polish=False, disp=False,
                                 vectorized=True, callback=cb)
    final = float(score(np.asarray(res.x, float).reshape(1, -1))[0])
    if final > best:
        best = final
        trace.append({"gen": len(trace), "best": best, "n_evals": 1,
                      "gen_best": best, "mean": best, "std": 0.0})
    pd.DataFrame(trace).to_csv(out / f"ms_trace_{tag}_s{seed:02d}.csv",
                               index=False, float_format="%.6g")
    log(f"  [{tag}] seed {seed:2d}: best={best:.4f}%")
    return res.x, pd.DataFrame(trace)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["before", "after"])
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--maxiter", type=int, default=C.DE_MAXITER)
    ap.add_argument("--popsize", type=int, default=C.DE_POPSIZE)
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()

    lo, _, hi = args.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    out = Path(args.out).resolve() / "multiseed"
    out.mkdir(parents=True, exist_ok=True)

    meta = C.META_JSON if args.case == "before" else Path(args.out) / "meta_retrained.json"
    log(f"case={args.case} meta={meta} seeds={seeds}")
    chain = ForwardChain(meta)

    df = pd.read_csv(C.DATA_CSV)
    df = df.loc[(df[list(set(C.LOG_FEATURES))] > 0).all(axis=1)].copy()
    bounds = get_athena_bounds(df, C.BOUNDS_QUANTILE_LOW, C.BOUNDS_QUANTILE_HIGH)

    rows = []
    for s in seeds:
        x, tr = de(chain, bounds, s, out, args.case, args.maxiter, args.popsize)
        rec = {"case": args.case, "seed": s,
               **{c: float(x[i]) for i, c in enumerate(F1)}}
        rec.update(chain.predict_all(x))
        rows.append(rec)
        pd.DataFrame(rows).to_csv(out / f"ms_recipes_{args.case}.csv",
                                  index=False, float_format="%.6g")

    t = pd.DataFrame(rows)
    log(f"[{args.case}] surrogate PCE across {len(t)} seeds: "
        f"mean {t.iv_Eff.mean():.4f} sd {t.iv_Eff.std(ddof=1):.4f} "
        f"min {t.iv_Eff.min():.4f} max {t.iv_Eff.max():.4f}")
    for tgt in C.IV_TARGETS:
        log(f"  {tgt:8s} mean {t[tgt].mean():.4f} sd {t[tgt].std(ddof=1):.4f} "
            f"range {t[tgt].max() - t[tgt].min():.5f}")


if __name__ == "__main__":
    main()
