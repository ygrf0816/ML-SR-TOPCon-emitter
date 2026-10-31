# -*- coding: utf-8 -*-
"""DE convergence trace for one I-V target, using the RETRAINED ('after') chain.

Same optimizer, bounds, population size and seed as
`run_r1_4_pipeline.run_de`, so the output is directly comparable with
`out/de_trace_after.csv` (which is `iv_Eff`).

    python run_de_target.py --target iv_Voc --out out
    python run_de_target.py --smoke                 # load chain, 3 rows, exit

Writes
    out/de_trace_after_{Tgt}.csv          same 12 columns as de_trace_after.csv
    out/de_population_after_{Tgt}.csv     gen, idx, fitness  (one row per individual)
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

import config as C                                  # noqa: E402
from fwd import ForwardChain, get_athena_bounds     # noqa: E402

F1 = list(C.ATHENA_FEATURES)
PY_TARGETS = {"iv_Voc": "Voc", "iv_Jsc": "Jsc", "iv_FF": "FF", "iv_Eff": "PCE"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_de(chain: ForwardChain, bounds, tag: str, target: str, out_dir: Path,
           maxiter: int, popsize: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    from scipy.optimize import differential_evolution

    vals: list[float] = []
    trace: list[dict] = []
    pops: list[np.ndarray] = []
    last, best = 0, -np.inf
    n_pop = int(popsize * len(F1))
    t0 = time.perf_counter()

    def score_batch(X: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(X, columns=F1)
        D = chain.predict_curve_frame(frame)
        return chain.predict_iv_frame(frame, D)[target].to_numpy()

    def obj(x):
        nonlocal best
        X = np.asarray(x, float)
        if X.ndim == 1:
            v = float(score_batch(X.reshape(1, -1))[0])
            vals.append(v)
            best = max(best, v)
            return -v
        if X.shape[1] != len(F1) and X.shape[0] == len(F1):
            X = X.T                       # scipy <1.12 passes (S, N)
        v = score_batch(X)
        vals.extend(v.tolist())
        best = max(best, float(v.max()))
        if X.shape[0] == n_pop:           # one full generation
            try:
                pops.append(np.column_stack([np.full(len(v), len(trace)),
                                             np.arange(len(v)), v]))
            except Exception:
                pass
        return -v

    def cb(_xk, convergence=0.0):
        nonlocal last
        chunk = np.asarray(vals[last:], float)
        if chunk.size:
            q = np.percentile(chunk, [5, 25, 50, 75, 95])
            trace.append({"gen": len(trace), "best": float(best), "n_evals": int(chunk.size),
                          "gen_best": float(chunk.max()), "gen_worst": float(chunk.min()),
                          "mean": float(chunk.mean()), "std": float(chunk.std(ddof=0)),
                          "median": float(q[2]), "p05": float(q[0]), "p25": float(q[1]),
                          "p75": float(q[3]), "p95": float(q[4])})
            last = len(vals)
            log(f"  [{tag}] gen {len(trace)-1:3d} best={best:.5f} mean={chunk.mean():.5f} "
                f"evals={len(vals)} ({time.perf_counter()-t0:.0f}s)")
        return False

    log(f"[{tag}] DE target={target} maxiter={maxiter} popsize={popsize} seed={seed}")
    res = differential_evolution(obj, bounds=bounds, strategy="best1bin",
                                 maxiter=maxiter, popsize=popsize, mutation=C.DE_MUTATION,
                                 recombination=C.DE_RECOMBINATION, seed=seed,
                                 tol=C.DE_TOL, atol=C.DE_ATOL, polish=False, disp=False,
                                 vectorized=True, callback=cb)
    final_v = float(score_batch(np.asarray(res.x, float).reshape(1, -1))[0])
    if final_v > best:
        log(f"  [{tag}] final vector scores {final_v:.5f} (was {best:.5f}); appended")
        best = final_v
        trace.append({"gen": len(trace), "best": best, "n_evals": 1,
                      "gen_best": best, "gen_worst": best, "mean": best,
                      "std": 0.0, "median": best, "p05": best, "p25": best,
                      "p75": best, "p95": best})
    trace.append({"gen": len(trace), "best": float(best), "n_evals": 0,
                  "gen_best": float(best), "gen_worst": np.nan, "mean": np.nan,
                  "std": np.nan, "median": np.nan, "p05": np.nan, "p25": np.nan,
                  "p75": np.nan, "p95": np.nan})

    tgt = PY_TARGETS[target]
    pd.DataFrame(trace).to_csv(out_dir / f"de_trace_after_{tgt}.csv",
                               index=False, float_format="%.6g")
    if pops:
        P = pd.concat([pd.DataFrame(g, columns=["gen", "idx", "fitness"]) for g in pops],
                      ignore_index=True)
        P.to_csv(out_dir / f"de_population_after_{tgt}.csv",
                 index=False, float_format="%.6g")
        log(f"  [{tag}] population rows {len(P):,} over {P.gen.nunique()} generations")
    # 最优配方本身（schema 与 de_best_after.csv 一致，供 z-score / TCAD 复算用）
    rec = {"case": tag, **{c: float(x) for c, x in zip(F1, res.x)}}
    rec.update(chain.predict_all(res.x))
    pd.DataFrame([rec]).to_csv(out_dir / f"de_best_after_{tgt}.csv",
                               index=False, float_format="%.6g")
    log(f"  [{tag}] recipe -> de_best_after_{tgt}.csv, "
        + " ".join(f"{c}={float(v):.6g}" for c, v in zip(F1, res.x)))
    log(f"[{tag}] done: {res.nfev} evals, best={best:.5f}, {time.perf_counter()-t0:.0f}s")
    return res.x, pd.DataFrame(trace)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="iv_Voc", choices=list(PY_TARGETS))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--meta", default=str(HERE / "out" / "meta_retrained.json"))
    ap.add_argument("--maxiter", type=int, default=C.DE_MAXITER)
    ap.add_argument("--popsize", type=int, default=C.DE_POPSIZE)
    ap.add_argument("--seed", type=int, default=C.DE_SEED)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    log(f"chain: {args.meta}")
    chain = ForwardChain(Path(args.meta))

    df = pd.read_csv(C.DATA_CSV)
    df = df.loc[(df[list(set(C.LOG_FEATURES))] > 0).all(axis=1)].copy()
    bounds = get_athena_bounds(df, C.BOUNDS_QUANTILE_LOW, C.BOUNDS_QUANTILE_HIGH)
    log(f"bounds from {len(df):,} rows: "
        + ", ".join(f"{c}=[{lo:.6g},{hi:.6g}]" for c, (lo, hi) in zip(F1, bounds)))

    if args.smoke:
        X = pd.DataFrame(np.array([[(lo + hi) / 2 for lo, hi in bounds]] * 3), columns=F1)
        D = chain.predict_curve_frame(X)
        P = chain.predict_iv_frame(X, D)
        log("smoke " + " | ".join(f"{t}={P[t].iloc[0]:.4f}" for t in P.columns))
        return 0

    run_de(chain, bounds, f"after_{PY_TARGETS[args.target]}", args.target,
           out, args.maxiter, args.popsize, args.seed)
    return 0

if __name__ == "__main__":
    sys.exit(main())
