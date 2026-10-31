"""R1-4 pipeline: score the augmentation, retrain Model 2, re-run inverse design.

Runs on the analysis workstation, inside `topcon_experiments/`:

    source ~/anaconda3/bin/activate autogluon
    cd ~/code/tcadproject/topcon_experiments
    python response_inverse_experiment/run_r1_4_pipeline.py \
        --aug response_inverse_experiment/topcon_dataset_v1_aug.csv \
        --out response_inverse_experiment/out

What it does, in order, and why each step is safe to repeat:

  1. Reproduce the ORIGINAL train/test split from `models_meta.json`
     (`test_indices_model2`). The test set never changes, so "before/after" is
     comparable and every number in the paper stays valid.
  2. Score the 50 new recipes with the CURRENT surrogate.  This is the missing
     measurement: does the model really under-predict at the top?
  3. Retrain the four Model-2 targets on train + 50 new rows (the new rows go
     into the TRAIN side only), then re-score the same test set.
  4. Run differential evolution on the retrained chain, take the objective value
     straight from the surrogate output, and record the per-generation trace in
     the same schema as `de_trace_iv_Eff.csv`.
  5. Compare the new optimum's surrogate value with the TCAD value that the
     local run already established for the same recipe family.

Nothing is overwritten: every artefact goes under `--out`.
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
ROOT = HERE.parent                      # .../topcon_experiments
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from topcon_experiments.config import (          # noqa: E402
    ATHENA_FEATURES,
    AUTOGLUON_PRESET,
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
    IV_TARGET_LABELS,
    IV_TARGETS,
    LOG_FEATURES,
    META_JSON,
    MODEL1_DIR,
    MODEL2_FEATURES,
    RANDOM_STATE,
)
from topcon_experiments.common.data import get_athena_bounds, load_meta, load_raw_dataframe  # noqa: E402
from topcon_experiments.common.forward import ForwardChain                  # noqa: E402
from topcon_experiments.common.autogluon_utils import safe_predict          # noqa: E402

from autogluon.tabular import TabularPredictor                              # noqa: E402


# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_frames(aug: pd.DataFrame):
    """Original and augmented frames, all with the training-time transform.

    Index handling matters: `models_meta.json` stores *positional* row indices of
    the original split, so the original frame must keep its own index untouched
    and the augmented rows must get indices that cannot collide with it.
    """
    df = load_raw_dataframe()
    mask = (df[list(set(LOG_FEATURES))] > 0).all(axis=1)
    df = df.loc[mask].copy()

    def _transform(frame: pd.DataFrame):
        X = frame[MODEL2_FEATURES].copy()
        for col in LOG_FEATURES:
            X[col] = np.log(X[col].astype(float))
        y = {t: frame[t].astype(float) for t in IV_TARGETS}
        return X, y

    X_o, y_o = _transform(df)

    aug = aug.copy()
    aug.index = np.arange(len(df), len(df) + len(aug))
    X_a, y_a = _transform(aug)
    return df, X_o, y_o, X_a, y_a


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def evaluate(predictors: dict[str, TabularPredictor], X: pd.DataFrame,
             y: dict[str, np.ndarray], tag: str) -> pd.DataFrame:
    rows = []
    for t in IV_TARGETS:
        pred = safe_predict(predictors[t], X)
        rows.append({"case": tag, "target": t, "n": len(X),
                     "R2": r2(y[t], pred),
                     "RMSE": float(np.sqrt(((np.asarray(y[t]) - pred) ** 2).mean())),
                     "bias": float((pred - np.asarray(y[t])).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def run_de(chain: ForwardChain, bounds, tag: str, out_dir: Path,
           maxiter: int, popsize: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Same DE as exp2_inverse.de_single, but with a swappable chain and tag."""
    from scipy.optimize import differential_evolution

    gen_vals: list[float] = []
    trace: list[dict] = []
    last = 0
    best_sofar = -np.inf
    t0 = time.perf_counter()

    def objective(x):
        nonlocal best_sofar
        v = float(chain.predict_iv_single(x, "iv_Eff"))     # surrogate output, as is
        gen_vals.append(v)
        best_sofar = max(best_sofar, v)
        return -v

    def callback(_xk, convergence=0.0):
        nonlocal last
        chunk = np.asarray(gen_vals[last:], float)
        if chunk.size:
            q05, q25, q50, q75, q95 = np.percentile(chunk, [5, 25, 50, 75, 95])
            trace.append({"gen": len(trace), "best": float(best_sofar),
                          "n_evals": int(chunk.size), "gen_best": float(chunk.max()),
                          "gen_worst": float(chunk.min()), "mean": float(chunk.mean()),
                          "std": float(chunk.std(ddof=0)), "median": float(q50),
                          "p05": float(q05), "p25": float(q25),
                          "p75": float(q75), "p95": float(q95)})
            last = len(gen_vals)
            log(f"  [{tag}] gen {len(trace)-1:3d}: best={best_sofar:.4f} "
                f"mean={chunk.mean():.4f} evals={len(gen_vals)} "
                f"({time.perf_counter()-t0:.0f}s)")
        return False

    log(f"[{tag}] DE maxiter={maxiter} popsize={popsize} seed={seed} "
        f"(objective = raw surrogate PCE)")
    res = differential_evolution(objective, bounds=bounds, strategy="best1bin",
                                 maxiter=maxiter, popsize=popsize, mutation=DE_MUTATION,
                                 recombination=DE_RECOMBINATION, seed=seed,
                                 tol=DE_TOL, atol=DE_ATOL, polish=False, disp=False,
                                 callback=callback)
    trace.append({"gen": len(trace), "best": float(best_sofar), "n_evals": 0,
                  "gen_best": float(best_sofar), "gen_worst": float("nan"),
                  "mean": float("nan"), "std": float("nan"), "median": float("nan"),
                  "p05": float("nan"), "p25": float("nan"),
                  "p75": float("nan"), "p95": float("nan")})
    df = pd.DataFrame(trace)
    df.to_csv(out_dir / f"de_trace_{tag}.csv", index=False, float_format="%.6g")
    log(f"[{tag}] DE done: {res.nfev} evals, best={best_sofar:.4f}%, "
        f"{time.perf_counter()-t0:.0f}s")
    return res.x, df


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", required=True)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--maxiter", type=int, default=DE_MAXITER)
    ap.add_argument("--popsize", type=int, default=DE_POPSIZE)
    ap.add_argument("--seed", type=int, default=DE_SEED)
    ap.add_argument("--skip-retrain", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    aug = pd.read_csv(args.aug)
    log(f"augmentation rows: {len(aug)}  (PCE {aug.iv_Eff.min():.2f}-{aug.iv_Eff.max():.2f})")

    meta = load_meta(META_JSON)
    test_idx = list(meta["test_indices_model2"])
    log(f"original meta: {len(meta['feature_columns_model2'])} model-2 features, "
        f"{len(test_idx)} test rows")

    df, X_o, y_o, X_aug, y_aug = build_frames(aug)
    avail = set(X_o.index)
    test_index = [i for i in test_idx if i in avail]
    train_index = [i for i in X_o.index if i not in set(test_idx)]
    log(f"split reproduced: train {len(train_index)} + test {len(test_index)} "
        f"(+{len(X_aug)} new rows on the train side)")

    X_test = X_o.loc[test_index]
    y_test = {t: y_o[t].loc[test_index].values for t in IV_TARGETS}
    y_aug_arr = {t: y_aug[t].values for t in IV_TARGETS}

    # ---------------- step 1: score the new recipes with the current model ----
    log("step 1/5: current surrogate on the 50 new recipes")
    old = {t: TabularPredictor.load(meta["model2_dirs"][t]) for t in IV_TARGETS}
    pred = {t: safe_predict(old[t], X_aug) for t in IV_TARGETS}
    score = pd.DataFrame({
        "file_base": aug["file_base"].values,
        **{f"tcad_{t}": y_aug_arr[t] for t in IV_TARGETS},
        **{f"surrogate_{t}": pred[t] for t in IV_TARGETS},
    })
    score["gap_iv_Eff"] = score["tcad_iv_Eff"] - score["surrogate_iv_Eff"]
    score.to_csv(out / "surrogate_vs_tcad_50.csv", index=False, float_format="%.6g")
    log(f"  PCE: TCAD mean {score.tcad_iv_Eff.mean():.3f}, "
        f"surrogate mean {score.surrogate_iv_Eff.mean():.3f}, "
        f"mean gap {score.gap_iv_Eff.mean():+.3f} pp, "
        f"max surrogate {score.surrogate_iv_Eff.max():.3f}")
    for lo, hi in ((0, 26.5), (26.5, 27.0), (27.0, 99)):
        s = score[(score.tcad_iv_Eff >= lo) & (score.tcad_iv_Eff < hi)]
        if len(s):
            log(f"  TCAD in [{lo},{hi}): n={len(s)} gap={s.gap_iv_Eff.mean():+.3f} pp")

    # ---------------- step 2: retrain model 2 on train + new ------------------
    retrained_dirs: dict[str, str] = {}
    if not args.skip_retrain:
        log("step 2/5: retraining Model 2 (new rows on the train side only)")
        X_tr = pd.concat([X_o.loc[train_index], X_aug], axis=0).reset_index(drop=True)
        for t in IV_TARGETS:
            y_tr = np.concatenate([y_o[t].loc[train_index].values, y_aug_arr[t]])
            train_df = X_tr.copy()
            train_df[t] = y_tr
            save = out / "models_retrained" / f"autogluon_{t}"
            save.mkdir(parents=True, exist_ok=True)
            log(f"  fitting {t} on {len(train_df)} rows -> {save}")
            p = TabularPredictor(label=t, path=str(save))
            p.fit(train_df, presets=AUTOGLUON_PRESET)
            retrained_dirs[t] = str(save)
            log(f"  {t} done; leaderboard best: "
                f"{p.leaderboard(silent=True).iloc[0]['model']}")
    else:
        log("step 2/5: skipped (--skip-retrain)")

    # ---------------- step 3: before/after on the SAME test set --------------
    log("step 3/5: test-set accuracy, before vs after")
    ev_old = evaluate(old, X_test, y_test, "before")
    ev_new = evaluate({t: TabularPredictor.load(retrained_dirs[t]) for t in IV_TARGETS},
                      X_test, y_test, "after") if retrained_dirs else pd.DataFrame()
    ev = pd.concat([ev_old, ev_new], ignore_index=True)
    ev.to_csv(out / "test_accuracy_before_after.csv", index=False, float_format="%.6g")
    for t in IV_TARGETS:
        a = ev[(ev.case == "before") & (ev.target == t)].R2.iloc[0]
        b = (ev[(ev.case == "after") & (ev.target == t)].R2.iloc[0]
             if len(ev_new) else float("nan"))
        log(f"  {t:8s} R2 {a:.4f} -> {b:.4f}  (delta {b - a:+.4f})")

    # high-efficiency slice of the test set
    hi = y_test["iv_Eff"] >= 25.0
    log(f"  test rows with true PCE >= 25%: {int(hi.sum())}")
    for t in IV_TARGETS:
        pa = safe_predict(old[t], X_test)
        ra = 1 - (((y_test[t][hi] - pa[hi]) ** 2).sum() /
                  ((y_test[t][hi] - y_test[t][hi].mean()) ** 2).sum())
        if retrained_dirs:
            pb = safe_predict(TabularPredictor.load(retrained_dirs[t]), X_test)
            rb = 1 - (((y_test[t][hi] - pb[hi]) ** 2).sum() /
                      ((y_test[t][hi] - y_test[t][hi].mean()) ** 2).sum())
        else:
            rb = float("nan")
        log(f"  high-PCE slice {t:8s} R2 {ra:.4f} -> {rb:.4f}")

    # ---------------- step 4: inverse design --------------------------------
    log("step 4/5: inverse design on the retrained chain")
    bounds = get_athena_bounds(df, BOUNDS_QUANTILE_LOW, BOUNDS_QUANTILE_HIGH)
    log(f"  bounds: {dict(zip(ATHENA_FEATURES, [tuple(round(v, 6) for v in b) for b in bounds]))}")

    chain_before = ForwardChain(META_JSON)
    x_before, tr_before = run_de(chain_before, bounds, "before",
                                 out, args.maxiter, args.popsize, args.seed)

    results = []
    for tag, chain in (("before", chain_before),):
        rec = {**{c: float(x_before[i]) for i, c in enumerate(ATHENA_FEATURES)}}
        rec.update(chain.predict_all(x_before))
        rec["optimized_metric"] = "iv_Eff"
        rec["case"] = tag
        results.append(rec)
        pd.DataFrame([rec]).to_csv(out / f"de_best_{tag}.csv",
                                   index=False, float_format="%.6g")
        log(f"  [{tag}] surrogate PCE = {rec['iv_Eff']:.4f}%")

    if retrained_dirs:
        meta_new = dict(meta)
        meta_new["model2_dirs"] = retrained_dirs
        (out / "meta_retrained.json").write_text(
            json.dumps(meta_new, indent=2, ensure_ascii=False), encoding="utf-8")
        chain_after = ForwardChain(out / "meta_retrained.json")
        x_after, tr_after = run_de(chain_after, bounds, "after",
                                   out, args.maxiter, args.popsize, args.seed)
        rec = {**{c: float(x_after[i]) for i, c in enumerate(ATHENA_FEATURES)}}
        rec.update(chain_after.predict_all(x_after))
        rec["optimized_metric"] = "iv_Eff"
        rec["case"] = "after"
        results.append(rec)
        pd.DataFrame([rec]).to_csv(out / "de_best_after.csv",
                                   index=False, float_format="%.6g")
        log(f"  [after] surrogate PCE = {rec['iv_Eff']:.4f}%")
        pd.concat([tr_before.assign(case="before"),
                   tr_after.assign(case="after")]).to_csv(
            out / "de_trace_compare.csv", index=False, float_format="%.6g")

    pd.DataFrame(results).to_csv(out / "de_best_summary.csv",
                                 index=False, float_format="%.6g")

    # ---------------- step 5: recipe for the local TCAD check ----------------
    log("step 5/5: writing the recipes for local TCAD re-verification")
    keep = ATHENA_FEATURES + CURVE_DESCRIPTORS + IV_TARGETS
    payload = pd.DataFrame(results)[["case"] + keep]
    payload.to_csv(out / "recipe_for_tcad.csv", index=False, float_format="%.6g")
    log(f"  -> {out / 'recipe_for_tcad.csv'}")
    log("done. copy the whole out/ directory back and run the local comparison.")


if __name__ == "__main__":
    main()
