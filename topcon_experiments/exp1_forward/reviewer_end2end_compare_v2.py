"""Reviewer-response diagnostic v2: end-to-end baseline + descriptor-error ablation.

Differences from v1 (reviewer_end2end_compare.py):
  * v1 died at the first predict(): the CatBoost component of the saved
    `defect_vac_gradient_max` model cannot be deserialised
    ("Not a double", catboost/library/.../json_value.cpp:450). That target spans
    ~1e15-1e19, so it is retrained here in log space -- the same convention the
    paper already applies to every multi-decade quantity (Note S2).
  * The eight model1 predictors that saved/load correctly are REUSED from the
    v1 output directory, so only one stage-1 model has to be refitted.

Routes (identical rows, identical split, identical AutoGluon preset):
  B2   model2 with GROUND-TRUTH descriptors        (upper bound / reported config)
  B1   model2 with MODEL1-PREDICTED descriptors    (actual two-stage forward path)
  A    end-to-end AutoGluon: 8 athena -> 4 I-V     (no explicit descriptor stage)
  LOO_TRUE[d] / LOO_PRED[d]  leave-one-out ablation (inference only)

Outputs -> outputs/exp1_forward/end_to_end/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from topcon_experiments.common.data import (
    preprocess_model1,
    preprocess_model2,
    train_test_split_xy,
)
from topcon_experiments.common.metrics import compute_regression_metrics
from topcon_experiments.config import (
    ATHENA_FEATURES,
    AUTOGLUON_PRESET,
    CURVE_DESCRIPTORS,
    IV_TARGETS,
    OUTPUT_ROOT,
)

DESCRIPTORS = list(CURVE_DESCRIPTORS)          # 9, giving the 17-dimensional union
MODEL2_FEATURES = ATHENA_FEATURES + DESCRIPTORS
OUT = OUTPUT_ROOT / "exp1_forward" / "end_to_end"

# Targets that need an internal log reparameterisation (numerical conditioning).
LOG_SPACE_LOCAL = {"defect_vac_gradient_max"}

# stage-1 models already trained and verified loadable in the v1 run
REUSE_M1 = {
    t: OUT / "model1" / f"autogluon_{t}"
    for t in DESCRIPTORS
    if t not in LOG_SPACE_LOCAL
}


def fit_predictor(X_tr: pd.DataFrame, y_tr: np.ndarray, label: str, path: Path) -> TabularPredictor:
    df = X_tr.reset_index(drop=True).copy()
    df[label] = pd.Series(np.asarray(y_tr, dtype=float)).reset_index(drop=True)
    predictor = TabularPredictor(label=label, path=str(path))
    predictor.fit(df, presets=AUTOGLUON_PRESET)
    return predictor


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: dict = {
        "preset": AUTOGLUON_PRESET,
        "descriptors": DESCRIPTORS,
        "n_features_model2": len(MODEL2_FEATURES),
        "log_space_local_targets": sorted(LOG_SPACE_LOCAL),
    }

    X1, y1, _ = preprocess_model1()
    X2_all, y2, _ = preprocess_model2()
    X2 = X2_all[MODEL2_FEATURES].copy()
    assert list(X1.index) == list(X2.index), "model1/model2 row sets differ"
    X2_train, X2_test, y2_train, y2_test = train_test_split_xy(X2, y2)
    tr_idx, te_idx = X2_train.index, X2_test.index
    X1_train, X1_test = X1.loc[tr_idx], X1.loc[te_idx]
    summary["n_train"], summary["n_test"] = int(len(tr_idx)), int(len(te_idx))

    # ---- stage 1 ---------------------------------------------------------
    m1_dirs: dict[str, str] = {t: str(p) for t, p in REUSE_M1.items()}
    for tgt in sorted(LOG_SPACE_LOCAL):
        t1 = time.time()
        yv = np.log(y1[tgt].loc[tr_idx].to_numpy(float))
        fit_predictor(X1_train, yv, tgt, OUT / "model1" / f"autogluon_{tgt}_log")
        m1_dirs[tgt] = str(OUT / "model1" / f"autogluon_{tgt}_log")
        print(f"[model1] {tgt} (log space) done in {time.time()-t1:.1f}s", flush=True)

    def predicted_descriptors(X_athena: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        cols, failed = {}, []
        for t in DESCRIPTORS:
            try:
                v = np.asarray(TabularPredictor.load(m1_dirs[t]).predict(X_athena), dtype=float).ravel()
                if t in LOG_SPACE_LOCAL:
                    v = np.exp(v)
                cols[t] = v
            except Exception as exc:                      # keep the run alive
                failed.append(f"{t}: {type(exc).__name__}")
                cols[t] = np.full(len(X_athena), np.nan)
        return pd.DataFrame(cols, index=X_athena.index)[DESCRIPTORS], failed

    pred_desc_test, failed = predicted_descriptors(X1_test)
    if failed:
        summary["stage1_load_failures"] = failed
        print("[warn] stage1 failures:", failed, flush=True)
    true_desc_test = X2_test[DESCRIPTORS]

    # ---- stage 2 (trained on ground-truth descriptors) -------------------
    m2_dirs: dict[str, str] = {}
    for tgt in IV_TARGETS:
        t2 = time.time()
        fit_predictor(X2_train, y2_train[tgt].to_numpy(float), tgt, OUT / "model2" / f"autogluon_{tgt}")
        m2_dirs[tgt] = str(OUT / "model2" / f"autogluon_{tgt}")
        print(f"[model2] {tgt} done in {time.time()-t2:.1f}s", flush=True)

    # ---- route A: end-to-end --------------------------------------------
    e2e_dirs: dict[str, str] = {}
    for tgt in IV_TARGETS:
        t3 = time.time()
        fit_predictor(X1_train, y2_train[tgt].to_numpy(float), tgt,
                      OUT / "model2_end_to_end" / f"autogluon_{tgt}")
        e2e_dirs[tgt] = str(OUT / "model2_end_to_end" / f"autogluon_{tgt}")
        print(f"[e2e] {tgt} done in {time.time()-t3:.1f}s", flush=True)

    summary["model1_dirs"] = m1_dirs
    summary["model2_dirs"] = m2_dirs
    summary["end_to_end_dirs"] = e2e_dirs

    # ---- evaluate --------------------------------------------------------
    def assemble(athena: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([athena[ATHENA_FEATURES], desc[DESCRIPTORS]], axis=1)[MODEL2_FEATURES]

    routes = {
        "B2_two_stage_true_desc": (assemble(X2_test, true_desc_test), m2_dirs),
        "B1_two_stage_pred_desc": (assemble(X2_test, pred_desc_test), m2_dirs),
        "A_end_to_end_8params": (X1_test[ATHENA_FEATURES], e2e_dirs),
    }
    rows = []
    for route, (X_in, dirs) in routes.items():
        for tgt in IV_TARGETS:
            y_hat = np.asarray(TabularPredictor.load(dirs[tgt]).predict(X_in), dtype=float).ravel()
            m = compute_regression_metrics(y2_test[tgt].to_numpy(float), y_hat, target=tgt)
            rows.append({"route": route, "target": tgt, "split": "test", **m})
            print(f"[eval] {route:24s} {tgt:8s} R2={m['R2']:+.4f} RMSE={m['RMSE']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "routes_metrics.csv", index=False)

    # ---- LOO ablations (inference only) ---------------------------------
    def score(desc: pd.DataFrame) -> dict[str, dict]:
        X_in = assemble(X2_test, desc)
        return {
            tgt: compute_regression_metrics(
                y2_test[tgt].to_numpy(float),
                np.asarray(TabularPredictor.load(m2_dirs[tgt]).predict(X_in), dtype=float).ravel(),
                target=tgt,
            )
            for tgt in IV_TARGETS
        }

    loo_true_rows, loo_pred_rows = [], []
    for d in DESCRIPTORS:
        desc = pred_desc_test.copy(); desc[d] = true_desc_test[d].to_numpy()
        for tgt, m in score(desc).items():
            loo_true_rows.append({"descriptor_from_truth": d, "target": tgt, **m})
        desc = true_desc_test.copy(); desc[d] = pred_desc_test[d].to_numpy()
        for tgt, m in score(desc).items():
            loo_pred_rows.append({"descriptor_from_prediction": d, "target": tgt, **m})
        print(f"[loo] {d} done", flush=True)

    pd.DataFrame(loo_true_rows).to_csv(OUT / "loo_true.csv", index=False)
    pd.DataFrame(loo_pred_rows).to_csv(OUT / "loo_pred.csv", index=False)

    d_rows = []
    for d in DESCRIPTORS:
        d_rows.append({"descriptor": d, **compute_regression_metrics(
            true_desc_test[d].to_numpy(float), pred_desc_test[d].to_numpy(float), target=d)})
    pd.DataFrame(d_rows).to_csv(OUT / "descriptor_stage1_metrics.csv", index=False)

    summary["total_seconds"] = time.time() - t0
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ALL DONE in {summary['total_seconds']:.1f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
