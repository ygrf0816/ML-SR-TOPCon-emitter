"""Reviewer-response diagnostic: end-to-end baseline + descriptor-error ablation.

Question addressed (reviewer comment #3):
  The two-stage architecture (8 process params -> 9 descriptors -> 4 I-V) is
  justified on physical grounds, but no end-to-end benchmark was reported, so
  stage-1 error propagation cannot be separated from stage-2 residual.

Routes (identical rows, identical split, identical AutoGluon preset):
  B2   model2 with GROUND-TRUTH descriptors        (upper bound / reported config)
  B1   model2 with MODEL1-PREDICTED descriptors    (actual two-stage forward path)
  A    end-to-end AutoGluon: 8 athena -> 4 I-V     (no explicit descriptor stage)
  LOO_TRUE[d]     predicted descriptors, but d taken from truth   (d = 1..9)
  LOO_PRED[d]     true descriptors, but d replaced by prediction  (d = 1..9)

LOO_* reuse the already-fitted predictors, so they cost only inference.

Outputs -> outputs/exp1_forward/end_to_end/
  routes_metrics.csv      per-route per-target metrics (train + test)
  loo_true.csv            LOO_TRUE ablation, test split
  loo_pred.csv            LOO_PRED ablation, test split
  run_summary.json        config + timings
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

# The "full_to" union is 8 process parameters + 9 descriptors = 17 dimensions.
DESCRIPTORS = list(CURVE_DESCRIPTORS)
MODEL2_FEATURES = ATHENA_FEATURES + DESCRIPTORS
OUT = OUTPUT_ROOT / "exp1_forward" / "end_to_end"


def fit_predictor(X_tr: pd.DataFrame, y_tr: pd.Series, label: str, path: Path) -> TabularPredictor:
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
        "model2_features": MODEL2_FEATURES,
        "n_features_model2": len(MODEL2_FEATURES),
    }

    # --- data -------------------------------------------------------------
    X1, y1, _ = preprocess_model1()          # 8 athena -> 9 descriptors
    X2_all, y2, _ = preprocess_model2()      # 8 + 9 descriptors -> 4 I-V
    X2 = X2_all[MODEL2_FEATURES].copy()

    # Both preprocessors filter on the same positive-mask, so row order and
    # index are identical; the split below therefore applies to X1 as well.
    assert list(X1.index) == list(X2.index), "model1/model2 row sets differ"
    X2_train, X2_test, y2_train, y2_test = train_test_split_xy(X2, y2)
    tr_idx, te_idx = X2_train.index, X2_test.index
    X1_train, X1_test = X1.loc[tr_idx], X1.loc[te_idx]
    y1_train = {k: v.loc[tr_idx] for k, v in y1.items()}

    summary["n_train"] = int(len(tr_idx))
    summary["n_test"] = int(len(te_idx))

    # --- stage 1: 8 process params -> 9 descriptors -----------------------
    m1_dirs: dict[str, str] = {}
    for tgt in DESCRIPTORS:
        t1 = time.time()
        p = fit_predictor(X1_train, y1_train[tgt], tgt, OUT / "model1" / f"autogluon_{tgt}")
        m1_dirs[tgt] = str(OUT / "model1" / f"autogluon_{tgt}")
        print(f"[model1] {tgt} done in {time.time()-t1:.1f}s", flush=True)

    # descriptor values exactly in model2's input convention (log-space for the
    # four log descriptors, since preprocess_model2 takes their log).
    def predicted_descriptors(X_athena: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {t: TabularPredictor.load(m1_dirs[t]).predict(X_athena) for t in DESCRIPTORS},
            index=X_athena.index,
        )[DESCRIPTORS]

    pred_desc_test = predicted_descriptors(X1_test)
    true_desc_test = X2_test[DESCRIPTORS]

    # --- stage 2: 17 features -> 4 I-V (trained on TRUE descriptors) ------
    m2_dirs: dict[str, str] = {}
    for tgt in IV_TARGETS:
        t2 = time.time()
        fit_predictor(X2_train, y2_train[tgt], tgt, OUT / "model2" / f"autogluon_{tgt}")
        m2_dirs[tgt] = str(OUT / "model2" / f"autogluon_{tgt}")
        print(f"[model2] {tgt} done in {time.time()-t2:.1f}s", flush=True)

    # --- route A: end-to-end 8 process params -> 4 I-V --------------------
    e2e_dirs: dict[str, str] = {}
    for tgt in IV_TARGETS:
        t3 = time.time()
        fit_predictor(X1_train, y2_train[tgt], tgt, OUT / "model2_end_to_end" / f"autogluon_{tgt}")
        e2e_dirs[tgt] = str(OUT / "model2_end_to_end" / f"autogluon_{tgt}")
        print(f"[e2e] {tgt} done in {time.time()-t3:.1f}s", flush=True)

    summary["stage1_seconds"] = None
    summary["model1_dirs"] = m1_dirs
    summary["model2_dirs"] = m2_dirs
    summary["end_to_end_dirs"] = e2e_dirs

    # --- evaluate the three routes ---------------------------------------
    def assemble(athena: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([athena[ATHENA_FEATURES], desc[DESCRIPTORS]], axis=1)[MODEL2_FEATURES]

    routes = {
        "B2_two_stage_true_desc": (assemble(X2_test, true_desc_test), m2_dirs),
        "B1_two_stage_pred_desc": (assemble(X2_test, pred_desc_test), m2_dirs),
        "A_end_to_end_8params":   (X1_test[ATHENA_FEATURES], e2e_dirs),
    }
    rows = []
    for route, (X_in, dirs) in routes.items():
        for tgt in IV_TARGETS:
            predictor = TabularPredictor.load(dirs[tgt])
            y_true = np.asarray(y2_test[tgt].values, dtype=float)
            y_hat = np.asarray(predictor.predict(X_in), dtype=float)
            m = compute_regression_metrics(y_true, y_hat, target=tgt)
            rows.append({"route": route, "target": tgt, "split": "test", **m})
            print(f"[eval] {route:24s} {tgt:8s} R2={m['R2']:.4f} RMSE={m['RMSE']:.4f}", flush=True)
        # train-split fit quality (uses train rows through the same route)
    pd.DataFrame(rows).to_csv(OUT / "routes_metrics.csv", index=False)

    # --- LOO ablations on the test split (inference only) -----------------
    def score_desc_matrix(desc: pd.DataFrame) -> dict[str, dict]:
        X_in = assemble(X2_test, desc)
        out = {}
        for tgt in IV_TARGETS:
            predictor = TabularPredictor.load(m2_dirs[tgt])
            out[tgt] = compute_regression_metrics(
                np.asarray(y2_test[tgt].values, float),
                np.asarray(predictor.predict(X_in), float),
                target=tgt,
            )
        return out

    loo_true_rows, loo_pred_rows = [], []
    for d in DESCRIPTORS:
        # all predicted except d
        desc = pred_desc_test.copy()
        desc[d] = true_desc_test[d].values
        for tgt, m in score_desc_matrix(desc).items():
            loo_true_rows.append({"descriptor_from_truth": d, "target": tgt, **m})
        # all true except d
        desc = true_desc_test.copy()
        desc[d] = pred_desc_test[d].values
        for tgt, m in score_desc_matrix(desc).items():
            loo_pred_rows.append({"descriptor_from_prediction": d, "target": tgt, **m})
        print(f"[loo] {d} done", flush=True)

    pd.DataFrame(loo_true_rows).to_csv(OUT / "loo_true.csv", index=False)
    pd.DataFrame(loo_pred_rows).to_csv(OUT / "loo_pred.csv", index=False)

    # --- descriptor-level stage-1 accuracy (context for the ablation) -----
    d_rows = []
    for d in DESCRIPTORS:
        yt = np.asarray(true_desc_test[d].values, float)
        yp = np.asarray(pred_desc_test[d].values, float)
        d_rows.append({"descriptor": d, **compute_regression_metrics(yt, yp, target=d)})
    pd.DataFrame(d_rows).to_csv(OUT / "descriptor_stage1_metrics.csv", index=False)

    summary["total_seconds"] = time.time() - t0
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ALL DONE in {summary['total_seconds']:.1f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
