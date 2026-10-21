"""AutoGluon for the process->theta stage + full-chain evaluation.

Question: if the intermediate theta predictor is a stronger AutoML ensemble
instead of a single GBM (or SR formulas), how much does the end-to-end curve
accuracy improve? Same split / features / evaluation window as all other
stages so the numbers are directly comparable.

Outputs (under outputs/exp4_symbolic/process_to_theta_sr/data/):
  * theta_autogluon_metrics.csv     per-target test R2 (AG vs GBM reference)
  * autogluon_chain_eval.csv        per-test-sample chained curve R2
  * autogluon_chain_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.exp4_symbolic.run_process_to_dg_chain import (
    DG_EXT_CSV,
    DG_TARGETS,
    EXT_TRIM_CSV,
    _load_residual_expr,
    _predict_curve,
    _r2_log,
)
from topcon_experiments.exp4_symbolic.run_process_to_theta_sr import OUT_DIR, build_xy
from topcon_experiments.exp6_experimental.literature_benchmark.fit_double_gaussian_to_sim import (
    N_FLOOR,
)

DATA_OUT = OUT_DIR / "data"
AG_MODEL_DIR = OUT_DIR / "autogluon_models"
TIME_LIMIT_S = 420  # per target
PRESET = "good_quality"


def main() -> None:
    from autogluon.tabular import TabularPredictor

    setup_runtime()
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    X, y, train_fb, test_fb = build_xy()

    metrics = []
    pred_test: dict[str, np.ndarray] = {}
    for tgt in DG_TARGETS:
        tr = X.loc[train_fb].copy()
        tr["_y"] = y.loc[train_fb, tgt].values
        model_path = AG_MODEL_DIR / tgt
        if (model_path / "predictor.pkl").exists():
            predictor = TabularPredictor.load(str(model_path))
            log(f"loaded cached AG predictor for {tgt}")
        else:
            predictor = TabularPredictor(
                label="_y", path=str(model_path), problem_type="regression",
                verbosity=1,
            ).fit(tr, presets=PRESET, time_limit=TIME_LIMIT_S)
        pr = predictor.predict(X.loc[test_fb]).values
        pred_test[tgt] = pr
        r2 = r2_score(y.loc[test_fb, tgt].values, pr)
        metrics.append({"target": tgt, "model": "autogluon", "test_R2": r2})
        log(f"AG theta[{tgt}]: test R2={r2:.4f}")
    save_csv(pd.DataFrame(metrics), DATA_OUT / "theta_autogluon_metrics.csv")

    # Full-chain evaluation on the test set.
    curves = pd.read_csv(EXT_TRIM_CSV)
    dg = pd.read_csv(DG_EXT_CSV).set_index("file_base")
    resid_expr = _load_residual_expr()
    rows = []
    for i, fb in enumerate(test_fb):
        sub = curves[curves["file_base"] == fb].sort_values("depth_um")
        z = sub["depth_um"].values.astype(float)
        N = sub["value_fitted"].values.astype(float)
        keep = N >= N_FLOOR
        z_f, N_f = z[keep], N[keep]
        if z_f.size < 6:
            continue
        params = (float(np.exp(pred_test["ln_N_p"][i])), float(pred_test["z_p_um"][i]),
                  float(pred_test["z_f1"][i]), float(pred_test["z_f2"][i]))
        g = dg.loc[fb]
        fitted = (float(g["N_p_cm3"]), float(g["z_p_um"]), float(g["z_f1"]), float(g["z_f2"]))
        rows.append({
            "file_base": fb,
            "r2_dg_ag": _r2_log(N_f, _predict_curve(None, z_f, *params)),
            "r2_chain_ag": _r2_log(N_f, _predict_curve(resid_expr, z_f, *params)),
            "r2_dg_fitted": _r2_log(N_f, _predict_curve(None, z_f, *fitted)),
            "n_fit": int(z_f.size),
        })
    df = pd.DataFrame(rows)
    save_csv(df, DATA_OUT / "autogluon_chain_eval.csv")
    summary = []
    for col in ("r2_dg_ag", "r2_chain_ag", "r2_dg_fitted"):
        s = df[col].dropna()
        summary.append({"model": col, "count": len(s), "median": s.median(),
                        "n_gt_09": int((s > 0.9).sum()),
                        "n_gt_095": int((s > 0.95).sum())})
    sdf = pd.DataFrame(summary)
    save_csv(sdf, DATA_OUT / "autogluon_chain_summary.csv")
    print("AutoGluon chain on test set (R2_log, N>=1e18 window):")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
