"""PySR symbolic regression: process parameters -> double-Gaussian parameters.

Goal: replace the black-box GBM in the process->theta stage with analytic
formulas so the FULL chain becomes symbolic:

    N(z) = N_DG(z; theta) [* exp(r_SR(z; theta))],  theta = f_SR(process)

Targets (one PySR run each): ln_N_p, z_p_um, z_f1, z_f2 — the per-sample DG
fits from ``run_process_to_dg_chain`` (5k+ samples, fit on N>=1e18 window).

Inputs: 8 Athena process params (ln c_boron) + Arrhenius physics features
(same construction as the GBM stage). Train/test split identical to the GBM
stage (train_test_split on file_base, RANDOM_STATE) so metrics are comparable
and the chain test set stays untouched.

Outputs (under outputs/exp4_symbolic/process_to_theta_sr/):
  * sr_formulas_theta_<target>.csv   PySR equations + train/test metrics
  * feature_columns.json             X column order (x0..xN mapping)
  * theta_sr_summary.csv             best-per-target summary vs GBM
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import ATHENA_FEATURES, OUTPUT_ROOT, RANDOM_STATE
from topcon_experiments.exp4_symbolic.run_process_to_dg_chain import (
    DG_EXT_CSV,
    DG_TARGETS,
    _physics_features,
)
from topcon_experiments.exp4_symbolic.sr_metrics import annotate_equation_metrics
from topcon_experiments.exp4_symbolic.sr_utils import build_pysr_regressor

OUT_DIR = OUTPUT_ROOT / "exp4_symbolic" / "process_to_theta_sr"


def build_xy() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Feature matrix / targets / split, identical to the GBM stage."""
    dg = pd.read_csv(DG_EXT_CSV).set_index("file_base")
    meta = load_raw_dataframe().set_index("file_base")
    ok_fb = [fb for fb in dg.index if fb in meta.index]
    X = meta.loc[ok_fb, ATHENA_FEATURES].astype(float).copy()
    # A couple of samples have missing Athena metadata; drop them.
    valid = X.notna().all(axis=1)
    X = X.loc[valid]
    ok_fb = list(X.index)
    X["athena_c_boron"] = np.log(X["athena_c_boron"])
    X = _physics_features(X)
    y = pd.DataFrame({
        "ln_N_p": np.log(dg.loc[ok_fb, "N_p_cm3"].astype(float)),
        "z_p_um": dg.loc[ok_fb, "z_p_um"].astype(float),
        "z_f1": dg.loc[ok_fb, "z_f1"].astype(float),
        "z_f2": dg.loc[ok_fb, "z_f2"].astype(float),
    }, index=ok_fb)
    train_fb, test_fb = train_test_split(ok_fb, test_size=0.2, random_state=RANDOM_STATE)
    return X, y, train_fb, test_fb


def main() -> None:
    setup_runtime()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, train_fb, test_fb = build_xy()
    (OUT_DIR / "feature_columns.json").write_text(
        json.dumps(list(X.columns), ensure_ascii=False, indent=2)
    )
    log(f"theta SR: {len(train_fb)} train / {len(test_fb)} test samples, "
        f"{X.shape[1]} features")

    summary = []
    for tgt in DG_TARGETS:
        out_csv = OUT_DIR / f"sr_formulas_theta_{tgt}.csv"
        if out_csv.exists():
            log(f"skip {tgt}: {out_csv.name} exists")
            eq_df = pd.read_csv(out_csv)
        else:
            X_tr, y_tr = X.loc[train_fb], y.loc[train_fb, tgt].values
            X_te, y_te = X.loc[test_fb], y.loc[test_fb, tgt].values
            log(f"PySR theta[{tgt}]: fit {len(X_tr)} rows")
            model = build_pysr_regressor(f"theta_{tgt}")
            model.fit(X_tr.values, y_tr)
            eq_df = model.equations_.copy()
            eq_df = annotate_equation_metrics(
                model, eq_df, X_tr.values, y_tr, X_te.values, y_te
            )
            eq_df["target"] = tgt
            eq_df["n_train"] = len(X_tr)
            eq_df["n_test"] = len(X_te)
            eq_df["preprocess_key"] = "process_to_theta"
            eq_df["preprocess_note"] = (
                "X=8个Athena工艺参数(c_boron取ln)+Arrhenius物理特征(ln Dt等)；"
                "y=对N>=1e18窗口逐样本拟合的双高斯参数；"
                "划分与process_to_dg_chain的GBM阶段一致(按file_base, 8:2)"
            )
            save_csv(eq_df, out_csv)
        best = eq_df.sort_values("test_R2", ascending=False).iloc[0]
        summary.append({
            "target": tgt,
            "best_test_R2": float(best["test_R2"]),
            "complexity": int(best["complexity"]),
            "equation": str(best["equation"]),
        })
        log(f"  {tgt}: best test_R2={best['test_R2']:.4f} "
            f"(complexity={best['complexity']})")

    save_csv(pd.DataFrame(summary), OUT_DIR / "theta_sr_summary.csv")
    log(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
