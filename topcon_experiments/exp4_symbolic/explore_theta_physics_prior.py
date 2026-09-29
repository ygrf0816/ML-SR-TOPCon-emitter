"""Do Fick/Arrhenius-shaped priors improve the process->theta stage?

Three predictors per DG parameter, all on the SAME train/test split as the
GBM/SR stages:

  1. physics  : a constrained analytic form derived from diffusion physics
                (fit with scipy least_squares; global parameters only)
  2. gbm      : plain HistGradientBoostingRegressor (reference, physics feats)
  3. hybrid   : physics backbone + GBM trained on its residual

Physics forms
-------------
  L^2(process) = c_B^gamma * [ a1*t1*exp(-Ea1/kT1) + a2*t2*exp(-Ea2/kT2) ]
      -> two-step additive Dt with concentration-enhanced diffusivity.
  z_f2 = sqrt(L^2)                                   (diffusion length)
  z_p  = c0 + c1*thick + c2*sqrt(L^2)                (interface + drive-in)
  z_f1 = c0 + c1*thick + c2*sqrt(L^2)
  ln_N_p = linear in [ln c_B, 1/T1K, 1/T2K, ln t1, ln t2, thick]
      -> Arrhenius/solubility log-linear form.

Output: outputs/exp4_symbolic/process_to_theta_sr/data/theta_physics_prior.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import ATHENA_FEATURES, RANDOM_STATE
from topcon_experiments.exp4_symbolic.run_process_to_dg_chain import DG_EXT_CSV, _physics_features
from topcon_experiments.exp4_symbolic.run_process_to_theta_sr import OUT_DIR, build_xy

K_B = 8.617333262e-5  # eV/K
OUT_CSV = OUT_DIR / "data" / "theta_physics_prior.csv"


def _raw_process(meta: pd.DataFrame, fbs: list[str]) -> pd.DataFrame:
    P = meta.loc[fbs, ATHENA_FEATURES].astype(float).copy()
    P["T1K"] = P["athena_temp1"] + 273.15
    P["T2K"] = P["athena_temp2"] + 273.15
    P["ln_cB"] = np.log(P["athena_c_boron"])
    return P


def _L2(p: np.ndarray, P: pd.DataFrame) -> np.ndarray:
    """Two-step additive Dt with concentration enhancement (all-positive parts)."""
    ln_a1, ln_a2, Ea1, Ea2, gamma = p
    term1 = np.exp(ln_a1) * P["athena_time1"] * np.exp(-Ea1 / (K_B * P["T1K"]))
    term2 = np.exp(ln_a2) * P["athena_time2"] * np.exp(-Ea2 / (K_B * P["T2K"]))
    # center ln c_B so gamma is identifiable independent of a1/a2 scale
    enh = np.exp(gamma * (P["ln_cB"] - P["ln_cB"].mean()))
    return (enh * (term1 + term2)).values


def _fit_sqrtL2_offset(P_tr, y_tr, with_offset: bool):
    """Fit y = [c0 + c1*thick +] c2*sqrt(L2(p)); return predict fn."""
    def model(q, P):
        p, rest = q[:5], q[5:]
        L = np.sqrt(np.maximum(_L2(p, P), 1e-30))
        if with_offset:
            c0, c1, c2 = rest
            return c0 + c1 * P["athena_thick"].values + c2 * L
        return rest[0] * L

    n_rest = 3 if with_offset else 1
    q0 = np.array([0.0, 0.0, 2.0, 2.0, 0.3] + [0.1] * n_rest)
    # scale init: match mean
    q0[0] = q0[1] = np.log(np.mean(y_tr) ** 2 / max(np.mean(
        P_tr["athena_time1"] * np.exp(-2.0 / (K_B * P_tr["T1K"]))
        + P_tr["athena_time2"] * np.exp(-2.0 / (K_B * P_tr["T2K"]))), 1e-30))
    res = least_squares(
        lambda q: model(q, P_tr) - y_tr, q0,
        bounds=([-60, -60, 0.5, 0.5, -3] + [-5] * n_rest,
                [60, 60, 6.0, 6.0, 3] + [5] * n_rest),
        max_nfev=20000,
    )
    return lambda P: model(res.x, P), res.x


def _fit_lnNp(P_tr, y_tr):
    A = np.column_stack([
        np.ones(len(P_tr)), P_tr["ln_cB"], 1.0 / P_tr["T1K"], 1.0 / P_tr["T2K"],
        np.log(P_tr["athena_time1"]), np.log(P_tr["athena_time2"]),
        P_tr["athena_thick"],
    ])
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)

    def predict(P):
        A_ = np.column_stack([
            np.ones(len(P)), P["ln_cB"], 1.0 / P["T1K"], 1.0 / P["T2K"],
            np.log(P["athena_time1"]), np.log(P["athena_time2"]),
            P["athena_thick"],
        ])
        return A_ @ coef
    return predict, coef


def _gbm():
    return HistGradientBoostingRegressor(
        random_state=RANDOM_STATE, max_iter=1500, learning_rate=0.05,
        max_leaf_nodes=63, min_samples_leaf=15, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
    )


def main() -> None:
    setup_runtime()
    X, y, train_fb, test_fb = build_xy()
    meta = load_raw_dataframe().set_index("file_base")
    P_tr = _raw_process(meta, train_fb)
    P_te = _raw_process(meta, test_fb)

    physics_fns = {}
    physics_fns["z_f2"], p_zf2 = _fit_sqrtL2_offset(P_tr, y.loc[train_fb, "z_f2"].values, False)
    physics_fns["z_p_um"], p_zp = _fit_sqrtL2_offset(P_tr, y.loc[train_fb, "z_p_um"].values, True)
    physics_fns["z_f1"], p_zf1 = _fit_sqrtL2_offset(P_tr, y.loc[train_fb, "z_f1"].values, True)
    physics_fns["ln_N_p"], c_np = _fit_lnNp(P_tr, y.loc[train_fb, "ln_N_p"].values)
    log(f"z_f2 physics params: Ea1={p_zf2[2]:.2f} eV, Ea2={p_zf2[3]:.2f} eV, gamma={p_zf2[4]:.2f}")
    log(f"z_p  physics params: Ea1={p_zp[2]:.2f} eV, Ea2={p_zp[3]:.2f} eV, gamma={p_zp[4]:.2f}, "
        f"c=[{p_zp[5]:.3f}, {p_zp[6]:.3f}, {p_zp[7]:.3f}]")

    rows = []
    for tgt in ("ln_N_p", "z_p_um", "z_f1", "z_f2"):
        y_tr = y.loc[train_fb, tgt].values
        y_te = y.loc[test_fb, tgt].values
        phys_tr = physics_fns[tgt](P_tr)
        phys_te = physics_fns[tgt](P_te)

        gbm_plain = _gbm().fit(X.loc[train_fb], y_tr)
        gbm_resid = _gbm().fit(X.loc[train_fb], y_tr - phys_tr)

        preds = {
            "physics": phys_te,
            "gbm": gbm_plain.predict(X.loc[test_fb]),
            "hybrid": phys_te + gbm_resid.predict(X.loc[test_fb]),
        }
        for name, pr in preds.items():
            rows.append({"target": tgt, "model": name,
                         "test_R2": r2_score(y_te, pr),
                         "test_RMSE": float(np.sqrt(np.mean((y_te - pr) ** 2)))})
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, OUT_CSV)
    print(df.pivot_table(index="target", columns="model", values="test_R2").round(4).to_string())


if __name__ == "__main__":
    main()
