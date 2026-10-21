"""Multi-objective NSGA-II Pareto optimization for IV metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import get_athena_bounds, load_raw_dataframe
from topcon_experiments.common.forward import load_forward_chain
from topcon_experiments.common.plot_utils import plot_pareto_2d, save_csv
from topcon_experiments.config import (
    ATHENA_FEATURES,
    BOUNDS_QUANTILE_HIGH,
    BOUNDS_QUANTILE_LOW,
    IV_TARGETS,
    META_JSON,
    NSGA_N_GEN,
    NSGA_POP_SIZE,
    OUTPUT_ROOT,
)


EXP2_OUT = OUTPUT_ROOT / "exp2_inverse"


class IVProblem(Problem):
    def __init__(self, chain, bounds: list[tuple[float, float]]):
        xl = np.array([b[0] for b in bounds])
        xu = np.array([b[1] for b in bounds])
        super().__init__(n_var=len(bounds), n_obj=len(IV_TARGETS), n_constr=0, xl=xl, xu=xu)
        self.chain = chain

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for row in X:
            iv = self.chain.predict_iv(row)
            F.append([-iv[t] for t in IV_TARGETS])
        out["F"] = np.array(F)


def extract_pareto_df(X: np.ndarray, F: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(len(X)):
        params = {col: float(X[i, j]) for j, col in enumerate(ATHENA_FEATURES)}
        iv = {t: float(-F[i, j]) for j, t in enumerate(IV_TARGETS)}
        rows.append({**params, **iv})
    return pd.DataFrame(rows)


def find_compromise(pareto_df: pd.DataFrame) -> pd.DataFrame:
    iv_cols = IV_TARGETS
    vals = pareto_df[iv_cols].values
    lo = vals.min(axis=0)
    hi = vals.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    norm = (vals - lo) / span
    ideal = np.ones(len(iv_cols))
    dist = np.linalg.norm(norm - ideal, axis=1)
    idx = int(np.argmin(dist))
    row = pareto_df.iloc[[idx]].copy()
    row["compromise_distance"] = dist[idx]
    return row


def main() -> None:
    EXP2_OUT.mkdir(parents=True, exist_ok=True)
    chain = load_forward_chain(META_JSON)
    df = load_raw_dataframe()
    bounds = get_athena_bounds(df, BOUNDS_QUANTILE_LOW, BOUNDS_QUANTILE_HIGH)

    problem = IVProblem(chain, bounds)
    algorithm = NSGA2(pop_size=NSGA_POP_SIZE)
    termination = get_termination("n_gen", NSGA_N_GEN)

    res = minimize(problem, algorithm, termination, seed=42, verbose=True)
    pareto_df = extract_pareto_df(res.X, res.F)
    save_csv(pareto_df, EXP2_OUT / "pareto_front.csv")

    plot_pareto_2d(pareto_df, "iv_Voc", "iv_Eff", EXP2_OUT / "pareto_voc_pce")
    plot_pareto_2d(pareto_df, "iv_FF", "iv_Jsc", EXP2_OUT / "pareto_ff_jsc")

    compromise = find_compromise(pareto_df)
    save_csv(compromise, EXP2_OUT / "pareto_best_compromise.csv")
    print(f"Pareto front: {len(pareto_df)} solutions")
    print(f"Compromise solution:\n{compromise.to_string(index=False)}")
    print(f"Saved to {EXP2_OUT}")


if __name__ == "__main__":
    main()
