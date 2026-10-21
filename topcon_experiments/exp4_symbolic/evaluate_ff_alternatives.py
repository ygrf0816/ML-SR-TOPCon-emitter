"""Compare FF prediction: SR full features vs SR from Voc/Jsc/Eff vs direct formula."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import preprocess_model2
from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import OUTPUT_ROOT, RANDOM_STATE, TEST_SIZE
from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy
from topcon_experiments.exp4_symbolic.sr_eval_utils import PREPROCESS_NOTES, run_pysr_tabular_eval

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
IV_TRIO = ["iv_Voc", "iv_Jsc", "iv_Eff"]


def evaluate_identity_split(df: pd.DataFrame) -> dict:
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    voc, jsc, eff, ff = (
        df["iv_Voc"].values.astype(float),
        df["iv_Jsc"].values.astype(float),
        df["iv_Eff"].values.astype(float),
        df["iv_FF"].values.astype(float),
    )
    ff_calc = 100.0 * eff / (voc * jsc)
    return {
        "method": "identity_FF_calc",
        "train_R2": float(r2_score(ff[tr], ff_calc[tr])),
        "test_R2": float(r2_score(ff[te], ff_calc[te])),
        "train_RMSE": float(np.sqrt(np.mean((ff[tr] - ff_calc[tr]) ** 2))),
        "test_RMSE": float(np.sqrt(np.mean((ff[te] - ff_calc[te]) ** 2))),
        "equation": "FF_calc = 100 * iv_Eff / (iv_Voc * iv_Jsc)",
        "preprocess_note": PREPROCESS_NOTES["ff_identity"],
    }


def run_trio_sr(df: pd.DataFrame) -> pd.DataFrame:
    X = df[IV_TRIO].astype(float)
    y = df["iv_FF"].astype(float).values
    return run_pysr_tabular_eval(X, y, "iv_trio_to_FF", "iv_trio_ff")


def load_full_ff_best() -> dict | None:
    path = EXP4_OUT / "sr_tabular_full_to_iv_FF.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    if d.empty or "test_R2" not in d.columns:
        return None
    best = sort_formulas_by_accuracy(d).iloc[0]
    return {
        "method": "sr_full_features",
        "train_R2": float(best.get("train_R2", np.nan)),
        "test_R2": float(best.get("test_R2", np.nan)),
        "equation": str(best.get("equation", "")),
        "preprocess_note": PREPROCESS_NOTES["model2_iv_full"],
    }


def main() -> None:
    setup_runtime()
    EXP4_OUT.mkdir(parents=True, exist_ok=True)

    X_full, y_map, meta = preprocess_model2()
    df = meta.copy()
    for c in IV_TRIO + ["iv_FF"]:
        df[c] = y_map[c].values

    rows: list[dict] = []

    ident = evaluate_identity_split(df)
    rows.append(ident)
    log(f"Identity FF: train_R2={ident['train_R2']:.4f}, test_R2={ident['test_R2']:.4f}")

    trio_eq = run_trio_sr(df)
    save_csv(trio_eq, EXP4_OUT / "sr_tabular_iv_trio_to_FF.csv")
    best_trio = sort_formulas_by_accuracy(trio_eq).iloc[0]
    rows.append({
        "method": "sr_iv_trio",
        "train_R2": float(best_trio["train_R2"]),
        "test_R2": float(best_trio["test_R2"]),
        "equation": str(best_trio["equation"]),
        "preprocess_note": PREPROCESS_NOTES["iv_trio_ff"],
    })
    log(f"SR iv_trio->FF: train_R2={best_trio['train_R2']:.4f}, test_R2={best_trio['test_R2']:.4f}")

    full_ff = load_full_ff_best()
    if full_ff:
        rows.append(full_ff)
        log(f"SR full->FF (cached): test_R2={full_ff['test_R2']:.4f}")

    cmp_df = pd.DataFrame(rows)
    save_csv(cmp_df, EXP4_OUT / "sr_ff_comparison.csv")
    log(f"FF comparison -> {EXP4_OUT / 'sr_ff_comparison.csv'}")


if __name__ == "__main__":
    main()
