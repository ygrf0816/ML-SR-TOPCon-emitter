"""Train/test evaluation and preprocessing notes for symbolic regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from topcon_experiments.config import RANDOM_STATE, SR_CURVE_MAX_FIT_ROWS, SR_TABULAR_MAX_ROWS
from topcon_experiments.exp4_symbolic.sr_metrics import annotate_equation_metrics
from topcon_experiments.exp4_symbolic.sr_split_utils import split_group_train_test, split_train_test
from topcon_experiments.exp4_symbolic.sr_utils import build_pysr_regressor

PREPROCESS_NOTES: dict[str, str] = {
    "model1_tabular": (
        "与 exp1 model1 对齐：样本要求 LOG_FEATURES+LOG_TARGETS_MODEL1 全部 >0；"
        "输入 X 对 LOG_FEATURES 取 **自然对数 ln**；"
        "目标 y 对 doping_N_peak / doping_dose / defect 剂量类取 **ln**，其余为原尺度。"
    ),
    "model2_iv_athena": (
        "与 exp1 model2 样本筛选一致（LOG_FEATURES>0），但输入 **仅 8 个 Athena 工艺参数**；"
        "athena_c_boron 取 **ln**；IV 目标为原尺度 (V, mA/cm², %, mW/cm²)。"
    ),
    "model2_iv_full": (
        "与 exp1 model2 完全对齐：X = Athena(8) + 掺杂描述符(7) + 缺陷描述符(3)；"
        "浓度类特征取 **ln**（见 LOG_FEATURES）；IV 目标为原尺度。"
    ),
    "curve": (
        "曲线 SR：X = Athena（athena_c_boron 取 **ln**）+ depth_um；目标 y = **ln(浓度)**。"
        "深度采样：**变化剧烈区域多取点**（按 |d ln c / d depth|），平坦区少取点；"
        "PySR 拟合使用过渡区加权（weights）。搜索目标为 **MSE**（loss）；"
        "公式排序以 **test_MSE / test_MAE** 为主，test_R² 为辅。"
    ),
    "iv_trio_ff": (
        "FF 替代方案 A：用 **实测** iv_Voc、iv_Jsc、iv_Eff（原尺度）作输入，SR 拟合 iv_FF。"
    ),
    "ff_identity": (
        "FF 替代方案 B（直接计算）：FF_calc = 100 × iv_Eff / (iv_Voc × iv_Jsc)，"
        "其中 iv_FF 单位为 %，iv_Voc 为 V，iv_Jsc 为 mA/cm²，iv_Eff 与 FF 同侧效率指标。"
    ),
}


def subsample_train(
    X: pd.DataFrame,
    y: np.ndarray,
    max_rows: int,
    weights: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y, weights
    rng = rng or np.random.default_rng(RANDOM_STATE)
    if weights is not None and len(weights) == len(X):
        p = weights / weights.sum()
        idx = rng.choice(len(X), size=max_rows, replace=False, p=p)
    else:
        idx = rng.choice(len(X), size=max_rows, replace=False)
    w_out = weights[idx] if weights is not None else None
    return X.iloc[idx].reset_index(drop=True), y[idx], w_out


def run_pysr_tabular_eval(
    X: pd.DataFrame,
    y: np.ndarray,
    task_name: str,
    preprocess_key: str,
) -> pd.DataFrame:
    train_idx, test_idx = split_train_test(len(X))
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    X_fit, y_fit, _ = subsample_train(X_train, y_train, SR_TABULAR_MAX_ROWS)
    print(
        f"  PySR {task_name}: fit {len(X_fit)} train rows "
        f"(train={len(X_train)}, test={len(X_test)}), loss=MSE"
    )

    model = build_pysr_regressor(f"tabular_{task_name}")
    model.fit(X_fit.values, y_fit)
    eq_df = model.equations_.copy()
    eq_df = annotate_equation_metrics(model, eq_df, X_train.values, y_train, X_test.values, y_test)

    target_suffix = task_name.split("_to_")[-1]
    eq_df["task"] = task_name
    eq_df["target"] = target_suffix
    eq_df["n_train"] = len(X_train)
    eq_df["n_test"] = len(X_test)
    eq_df["n_fit"] = len(X_fit)
    eq_df["preprocess_key"] = preprocess_key
    eq_df["preprocess_note"] = PREPROCESS_NOTES.get(preprocess_key, "")
    return eq_df


def run_pysr_curve_eval(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    task_name: str,
    sample_weights: np.ndarray | None = None,
    *,
    max_fit_rows: int | None = None,
) -> pd.DataFrame:
    train_idx, test_idx = split_group_train_test(groups)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]
    w_train = sample_weights[train_idx] if sample_weights is not None else None

    fit_cap = SR_CURVE_MAX_FIT_ROWS if max_fit_rows is None else max_fit_rows
    X_fit, y_fit, w_fit = subsample_train(X_train, y_train, fit_cap, w_train)
    weight_note = "uniform weights" if w_fit is not None and np.allclose(w_fit, w_fit[0]) else "adaptive weights"
    print(
        f"  PySR curve {task_name}: fit {len(X_fit)} pts "
        f"(train={len(X_train)}, test={len(X_test)}), {weight_note}, loss=MSE"
    )

    model = build_pysr_regressor(task_name)
    if w_fit is not None:
        model.fit(X_fit.values, y_fit, weights=w_fit)
    else:
        model.fit(X_fit.values, y_fit)
    eq_df = model.equations_.copy()
    eq_df = annotate_equation_metrics(model, eq_df, X_train.values, y_train, X_test.values, y_test)
    eq_df["target"] = "ln_curve"
    eq_df["n_train"] = len(X_train)
    eq_df["n_test"] = len(X_test)
    eq_df["n_fit"] = len(X_fit)
    eq_df["preprocess_key"] = "curve"
    eq_df["preprocess_note"] = PREPROCESS_NOTES["curve"]
    return eq_df
