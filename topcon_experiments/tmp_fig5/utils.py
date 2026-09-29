"""Shared helpers for tmp Fig.5-style plots."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

from topcon_experiments.common.data import preprocess_model1, preprocess_model2
from topcon_experiments.config import (
    ATHENA_FEATURES,
    IV_TARGET_LABELS,
    MODEL2_FEATURES,
    OUTPUT_ROOT,
)
from topcon_experiments.exp4_symbolic.sr_equation_utils import eval_sympy_expr, sort_formulas_by_accuracy
from topcon_experiments.exp4_symbolic.sr_metrics import regression_metrics
from topcon_experiments.exp4_symbolic.sr_split_utils import split_train_test

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
TMP_OUT = OUTPUT_ROOT / "tmp_fig5"

ATHENA_DISPLAY: dict[str, str] = {
    "athena_c_boron": "C_B",
    "athena_thick": "Thick",
    "athena_temp1": "T1",
    "athena_time1": "t1",
    "athena_temp2": "T2",
    "athena_time2": "t2",
    "athena_F_N2": "F_n2",
    "athena_F_O2": "F_o2",
}

IV_TASKS = {
    "iv_Eff": "PCE",
    "iv_Voc": "Voc",
    "iv_Jsc": "Jsc",
    "iv_FF": "FF",
}


def extract_json_block(text: str) -> dict | None:
    if not isinstance(text, str) or not text.strip():
        return None
    for pattern in (
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
    ):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                import json

                return json.loads(m.group(1))
            except Exception:
                continue
    return None


def formula_uses_feature(equation: str, feat_idx: int) -> bool:
    return bool(re.search(rf"\bx{feat_idx}\b", str(equation)))


def formula_complexity(expr: str, n_features: int = 18) -> int:
    try:
        local = {f"x{i}": sp.Symbol(f"x{i}") for i in range(n_features)}
        local.update({c: sp.Symbol(c) for c in "abcdefghijklmnopqrstuvwxyz"})
        node = sp.sympify(str(expr), locals=local)
        return int(sp.count_ops(node, visual=False))
    except Exception:
        return int(len(re.findall(r"[+\-*/^()]", str(expr))) + str(expr).count("x"))


_PARAM_ASSIGN_RE = re.compile(
    r"\b([a-wyz])\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    flags=re.IGNORECASE,
)


def _clean_llm_expr(expr: str) -> str:
    s = str(expr).strip()
    if "from sympy" in s.lower() or " symbols(" in s:
        m = re.search(r"expr\s*=\s*(.+)$", s, re.DOTALL | re.IGNORECASE)
        if m:
            s = m.group(1).strip()
    return s.replace("^", "**")


def _constants_from_notes(notes: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in _PARAM_ASSIGN_RE.finditer(str(notes)):
        key = m.group(1).lower()
        out[key] = float(m.group(2))
    return out


def _expr_uses_param(expr: str, letter: str) -> bool:
    return bool(re.search(rf"\b{re.escape(letter)}\b", str(expr), flags=re.IGNORECASE))


def _remaining_param_symbols(expr_obj) -> set[str]:
    return {
        str(s)
        for s in expr_obj.free_symbols
        if re.fullmatch(r"[a-wyz]", str(s), flags=re.IGNORECASE)
    }


def materialize_simplified_expr(
    parsed: dict,
    n_features: int,
    *,
    sr_expr_fallback: str | None = None,
) -> str | None:
    if not parsed:
        return sr_expr_fallback
    raw = parsed.get("sympy_form") or parsed.get("simplified_formula")
    if not raw:
        return sr_expr_fallback
    expr = _clean_llm_expr(raw)
    consts = _constants_from_notes(parsed.get("simplification_notes", ""))
    used_consts = {k: v for k, v in consts.items() if _expr_uses_param(expr, k)}
    if used_consts:
        local = {k: sp.Float(v) for k, v in used_consts.items()}
        local.update({f"x{i}": sp.Symbol(f"x{i}") for i in range(n_features)})
        try:
            mat = sp.sympify(expr, locals=local)
            stray = _remaining_param_symbols(mat)
            if stray:
                if sr_expr_fallback:
                    return sr_expr_fallback
            return str(mat)
        except Exception:
            pass
    if re.search(r"\b[a-wyz]\b", expr, flags=re.IGNORECASE) and sr_expr_fallback:
        return sr_expr_fallback
    return expr


def load_xy_for_task(task: str) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Return X, y, n_features for a tabular SR task name."""
    if task.startswith("athena_to_iv_"):
        target = task.replace("athena_to_iv_", "")
        X_full, y_map, _ = preprocess_model2()
        X = X_full[ATHENA_FEATURES].copy()
        y = y_map[f"iv_{target}" if not target.startswith("iv_") else target].values.astype(float)
        return X, y, len(ATHENA_FEATURES)
    if task.startswith("full_to_iv_"):
        target = task.replace("full_to_iv_", "")
        X_full, y_map, _ = preprocess_model2()
        y = y_map[f"iv_{target}" if not target.startswith("iv_") else target].values.astype(float)
        return X_full, y, len(MODEL2_FEATURES)
    if task.startswith("athena_to_"):
        X1, y_map, _ = preprocess_model1()
        raw = task.replace("athena_to_", "").replace("_log", "")
        key = raw if raw in y_map else f"{raw}_log" if f"{raw}_log" in y_map else raw
        for k in y_map:
            if k.replace("_log", "") == raw.replace("_log", ""):
                key = k
                break
        y = y_map[key].values.astype(float)
        return X1, y, len(ATHENA_FEATURES)
    raise ValueError(f"unsupported task: {task}")


def eval_expr_metrics(expr: str, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    train_idx, test_idx = split_train_test(len(X))
    X_test = X.iloc[test_idx].values.astype(float)
    y_test = y[test_idx]
    try:
        pred = eval_sympy_expr(expr, X_test)
        m = regression_metrics(y_test, pred)
        return {"test_R2": m["R2"], "test_MSE": m["MSE"], "test_MAE": m["MAE"]}
    except Exception:
        return {"test_R2": float("nan"), "test_MSE": float("nan"), "test_MAE": float("nan")}


def formula_var_indices(equation: str) -> set[int]:
    return {int(m.group(1)) for m in re.finditer(r"\bx(\d+)\b", str(equation))}


def best_univariate_row(sr_df: pd.DataFrame, feat_idx: int) -> pd.Series | None:
    """Best formula that uses ONLY x{feat_idx} (single-feature SR style)."""
    hits = sr_df[
        sr_df["equation"].astype(str).apply(lambda e: formula_var_indices(e) == {feat_idx})
    ]
    if hits.empty:
        return None
    return sort_formulas_by_accuracy(hits).iloc[0]


def best_row_using_feature(sr_df: pd.DataFrame, feat_idx: int) -> pd.Series | None:
    """Deprecated for Fig.5: returns multi-feature formula containing xi (misleading)."""
    hits = sr_df[sr_df["equation"].astype(str).apply(lambda e: formula_uses_feature(e, feat_idx))]
    if hits.empty:
        return None
    return sort_formulas_by_accuracy(hits).iloc[0]


def llm_simplify_row(simp_df: pd.DataFrame, equation: str) -> pd.Series | None:
    if simp_df.empty:
        return None
    hits = simp_df[simp_df["equation"].astype(str) == str(equation)]
    if hits.empty:
        return None
    return hits.iloc[0]


def llm_text_row(llm_df: pd.DataFrame, equation: str, analysis_type: str) -> str:
    if llm_df.empty:
        return ""
    sub = llm_df[llm_df["equation"].astype(str) == str(equation)]
    if "analysis_type" in sub.columns:
        typed = sub[sub["analysis_type"].astype(str) == analysis_type]
        if not typed.empty:
            sub = typed
    if sub.empty:
        return ""
    return str(sub.iloc[0].get("response", "")).strip()


def task_label(task: str) -> str:
    if task.startswith("athena_to_iv_"):
        t = task.replace("athena_to_iv_", "")
        return IV_TARGET_LABELS.get(f"iv_{t}" if not t.startswith("iv_") else t, t)
    return task.replace("athena_to_", "").replace("full_to_", "full->")
