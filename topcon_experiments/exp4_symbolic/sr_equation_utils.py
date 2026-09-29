"""Evaluate PySR equations from CSV (sympy) on feature matrices."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import sympy as sp


def _sympy_symbols(n_features: int) -> tuple:
    return sp.symbols(" ".join(f"x{i}" for i in range(n_features)))


def eval_sympy_expr(expr_str: str, X: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    syms = _sympy_symbols(n)
    local = {f"x{i}": syms[i] for i in range(n)}
    expr = sp.sympify(str(expr_str), locals=local)
    fn = sp.lambdify(syms, expr, modules=["numpy"])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = fn(*[X[:, i] for i in range(n)])
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 0:
        arr = np.full(X.shape[0], float(arr))
    elif arr.shape[0] != X.shape[0]:
        arr = np.broadcast_to(arr, (X.shape[0],))
    return np.where(np.isfinite(arr), arr, np.nan)


from topcon_experiments.exp4_symbolic.sr_metrics import sort_formulas_by_metrics

# Backward-compatible alias
sort_formulas_by_accuracy = sort_formulas_by_metrics


def best_row_by_test_r2(path) -> pd.Series:
    df = pd.read_csv(path)
    return sort_formulas_by_metrics(df).iloc[0]


def best_formula_row(path) -> pd.Series:
    return best_row_by_test_r2(path)


def top_rows_by_test_r2(path, n: int = 3) -> pd.DataFrame:
    df = pd.read_csv(path)
    return sort_formulas_by_metrics(df).head(n).reset_index(drop=True)


def predict_from_row(row: pd.Series, X: np.ndarray) -> np.ndarray:
    expr = row.get("sympy_format") or row.get("equation")
    return eval_sympy_expr(str(expr), X)


def predict_from_sr_csv(path, X: np.ndarray) -> np.ndarray:
    return predict_from_row(best_row_by_test_r2(path), X)


def substitute_ff_composite(voc_expr: str, jsc_expr: str, eff_expr: str, n_features: int) -> tuple[str, str]:
    syms = _sympy_symbols(n_features)
    local = {f"x{i}": syms[i] for i in range(n_features)}
    voc = sp.sympify(voc_expr, locals=local)
    jsc = sp.sympify(jsc_expr, locals=local)
    eff = sp.sympify(eff_expr, locals=local)
    ff = sp.simplify(100 * eff / (voc * jsc))
    return str(ff), str(sp.simplify(ff))
