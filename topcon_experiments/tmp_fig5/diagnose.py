"""Coverage diagnostics for tmp Fig.5 plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import ATHENA_FEATURES
from topcon_experiments.tmp_fig5.utils import (
    ATHENA_DISPLAY,
    EXP4_OUT,
    IV_TASKS,
    TMP_OUT,
    best_univariate_row,
    llm_simplify_row,
    llm_text_row,
)


def build_coverage_report(out_dir: Path | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for iv_key, short in IV_TASKS.items():
        task = f"athena_to_{iv_key}"
        sr_path = EXP4_OUT / f"sr_tabular_{task}.csv"
        if not sr_path.exists():
            continue
        sr_df = pd.read_csv(sr_path)
        simp_df = pd.read_csv(EXP4_OUT / f"llm_simplified_tabular_{task}.csv") if (EXP4_OUT / f"llm_simplified_tabular_{task}.csv").exists() else pd.DataFrame()
        blind_df = pd.read_csv(EXP4_OUT / f"llm_blind_math_tabular_{task}.csv") if (EXP4_OUT / f"llm_blind_math_tabular_{task}.csv").exists() else pd.DataFrame()
        phys_df = pd.read_csv(EXP4_OUT / f"llm_physics_tabular_{task}.csv") if (EXP4_OUT / f"llm_physics_tabular_{task}.csv").exists() else pd.DataFrame()

        for fi, col in enumerate(ATHENA_FEATURES):
            feat = ATHENA_DISPLAY[col]
            uni = best_univariate_row(sr_df, fi)
            multi_contain = sr_df[sr_df["equation"].astype(str).str.contains(rf"\bx{fi}\b", regex=True)]
            eq = str(uni.get("equation", "")) if uni is not None else ""
            rows.append({
                "target": short,
                "feature": feat,
                "feature_col": col,
                "has_univariate_sr": uni is not None,
                "has_multivariate_using_feat": not multi_contain.empty,
                "n_multivariate_formulas": len(multi_contain),
                "univariate_test_R2": float(uni.get("test_R2", float("nan"))) if uni is not None else float("nan"),
                "has_llm_simplify": llm_simplify_row(simp_df, eq) is not None if eq else False,
                "blind_math_len": len(llm_text_row(blind_df, eq, "blind_math")) if eq else 0,
                "physics_len": len(llm_text_row(phys_df, eq, "physics")) if eq else 0,
                "status": _status(uni, simp_df, blind_df, phys_df, eq),
            })

    df = pd.DataFrame(rows)
    out_dir = out_dir or TMP_OUT
    save_csv(df, out_dir / "data" / "coverage_report.csv")
    return df


def _status(uni, simp_df, blind_df, phys_df, eq: str) -> str:
    if uni is None:
        return "missing_univariate_sr"
    if not eq:
        return "missing_equation"
    if llm_simplify_row(simp_df, eq) is None:
        return "missing_llm_simplify"
    if len(llm_text_row(blind_df, eq, "blind_math")) < 80:
        return "short_blind_math"
    if len(llm_text_row(phys_df, eq, "physics")) < 80:
        return "short_physics"
    return "ok"


def summarize_coverage(df: pd.DataFrame) -> dict[str, float]:
    n = len(df)
    if n == 0:
        return {}
    return {
        "cells_total": n,
        "univariate_sr_pct": 100 * df["has_univariate_sr"].mean(),
        "llm_simplify_pct": 100 * df["has_llm_simplify"].mean(),
        "semantic_ready_pct": 100 * (df["status"] == "ok").mean(),
    }
