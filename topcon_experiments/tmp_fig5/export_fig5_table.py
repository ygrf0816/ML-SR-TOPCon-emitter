"""Export SR vs LLM-simplified comparison table for paper Figure 5."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.tmp_fig5.sr_llm_comparison import build_global_comparison_table
from topcon_experiments.tmp_fig5.utils import TMP_OUT

COMPLEXITY_NOTE = (
    "Complexity = SymPy count_ops(expr, visual=False) on the evaluated expression "
    "(sympy_format for PySR; materialized numeric form for LLM-simplified when symbolic "
    "parameters a,b,c cannot be resolved). Each arithmetic operator (+, -, *, /, ^), "
    "function call (log, exp, sqrt, abs, square, inv), and power counts as one operation; "
    "constants and variables do not. Fallback if sympify fails: count of operators plus x-index tokens."
)


def build_paper_table() -> pd.DataFrame:
    raw = build_global_comparison_table()
    raw = raw[raw["method"].isin(("SR", "LLM_simplified"))].copy()
    if raw.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for task, grp in raw.groupby("task"):
        sr = grp[grp["method"] == "SR"]
        llm = grp[grp["method"] == "LLM_simplified"]
        if sr.empty:
            continue
        s = sr.iloc[0]
        l = llm.iloc[0] if not llm.empty else None

        sr_c = int(s.get("complexity", 0))
        llm_c = int(l.get("complexity", 0)) if l is not None else None
        delta_c = sr_c - llm_c if llm_c is not None else None
        delta_r2 = (
            float(l.get("test_R2", float("nan"))) - float(s.get("test_R2", float("nan")))
            if l is not None
            else float("nan")
        )

        rows.append({
            "task": task,
            "task_label": s.get("task_label", task),
            "sr_equation_pysr": s.get("equation", ""),
            "sr_expr_eval": s.get("expr_eval", ""),
            "sr_complexity": sr_c,
            "sr_test_R2": float(s.get("test_R2", float("nan"))),
            "sr_test_MSE": float(s.get("test_MSE", float("nan"))),
            "llm_simplified_formula": l.get("equation", "") if l is not None else "",
            "llm_expr_eval": l.get("expr_eval", "") if l is not None else "",
            "llm_complexity": llm_c,
            "llm_test_R2": float(l.get("test_R2", float("nan"))) if l is not None else float("nan"),
            "llm_test_MSE": float(l.get("test_MSE", float("nan"))) if l is not None else float("nan"),
            "complexity_reduction": delta_c,
            "complexity_reduction_pct": 100 * delta_c / sr_c if delta_c is not None and sr_c else None,
            "delta_test_R2": delta_r2,
            "llm_simplification_notes": l.get("simplification_notes", "") if l is not None else "",
        })

    return pd.DataFrame(rows)


def main() -> None:
    out_dir = TMP_OUT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl = build_paper_table()
    save_csv(tbl, out_dir / "fig5_sr_llm_comparison_table.csv")

    note_path = out_dir / "fig5_complexity_definition.txt"
    note_path.write_text(COMPLEXITY_NOTE + "\n", encoding="utf-8")
    print(f"Wrote {len(tbl)} rows -> {out_dir / 'fig5_sr_llm_comparison_table.csv'}")
    print(f"Complexity note -> {note_path}")


if __name__ == "__main__":
    main()
