"""Panel A: SR vs LLM-simplified comparison for joint (multivariate) SR results."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import ATHENA_FEATURES, IV_TARGETS, OUTPUT_ROOT
from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy
from topcon_experiments.tmp_fig5.utils import (
    ATHENA_DISPLAY,
    EXP4_OUT,
    IV_TASKS,
    TMP_OUT,
    eval_expr_metrics,
    extract_json_block,
    formula_complexity,
    llm_simplify_row,
    load_xy_for_task,
    materialize_simplified_expr,
    task_label,
)

SKIP_SR = {"sr_tabular_formulas_summary.csv", "sr_tabular_full_iv_summary.csv"}

# Paper-facing task groups (global best joint formula per task)
PAPER_TASK_GROUPS: dict[str, list[str]] = {
    "IV (Athena)": [
        "athena_to_iv_Eff",
        "athena_to_iv_Voc",
        "athena_to_iv_Jsc",
        "athena_to_iv_FF",
    ],
    "IV (full model)": [
        "full_to_iv_Eff",
        "full_to_iv_Voc",
        "full_to_iv_Jsc",
        "full_to_iv_FF",
    ],
    "Doping": [
        "athena_to_doping_N_peak_log",
        "athena_to_doping_dose_log",
        "athena_to_doping_FWHM",
        "athena_to_doping_junction_depth",
        "athena_to_doping_R_sheet",
    ],
    "Defect": [
        "athena_to_defect_vac_dose_log",
        "athena_to_defect_vac_N_peak_log",
    ],
    "Curves": ["doping_curve_tail_full", "defect_curve"],
}

IV_SHORT = {"athena_to_iv_Eff": "PCE", "athena_to_iv_Voc": "Voc", "athena_to_iv_Jsc": "Jsc", "athena_to_iv_FF": "FF"}


def _compare_one_task(task: str, sr_path: Path) -> list[dict]:
    sr_df = pd.read_csv(sr_path)
    sr_best = sort_formulas_by_accuracy(sr_df).iloc[0]
    simp_path = EXP4_OUT / f"llm_simplified_tabular_{task}.csv"
    simp_df = pd.read_csv(simp_path) if simp_path.exists() else pd.DataFrame()

    X, y, n_feat = load_xy_for_task(task)
    sr_eq = str(sr_best.get("sympy_format") or sr_best.get("equation", ""))
    sr_m = eval_expr_metrics(sr_eq, X, y)
    sr_row = {
        "task": task,
        "task_label": task_label(task),
        "method": "SR",
        "equation": str(sr_best.get("equation", "")),
        "expr_eval": sr_eq,
        "complexity": int(sr_best.get("complexity", formula_complexity(sr_eq, n_feat))),
        "test_R2": float(sr_best.get("test_R2", sr_m["test_R2"])),
        "test_MSE": float(sr_best.get("test_MSE", sr_m["test_MSE"])),
    }

    simp_llm = llm_simplify_row(simp_df, str(sr_best.get("equation", "")))
    if simp_llm is None and not simp_df.empty:
        simp_llm = simp_df.iloc[0]
    parsed = extract_json_block(str(simp_llm.get("response", ""))) if simp_llm is not None else None
    simp_expr = materialize_simplified_expr(parsed or {}, n_feat, sr_expr_fallback=sr_eq) if parsed else sr_eq
    simp_m = eval_expr_metrics(simp_expr, X, y)
    simp_row = {
        "task": task,
        "task_label": task_label(task),
        "method": "LLM_simplified",
        "equation": str(parsed.get("simplified_formula", "")) if parsed else "",
        "expr_eval": simp_expr,
        "complexity": formula_complexity(simp_expr, n_feat),
        "test_R2": simp_m["test_R2"],
        "test_MSE": simp_m["test_MSE"],
        "simplification_notes": str(parsed.get("simplification_notes", "")) if parsed else "",
    }
    return [sr_row, simp_row]


def build_global_comparison_table() -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(EXP4_OUT.glob("sr_tabular_*.csv")):
        if path.name in SKIP_SR:
            continue
        task = path.stem.replace("sr_tabular_", "")
        try:
            rows.extend(_compare_one_task(task, path))
        except Exception as exc:
            rows.append({"task": task, "method": "error", "notes": str(exc)})

    # curves
    from topcon_experiments.config import SR_DOPING_CURVE_FORMULA_PATH, SR_DEFECT_CURVE_FORMULA_PATH

    for curve, fpath in [("doping_curve_tail_full", SR_DOPING_CURVE_FORMULA_PATH), ("defect_curve", SR_DEFECT_CURVE_FORMULA_PATH)]:
        if not fpath.exists():
            continue
        sr_df = pd.read_csv(fpath)
        sr_best = sort_formulas_by_accuracy(sr_df).iloc[0]
        src = "doping" if "doping" in curve else "defect"
        simp_df = pd.read_csv(EXP4_OUT / f"llm_simplified_{src}.csv") if (EXP4_OUT / f"llm_simplified_{src}.csv").exists() else pd.DataFrame()
        eq = str(sr_best.get("equation", ""))
        rows.append({
            "task": curve,
            "task_label": curve,
            "method": "SR",
            "equation": eq,
            "expr_eval": str(sr_best.get("sympy_format") or eq),
            "complexity": int(sr_best.get("complexity", np.nan)),
            "test_R2": float(sr_best.get("test_R2", np.nan)),
            "test_MSE": float(sr_best.get("test_MSE", np.nan)),
        })
        simp_llm = llm_simplify_row(simp_df, eq)
        parsed = extract_json_block(str(simp_llm.get("response", ""))) if simp_llm is not None else None
        simp_expr = materialize_simplified_expr(parsed or {}, 18) if parsed else eq
        rows.append({
            "task": curve,
            "task_label": curve,
            "method": "LLM_simplified",
            "equation": str(parsed.get("simplified_formula", "")) if parsed else "",
            "expr_eval": simp_expr,
            "complexity": formula_complexity(simp_expr, 18),
            "test_R2": float(sr_best.get("test_R2", np.nan)),
            "test_MSE": float(sr_best.get("test_MSE", np.nan)),
            "simplification_notes": str(parsed.get("simplification_notes", "")) if parsed else "",
        })
    return pd.DataFrame(rows)


def _short_task_label(task: str) -> str:
    if task in IV_SHORT:
        return IV_SHORT[task]
    if task.startswith("full_to_iv_"):
        return "full→" + IV_SHORT.get("athena_to_" + task.replace("full_to_", ""), task)
    if task.startswith("athena_to_doping_"):
        return task.replace("athena_to_doping_", "").replace("_log", "")
    if task.startswith("athena_to_defect_"):
        return task.replace("athena_to_defect_vac_", "").replace("_log", "")
    if task == "doping_curve_tail_full":
        return "doping tail"
    if task == "defect_curve":
        return "defect curve"
    return task_label(task)


def _plot_iv_sr_llm_bars(tbl: pd.DataFrame, out_png: Path) -> None:
    """Optimization summary: complexity drop (bars) + preserved R² (labels)."""
    iv_tasks = PAPER_TASK_GROUPS["IV (Athena)"]
    sub = tbl[tbl["task"].isin(iv_tasks)].copy()
    if sub.empty:
        return
    sub["label"] = sub["task"].map(_short_task_label)
    order = [_short_task_label(t) for t in iv_tasks]
    piv_c = sub.pivot_table(index="label", columns="method", values="complexity", aggfunc="first").reindex(order)
    piv_r2 = sub.pivot_table(index="label", columns="method", values="test_R2", aggfunc="first").reindex(order)

    c_sr = piv_c.get("SR", pd.Series(dtype=float))
    c_llm = piv_c.get("LLM_simplified", pd.Series(dtype=float))
    r2_sr = piv_r2.get("SR", pd.Series(dtype=float))
    drop_pct = 100 * (c_sr - c_llm) / c_sr.replace(0, np.nan)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(drop_pct))
    colors = ["#21908C" if v >= 0 else "#440154" for v in drop_pct.fillna(0)]
    bars = ax.bar(x, drop_pct.fillna(0), color=colors, edgecolor="0.2", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(drop_pct.index)
    ax.set_ylabel("complexity reduction (%)")
    ax.set_title("LLM simplification of joint IV formulas")
    ax.axhline(0, color="0.7", lw=0.8)

    for i, bar in enumerate(bars):
        r2 = r2_sr.iloc[i] if i < len(r2_sr) else float("nan")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"R²={r2:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height() * 0.45, 2),
            f"{int(c_sr.iloc[i])}→{int(c_llm.iloc[i])}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_complexity_reduction(tbl: pd.DataFrame, out_png: Path) -> None:
    """SR vs LLM complexity for IV-athena (shows simplification without accuracy loss)."""
    iv_tasks = PAPER_TASK_GROUPS["IV (Athena)"]
    sub = tbl[tbl["task"].isin(iv_tasks)].copy()
    if sub.empty:
        return
    sub["label"] = sub["task"].map(_short_task_label)
    order = [_short_task_label(t) for t in iv_tasks]
    piv_c = sub.pivot_table(index="label", columns="method", values="complexity", aggfunc="first").reindex(order)
    piv_r2 = sub.pivot_table(index="label", columns="method", values="test_R2", aggfunc="first").reindex(order)

    apply_plot_style()
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(piv_c))
    w = 0.35
    c_sr = piv_c.get("SR", pd.Series(dtype=float))
    c_llm = piv_c.get("LLM_simplified", pd.Series(dtype=float))
    bars1 = ax1.bar(x - w / 2, c_sr, w, label="PySR complexity", color="#3B528B")
    bars2 = ax1.bar(x + w / 2, c_llm, w, label="LLM complexity", color="#5DC863")
    ax1.set_xticks(x)
    ax1.set_xticklabels(piv_c.index)
    ax1.set_ylabel("formula complexity")
    ax1.set_title("LLM simplification preserves R² while reducing complexity")

    ax2 = ax1.twinx()
    r2_sr = piv_r2.get("SR", pd.Series(dtype=float))
    r2_llm = piv_r2.get("LLM_simplified", pd.Series(dtype=float))
    ax2.plot(x, r2_sr, "o--", color="#440154", label="PySR R²", markersize=6)
    ax2.plot(x, r2_llm, "s--", color="#21908C", label="LLM R²", markersize=6)
    ax2.set_ylabel("test R²")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_all_tasks_heatmap(tbl: pd.DataFrame, out_png: Path) -> None:
    """Supplementary: all tasks SR vs LLM R² (rows grouped by category)."""
    rows: list[dict] = []
    for group, tasks in PAPER_TASK_GROUPS.items():
        for task in tasks:
            hit = tbl[(tbl["task"] == task) & (tbl["method"] == "SR")]
            if hit.empty:
                continue
            label = _short_task_label(task)
            for method in ("SR", "LLM_simplified"):
                m = tbl[(tbl["task"] == task) & (tbl["method"] == method)]
                if m.empty:
                    continue
                rows.append({
                    "group": group,
                    "task": f"{group}: {label}",
                    "method": method,
                    "test_R2": float(m.iloc[0]["test_R2"]),
                    "complexity": float(m.iloc[0].get("complexity", np.nan)),
                })
    if not rows:
        return
    df = pd.DataFrame(rows)
    piv_r2 = df.pivot_table(index="task", columns="method", values="test_R2", aggfunc="first")
    piv_c = df.pivot_table(index="task", columns="method", values="complexity", aggfunc="first")

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(piv_r2)), max(6, 0.35 * len(piv_r2))))
    annot = piv_c.astype(object).copy()
    for i in range(len(annot)):
        for j in range(len(annot.columns)):
            v = piv_c.iloc[i, j]
            annot.iloc[i, j] = "" if pd.isna(v) else f"{int(v)}"
    sns.heatmap(
        piv_r2.astype(float),
        annot=annot,
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        cbar_kws={"label": "test R²"},
        ax=ax,
    )
    ax.set_title("SR vs LLM-simplified — joint best formula per task")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or TMP_OUT
    data_dir = out_dir / "data"
    plot_dir = out_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    global_tbl = build_global_comparison_table()
    save_csv(global_tbl, data_dir / "sr_vs_llm_all_tasks.csv")

    written = [data_dir / "sr_vs_llm_all_tasks.csv"]
    if not global_tbl.empty:
        ok = global_tbl[global_tbl["method"].isin(("SR", "LLM_simplified"))]
        _plot_iv_sr_llm_bars(ok, plot_dir / "panel_sr_llm_iv_athena.png")
        _plot_complexity_reduction(ok, plot_dir / "panel_complexity_reduction_iv.png")
        _plot_all_tasks_heatmap(ok, plot_dir / "panel_sr_llm_all_tasks.png")
        written.extend([
            plot_dir / "panel_sr_llm_iv_athena.png",
            plot_dir / "panel_complexity_reduction_iv.png",
            plot_dir / "panel_sr_llm_all_tasks.png",
        ])

    return written
