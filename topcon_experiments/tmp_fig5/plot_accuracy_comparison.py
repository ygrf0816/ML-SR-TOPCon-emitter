"""SR vs LLM-simplified vs AutoGluon accuracy comparison plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.common.variable_labels import label_for
from topcon_experiments.config import OUTPUT_ROOT
from topcon_experiments.tmp_fig5.export_fig5_table import build_paper_table
from topcon_experiments.tmp_fig5.sr_llm_comparison import PAPER_TASK_GROUPS, _short_task_label
from topcon_experiments.tmp_fig5.utils import TMP_OUT, load_xy_for_task

EXP1_OUT = OUTPUT_ROOT / "exp1_forward"
PLOT_OUT = OUTPUT_ROOT / "exp4_symbolic" / "sr_plots"


def _task_to_autogluon_target(task: str) -> str | None:
    if task.startswith("athena_to_iv_"):
        return f"iv_{task.replace('athena_to_iv_', '')}"
    if task.startswith("full_to_iv_"):
        return f"iv_{task.replace('full_to_iv_', '')}"
    if task.startswith("athena_to_doping_"):
        return "doping_" + task.replace("athena_to_doping_", "").replace("_log", "")
    if task.startswith("athena_to_defect_vac_"):
        return "defect_vac_" + task.replace("athena_to_defect_vac_", "").replace("_log", "")
    return None


def _load_autogluon_test_metrics() -> pd.DataFrame:
    frames = []
    for model, path in [
        ("model1", EXP1_OUT / "metrics_model1.csv"),
        ("model2", EXP1_OUT / "metrics_model2.csv"),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["split"] == "test"].copy()
        df["autogluon_model"] = model
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _task_rmse_norm(task: str, mse: float) -> float:
    if not np.isfinite(mse):
        return float("nan")
    try:
        _, y, _ = load_xy_for_task(task)
        std = float(np.std(y))
        if std <= 0:
            return float("nan")
        return float(np.sqrt(mse) / std)
    except Exception:
        return float("nan")


def build_three_way_table() -> pd.DataFrame:
    sr_llm = build_paper_table()
    if sr_llm.empty:
        return pd.DataFrame()

    ag = _load_autogluon_test_metrics()
    ag_map = {}
    if not ag.empty:
        for _, row in ag.iterrows():
            ag_map[str(row["target"])] = row

    rows: list[dict] = []
    task_order: list[str] = []
    for group, tasks in PAPER_TASK_GROUPS.items():
        for task in tasks:
            hit = sr_llm[sr_llm["task"] == task]
            if hit.empty:
                continue
            task_order.append(task)
            ag_target = _task_to_autogluon_target(task)
            ag_row = ag_map.get(ag_target) if ag_target else None

            sr_r2 = float(hit["sr_test_R2"].iloc[0])
            sr_mse = float(hit["sr_test_MSE"].iloc[0])
            llm_r2 = float(hit["llm_test_R2"].iloc[0])
            llm_mse = float(hit["llm_test_MSE"].iloc[0])

            ag_r2 = float(ag_row["R2"]) if ag_row is not None else float("nan")
            ag_rmse = float(ag_row["RMSE_primary"]) if ag_row is not None else float("nan")
            ag_mae = float(ag_row["MAE_primary"]) if ag_row is not None else float("nan")
            ag_nrmse_pct = float(ag_row["nRMSE_pct"]) if ag_row is not None else float("nan")

            rows.append({
                "group": group,
                "task": task,
                "label": _short_task_label(task),
                "display_label": f"{group}: {_short_task_label(task)}",
                "autogluon_target": ag_target or "",
                "sr_R2": sr_r2,
                "sr_RMSE": float(np.sqrt(sr_mse)) if np.isfinite(sr_mse) and sr_mse >= 0 else float("nan"),
                "sr_nRMSE": _task_rmse_norm(task, sr_mse),
                "llm_R2": llm_r2,
                "llm_RMSE": float(np.sqrt(llm_mse)) if np.isfinite(llm_mse) and llm_mse >= 0 else float("nan"),
                "llm_nRMSE": _task_rmse_norm(task, llm_mse),
                "autogluon_R2": ag_r2,
                "autogluon_RMSE": ag_rmse,
                "autogluon_MAE": ag_mae,
                "autogluon_nRMSE_pct": ag_nrmse_pct,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        order_map = {t: i for i, t in enumerate(task_order)}
        df["_ord"] = df["task"].map(order_map)
        df = df.sort_values("_ord").drop(columns="_ord")
    return df


def _plot_combo(df: pd.DataFrame, out_png: Path, *, title_suffix: str = "") -> None:
    if df.empty:
        return
    apply_plot_style()
    n = len(df)
    x = np.arange(n)
    w = 0.26

    fig, ax1 = plt.subplots(figsize=(max(14, 0.55 * n), 6.5))
    ax1.bar(x - w, df["sr_R2"], w, label="PySR (pre-LLM)", color="#440154", alpha=0.9)
    ax1.bar(x, df["llm_R2"], w, label="LLM simplified", color="#21908C", alpha=0.9)
    ax1.bar(x + w, df["autogluon_R2"], w, label="AutoGluon", color="#FDE725", edgecolor="#666", alpha=0.95)

    ax1.set_ylabel("test R²")
    ax1.set_ylim(0, 1.08)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
    ax1.set_title(f"Prediction accuracy: PySR vs LLM-simplified vs AutoGluon{title_suffix}")
    ax1.axhline(0, color="0.5", lw=0.6)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, df["sr_nRMSE"], "o-", color="#440154", lw=1.5, ms=5, label="PySR nRMSE")
    ax2.plot(x, df["llm_nRMSE"], "s-", color="#21908C", lw=1.5, ms=5, label="LLM nRMSE")
    ag_nrmse = df["autogluon_nRMSE_pct"] / 100.0
    ax2.plot(x, ag_nrmse, "^-", color="#CC9900", lw=1.5, ms=5, label="AutoGluon nRMSE")

    ax2.set_ylabel("normalized RMSE (line)")
    ymax = np.nanmax(np.concatenate([
        df["sr_nRMSE"].values, df["llm_nRMSE"].values, ag_nrmse.values,
    ]))
    ax2.set_ylim(0, max(0.15, float(ymax) * 1.25) if np.isfinite(ymax) else 0.15)

    # group separators
    prev_group = None
    for i, g in enumerate(df["group"]):
        if prev_group is not None and g != prev_group:
            ax1.axvline(i - 0.5, color="0.75", ls="--", lw=0.8)
        prev_group = g

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_by_group(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    written: list[Path] = []
    for group, sub in df.groupby("group", sort=False):
        safe = group.replace(" ", "_").replace("(", "").replace(")", "").replace("→", "to")
        out = out_dir / f"accuracy_compare_{safe}.png"
        _plot_combo(sub.reset_index(drop=True), out, title_suffix=f" — {group}")
        written.append(out)
    return written


def run(out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or TMP_OUT
    data_dir = out_dir / "data"
    plot_dir = out_dir / "plots"
    sr_plot_dir = PLOT_OUT
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    sr_plot_dir.mkdir(parents=True, exist_ok=True)

    df = build_three_way_table()
    if df.empty:
        return []

    csv_path = data_dir / "sr_llm_autogluon_accuracy_compare.csv"
    save_csv(df, csv_path)
    save_csv(df, sr_plot_dir / "sr_llm_autogluon_accuracy_compare.csv")

    written = [csv_path]
    all_png = plot_dir / "accuracy_sr_llm_autogluon_all.png"
    _plot_combo(df, all_png)
    written.append(all_png)

    sr_all_png = sr_plot_dir / "sr_llm_autogluon_accuracy_all.png"
    _plot_combo(df, sr_all_png)
    written.append(sr_all_png)

    for p in _plot_by_group(df, plot_dir):
        written.append(p)
    for p in _plot_by_group(df, sr_plot_dir):
        written.append(p)

    return written


if __name__ == "__main__":
    paths = run()
    for p in paths:
        print(p)
