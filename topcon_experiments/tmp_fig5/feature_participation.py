"""Which Athena features appear in the best joint SR formula per IV target."""

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
from topcon_experiments.config import ATHENA_FEATURES
from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy
from topcon_experiments.tmp_fig5.utils import (
    ATHENA_DISPLAY,
    EXP4_OUT,
    IV_TASKS,
    TMP_OUT,
    formula_var_indices,
)


def build_participation_table() -> pd.DataFrame:
    rows: list[dict] = []
    feat_labels = [ATHENA_DISPLAY[c] for c in ATHENA_FEATURES]

    for iv_key, short in IV_TASKS.items():
        task = f"athena_to_{iv_key}"
        path = EXP4_OUT / f"sr_tabular_{task}.csv"
        if not path.exists():
            continue
        sr_df = pd.read_csv(path)
        best = sort_formulas_by_accuracy(sr_df).iloc[0]
        eq = str(best.get("equation", ""))
        used = formula_var_indices(eq)
        for fi, col in enumerate(ATHENA_FEATURES):
            rows.append({
                "target": short,
                "feature": feat_labels[fi],
                "feature_col": col,
                "feature_idx": fi,
                "in_best_formula": int(fi in used),
                "best_equation": eq,
                "best_test_R2": float(best.get("test_R2", np.nan)),
                "best_complexity": int(best.get("complexity", np.nan)),
            })

    return pd.DataFrame(rows)


def participation_matrix(detail: pd.DataFrame) -> pd.DataFrame:
    mat = detail.pivot_table(
        index="feature",
        columns="target",
        values="in_best_formula",
        aggfunc="max",
    )
    col_order = list(IV_TASKS.values())
    mat = mat.reindex(columns=[c for c in col_order if c in mat.columns])
    feat_order = [ATHENA_DISPLAY[c] for c in ATHENA_FEATURES]
    return mat.reindex(feat_order)


def plot_participation_heatmap(mat: pd.DataFrame, out_png: Path) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(max(5.5, 1.1 * len(mat.columns)), max(5, 0.55 * len(mat))))
    annot = mat.astype(object).copy()
    for i in range(len(annot)):
        for j in range(len(annot.columns)):
            annot.iloc[i, j] = "yes" if mat.iloc[i, j] else ""

    sns.heatmap(
        mat.astype(float),
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        cbar_kws={"label": "feature in best joint formula"},
        ax=ax,
    )
    ax.set_title("Athena feature roles in best joint IV formulas")
    ax.set_ylabel("Process parameter")
    ax.set_xlabel("IV target")
    plt.setp(ax.get_xticklabels(), rotation=0)
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

    detail = build_participation_table()
    mat = participation_matrix(detail)
    save_csv(detail, data_dir / "iv_feature_participation_detail.csv")
    save_csv(mat.reset_index().rename(columns={"index": "feature"}), data_dir / "iv_feature_participation_matrix.csv")
    png = plot_dir / "panel_feature_participation_iv.png"
    plot_participation_heatmap(mat, png)
    return [data_dir / "iv_feature_participation_detail.csv", png]
