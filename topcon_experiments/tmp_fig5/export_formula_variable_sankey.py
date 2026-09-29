"""Export Origin-ready Sankey data from all ranked SR formulas.

Weights:
    rank_weight = 1 / rank
    weighted_presence_score = rank_weight if a variable appears in the formula
    weighted_occurrence_score = rank_weight * occurrence_count(xi)

The Origin-facing edge table uses weighted_occurrence_share by default so each
task contributes a total flow of 1.0 and widths compare variable share within a
task. Raw scores are kept in adjacent columns.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy
from topcon_experiments.exp4_symbolic.sr_variable_map import PHYSICAL_LABEL_ZH
from topcon_experiments.tmp_fig5.plot_physics_meaning_map import FEATURE_LABELS, _feature_order, _formula_path
from topcon_experiments.tmp_fig5.sr_llm_comparison import PAPER_TASK_GROUPS, _short_task_label
from topcon_experiments.tmp_fig5.utils import TMP_OUT, formula_var_indices


def _paper_tasks() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for group, tasks in PAPER_TASK_GROUPS.items():
        rows.extend((group, task) for task in tasks)
    return rows


def _var_occurrences(equation: str, idx: int) -> int:
    return len(re.findall(rf"\bx{idx}\b", str(equation)))


def _metric(row: pd.Series, col: str) -> float:
    try:
        return float(row.get(col, np.nan))
    except Exception:
        return float("nan")


def build_formula_variable_tables(top_n: int | None = None) -> dict[str, pd.DataFrame]:
    formula_rows: list[dict] = []
    detail_rows: list[dict] = []

    for group, task in _paper_tasks():
        path = _formula_path(task)
        if not path.exists():
            continue
        df = sort_formulas_by_accuracy(pd.read_csv(path)).reset_index(drop=True)
        if top_n is not None and top_n > 0:
            df = df.head(top_n).copy()

        features = _feature_order(task)
        task_label = _short_task_label(task)
        display_label = f"{group}: {task_label}"

        for i, row in df.iterrows():
            rank = i + 1
            rank_weight = 1.0 / rank
            eq = str(row.get("equation", ""))
            used_idx = sorted(formula_var_indices(eq))
            valid_used = [idx for idx in used_idx if idx < len(features)]

            formula_rows.append({
                "group": group,
                "task": task,
                "task_label": task_label,
                "display_label": display_label,
                "formula_rank": rank,
                "rank_weight": rank_weight,
                "equation": eq,
                "complexity": _metric(row, "complexity"),
                "test_R2": _metric(row, "test_R2"),
                "test_MSE": _metric(row, "test_MSE"),
                "test_MAE": _metric(row, "test_MAE"),
                "loss": _metric(row, "loss"),
                "n_variables_used": len(valid_used),
                "variables_used": ";".join(features[idx] for idx in valid_used),
            })

            for idx in valid_used:
                feature = features[idx]
                occurrences = max(1, _var_occurrences(eq, idx))
                detail_rows.append({
                    "group": group,
                    "task": task,
                    "task_label": task_label,
                    "display_label": display_label,
                    "formula_rank": rank,
                    "rank_weight": rank_weight,
                    "equation": eq,
                    "x_index": idx,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "feature_zh": PHYSICAL_LABEL_ZH.get(feature, feature),
                    "occurrence_count": occurrences,
                    "presence_score": rank_weight,
                    "occurrence_score": rank_weight * occurrences,
                    "test_R2": _metric(row, "test_R2"),
                    "test_MSE": _metric(row, "test_MSE"),
                    "complexity": _metric(row, "complexity"),
                })

    formulas = pd.DataFrame(formula_rows)
    details = pd.DataFrame(detail_rows)
    if details.empty:
        return {
            "formulas": formulas,
            "details": details,
            "edges": pd.DataFrame(),
            "nodes": pd.DataFrame(),
            "matrix_occurrence_score": pd.DataFrame(),
            "matrix_occurrence_share": pd.DataFrame(),
        }

    agg = (
        details.groupby(["group", "task", "task_label", "display_label", "feature", "feature_label", "feature_zh"], as_index=False)
        .agg(
            formula_count_using_feature=("formula_rank", "nunique"),
            total_occurrence_count=("occurrence_count", "sum"),
            weighted_presence_score=("presence_score", "sum"),
            weighted_occurrence_score=("occurrence_score", "sum"),
            mean_rank_when_used=("formula_rank", "mean"),
            best_rank_when_used=("formula_rank", "min"),
        )
    )

    totals = agg.groupby("task", as_index=False).agg(
        task_weighted_presence_total=("weighted_presence_score", "sum"),
        task_weighted_occurrence_total=("weighted_occurrence_score", "sum"),
    )
    agg = agg.merge(totals, on="task", how="left")
    agg["weighted_presence_share"] = agg["weighted_presence_score"] / agg["task_weighted_presence_total"].replace(0, np.nan)
    agg["weighted_occurrence_share"] = agg["weighted_occurrence_score"] / agg["task_weighted_occurrence_total"].replace(0, np.nan)

    # Origin Sankey convention: Source, Target, Value.
    edges = agg.copy()
    edges.insert(0, "Source", edges["feature_label"])
    edges.insert(1, "Target", edges["display_label"])
    edges.insert(2, "Value", edges["weighted_occurrence_share"])
    edges["Value_raw_weighted_occurrence_score"] = edges["weighted_occurrence_score"]
    edges["Value_weighted_presence_share"] = edges["weighted_presence_share"]
    edges["edge_note"] = "Value = weighted_occurrence_share; rank_weight = 1/rank"

    node_rows: list[dict] = []
    for feature, sub in agg.groupby("feature", sort=False):
        node_rows.append({
            "node": FEATURE_LABELS.get(feature, feature),
            "node_type": "feature",
            "raw_feature": feature,
            "group": ";".join(sorted(sub["group"].unique())),
        })
    for _, r in formulas[["group", "task", "display_label"]].drop_duplicates().iterrows():
        node_rows.append({
            "node": r["display_label"],
            "node_type": "task",
            "raw_feature": "",
            "group": r["group"],
        })
    nodes = pd.DataFrame(node_rows)

    matrix_occurrence_score = (
        agg.pivot_table(
            index="display_label",
            columns="feature_label",
            values="weighted_occurrence_score",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    matrix_occurrence_share = (
        agg.pivot_table(
            index="display_label",
            columns="feature_label",
            values="weighted_occurrence_share",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    return {
        "formulas": formulas,
        "details": details,
        "edges": edges,
        "nodes": nodes,
        "matrix_occurrence_score": matrix_occurrence_score,
        "matrix_occurrence_share": matrix_occurrence_share,
    }


def run(out_dir: Path | None = None, *, top_n: int | None = None) -> list[Path]:
    out_dir = out_dir or TMP_OUT
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_top{top_n}" if top_n is not None and top_n > 0 else "_all"
    tables = build_formula_variable_tables(top_n=top_n)
    outputs: list[Path] = []
    for name, df in tables.items():
        path = data_dir / f"origin_formula_variable_sankey_{name}{suffix}.csv"
        save_csv(df, path)
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ranked-formula variable Sankey data for Origin.")
    parser.add_argument("--top-n", type=int, default=0, help="Use only top N formulas per task; 0 = all formulas")
    args = parser.parse_args()
    top_n = args.top_n if args.top_n and args.top_n > 0 else None
    for path in run(top_n=top_n):
        print(path)


if __name__ == "__main__":
    main()
