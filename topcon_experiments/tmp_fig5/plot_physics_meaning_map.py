"""Objective physical-meaning maps for SR formulas.

The figure intentionally avoids LLM plausibility scores. It combines:
1) variables actually used by the best SR formulas;
2) a hand-defined variable -> physics-mechanism map;
3) keyword evidence extracted from the LLM physics explanations.
"""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import ConnectionPatch, FancyBboxPatch

from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import ATHENA_FEATURES, MODEL2_FEATURES, OUTPUT_ROOT
from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy
from topcon_experiments.exp4_symbolic.sr_variable_map import PHYSICAL_LABEL_ZH
from topcon_experiments.tmp_fig5.sr_llm_comparison import PAPER_TASK_GROUPS, _short_task_label
from topcon_experiments.tmp_fig5.utils import EXP4_OUT, TMP_OUT, formula_var_indices, llm_text_row

SR_PLOT_OUT = OUTPUT_ROOT / "exp4_symbolic" / "sr_plots"

GROUP_COLORS = {
    "IV (Athena)": "#440154",
    "IV (full model)": "#31688e",
    "Doping": "#35b779",
    "Defect": "#fdae61",
    "Curves": "#d53e4f",
}

MECHANISMS = {
    "boron_supply": {
        "label": "Boron supply / dose",
        "keywords": ["boron", "dopant", "doping", "dose", "source", "concentration"],
    },
    "thermal_diffusion": {
        "label": "Thermal diffusion",
        "keywords": ["diffusion", "thermal", "temperature", "time", "drive-in", "erfc", "gaussian"],
    },
    "oxidation_passivation": {
        "label": "Oxidation / passivation",
        "keywords": ["oxidation", "oxide", "passivation", "tunnel", "siox", "ambient", "o2", "oxygen"],
    },
    "profile_shape": {
        "label": "Profile shape",
        "keywords": ["profile", "junction", "depth", "peak", "tail", "fwhm", "gradient", "shallow"],
    },
    "defect_recombination": {
        "label": "Defect / recombination",
        "keywords": ["defect", "vacancy", "recombination", "auger", "srh", "trap", "loss"],
    },
    "transport_resistance": {
        "label": "Transport / resistance",
        "keywords": ["resistance", "sheet", "series", "contact", "transport", "fill factor", "ff"],
    },
    "device_output": {
        "label": "Device output",
        "keywords": ["voc", "jsc", "pce", "efficiency", "current", "voltage", "iv"],
    },
}

FEATURE_MECHANISMS = {
    "athena_thick": ["boron_supply", "oxidation_passivation"],
    "athena_c_boron": ["boron_supply"],
    "athena_temp1": ["thermal_diffusion"],
    "athena_time1": ["thermal_diffusion"],
    "athena_temp2": ["thermal_diffusion", "oxidation_passivation"],
    "athena_time2": ["thermal_diffusion", "oxidation_passivation"],
    "athena_F_N2": ["oxidation_passivation"],
    "athena_F_O2": ["oxidation_passivation"],
    "depth_um": ["profile_shape"],
    "doping_N_peak": ["boron_supply", "defect_recombination"],
    "doping_x_peak": ["profile_shape"],
    "doping_junction_depth": ["profile_shape", "defect_recombination"],
    "doping_FWHM": ["profile_shape", "thermal_diffusion"],
    "doping_gradient_max": ["profile_shape"],
    "doping_dose": ["boron_supply", "transport_resistance"],
    "doping_R_sheet": ["transport_resistance", "profile_shape"],
    "defect_vac_N_peak": ["defect_recombination"],
    "defect_vac_gradient_max": ["defect_recombination", "profile_shape"],
    "defect_vac_dose": ["defect_recombination"],
}

FEATURE_LABELS = {
    "athena_thick": "BSG thick",
    "athena_c_boron": "B source",
    "athena_temp1": "T1",
    "athena_time1": "t1",
    "athena_temp2": "T2",
    "athena_time2": "t2",
    "athena_F_N2": "N2 flow",
    "athena_F_O2": "O2 flow",
    "depth_um": "depth",
    "doping_N_peak": "Na peak",
    "doping_x_peak": "Na peak depth",
    "doping_junction_depth": "junction depth",
    "doping_FWHM": "FWHM",
    "doping_gradient_max": "Na gradient",
    "doping_dose": "Na dose",
    "doping_R_sheet": "R_sheet",
    "defect_vac_N_peak": "vac peak",
    "defect_vac_gradient_max": "vac gradient",
    "defect_vac_dose": "vac dose",
}


def _paper_tasks() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for group, tasks in PAPER_TASK_GROUPS.items():
        out.extend((group, task) for task in tasks)
    return out


def _feature_order(task: str) -> list[str]:
    if task == "doping_curve_tail_full":
        return list(MODEL2_FEATURES) + ["depth_um"]
    if task == "defect_curve":
        return list(ATHENA_FEATURES) + ["depth_um"]
    if task.startswith("full_to_iv_"):
        return list(MODEL2_FEATURES)
    return list(ATHENA_FEATURES)


def _formula_path(task: str) -> Path:
    if task == "doping_curve_tail_full":
        from topcon_experiments.config import SR_DOPING_CURVE_FORMULA_PATH

        return SR_DOPING_CURVE_FORMULA_PATH
    if task == "defect_curve":
        from topcon_experiments.config import SR_DEFECT_CURVE_FORMULA_PATH

        return SR_DEFECT_CURVE_FORMULA_PATH
    return EXP4_OUT / f"sr_tabular_{task}.csv"


def _best_row(task: str) -> pd.Series | None:
    path = _formula_path(task)
    if not path.exists():
        return None
    return sort_formulas_by_accuracy(pd.read_csv(path)).iloc[0]


def _llm_physics_text(task: str, equation: str) -> str:
    if task in ("doping_curve_tail_full", "defect_curve"):
        src = "doping" if task == "doping_curve_tail_full" else "defect"
        path = EXP4_OUT / f"llm_physics_{src}.csv"
    else:
        path = EXP4_OUT / f"llm_physics_tabular_{task}.csv"
    if not path.exists():
        return ""
    return llm_text_row(pd.read_csv(path), equation, "physics")


def _keyword_hits(text: str) -> dict[str, tuple[int, str]]:
    lower = str(text).lower()
    out: dict[str, tuple[int, str]] = {}
    for mech, spec in MECHANISMS.items():
        hits: list[str] = []
        for kw in spec["keywords"]:
            kw_l = kw.lower()
            if re.fullmatch(r"[a-z0-9]+", kw_l):
                pattern = rf"\b{re.escape(kw_l)}\b"
            else:
                pattern = re.escape(kw_l)
            count = len(re.findall(pattern, lower))
            if count:
                hits.extend([kw] * count)
        out[mech] = (len(hits), ";".join(sorted(set(hits))))
    return out


def _variable_occurrence_counts(expr: str, used_idx: list[int]) -> dict[int, int]:
    """Structural variable weight: occurrences of xi in the discovered formula."""
    counts: dict[int, int] = {}
    for idx in used_idx:
        n = len(re.findall(rf"\bx{idx}\b", str(expr)))
        counts[idx] = max(1, n)
    return counts


def build_physics_meaning_tables() -> dict[str, pd.DataFrame]:
    task_rows: list[dict] = []
    feature_rows: list[dict] = []
    keyword_rows: list[dict] = []
    formula_edges: list[dict] = []
    mechanism_edges: list[dict] = []

    for group, task in _paper_tasks():
        row = _best_row(task)
        if row is None:
            continue
        eq = str(row.get("equation", ""))
        used_idx = sorted(formula_var_indices(eq))
        features = _feature_order(task)
        used_pairs = [(i, features[i]) for i in used_idx if i < len(features)]
        used_features = [feat for _, feat in used_pairs]
        var_counts = _variable_occurrence_counts(eq, [i for i, _ in used_pairs])
        total_var_count = sum(var_counts.values()) or 1
        llm_text = _llm_physics_text(task, eq)
        kh = _keyword_hits(llm_text)
        total_keyword_hits = sum(v[0] for v in kh.values()) or 1

        task_rows.append({
            "group": group,
            "task": task,
            "label": _short_task_label(task),
            "display_label": f"{group}: {_short_task_label(task)}",
            "equation": eq,
            "test_R2": float(row.get("test_R2", np.nan)),
            "test_MSE": float(row.get("test_MSE", np.nan)),
            "complexity": int(row.get("complexity", np.nan)),
            "n_features_used": len(used_features),
            "features_used": ";".join(used_features),
            "uses_depth": int("depth_um" in used_features),
            "llm_physics_text_len": len(llm_text),
        })

        for feat_idx, feat in used_pairs:
            var_weight = var_counts.get(feat_idx, 1)
            var_share = var_weight / total_var_count
            feature_rows.append({
                "group": group,
                "task": task,
                "task_label": _short_task_label(task),
                "display_label": f"{group}: {_short_task_label(task)}",
                "feature": feat,
                "feature_label": FEATURE_LABELS.get(feat, PHYSICAL_LABEL_ZH.get(feat, feat)),
                "feature_zh": PHYSICAL_LABEL_ZH.get(feat, feat),
                "variable_occurrences": var_weight,
                "variable_share_in_formula": var_share,
            })
            formula_edges.append({
                "source": f"feature::{feat}",
                "target": f"task::{task}",
                "edge_type": "formula_variable_to_task",
                "task": task,
                "feature": feat,
                "mechanism": "",
                "weight": var_weight,
                "weight_share": var_share,
                "evidence": f"x{feat_idx} appears {var_weight} time(s) in best SR formula",
            })
            for mech in FEATURE_MECHANISMS.get(feat, []):
                mechanism_edges.append({
                    "source": f"feature::{feat}",
                    "target": f"mechanism::{mech}",
                    "edge_type": "feature_physics_prior",
                    "task": task,
                    "feature": feat,
                    "mechanism": mech,
                    "weight": 1,
                    "weight_share": np.nan,
                    "evidence": "manual feature-to-mechanism mapping",
                })

        for mech, (hits, terms) in kh.items():
            keyword_rows.append({
                "group": group,
                "task": task,
                "task_label": _short_task_label(task),
                "display_label": f"{group}: {_short_task_label(task)}",
                "mechanism": mech,
                "mechanism_label": MECHANISMS[mech]["label"],
                "keyword_hits": hits,
                "matched_terms": terms,
            })
            if hits > 0:
                hit_share = hits / total_keyword_hits
                mechanism_edges.append({
                    "source": f"task::{task}",
                    "target": f"mechanism::{mech}",
                    "edge_type": "llm_keyword_evidence",
                    "task": task,
                    "feature": "",
                    "mechanism": mech,
                    "weight": hits,
                    "weight_share": hit_share,
                    "evidence": terms,
                })

    tasks_df = pd.DataFrame(task_rows)
    features_df = pd.DataFrame(feature_rows)
    keywords_df = pd.DataFrame(keyword_rows)
    edges_df = pd.DataFrame(formula_edges + mechanism_edges)

    feature_matrix = (
        features_df.assign(value=1)
        .pivot_table(index="display_label", columns="feature_label", values="value", aggfunc="max", fill_value=0)
        .reset_index()
        if not features_df.empty
        else pd.DataFrame()
    )
    mechanism_matrix = (
        keywords_df.pivot_table(
            index="display_label",
            columns="mechanism_label",
            values="keyword_hits",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        if not keywords_df.empty
        else pd.DataFrame()
    )

    node_rows: list[dict] = []
    for _, r in tasks_df.iterrows():
        node_rows.append({
            "node_id": f"task::{r['task']}",
            "label": r["label"],
            "node_type": "task",
            "group": r["group"],
        })
    for feat, sub in features_df.groupby("feature", sort=False):
        node_rows.append({
            "node_id": f"feature::{feat}",
            "label": FEATURE_LABELS.get(feat, feat),
            "node_type": "feature",
            "group": ";".join(sorted(sub["group"].unique())),
        })
    for mech, spec in MECHANISMS.items():
        node_rows.append({
            "node_id": f"mechanism::{mech}",
            "label": spec["label"],
            "node_type": "mechanism",
            "group": "physics",
        })

    return {
        "tasks": tasks_df,
        "features": features_df,
        "keywords": keywords_df,
        "edges": edges_df,
        "nodes": pd.DataFrame(node_rows),
        "feature_matrix": feature_matrix,
        "mechanism_matrix": mechanism_matrix,
    }


def _bezier(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, color: str, lw: float, alpha: float) -> None:
    con = ConnectionPatch(
        xyA=(x1, y1),
        xyB=(x2, y2),
        coordsA="data",
        coordsB="data",
        axesA=ax,
        axesB=ax,
        connectionstyle="arc3,rad=0.15",
        color=color,
        linewidth=lw,
        alpha=alpha,
        zorder=1,
    )
    ax.add_artist(con)


def plot_network_panel(tables: dict[str, pd.DataFrame], ax: plt.Axes) -> None:
    tasks = tables["tasks"].copy()
    features = tables["features"].copy()
    keywords = tables["keywords"].copy()
    edges = tables["edges"].copy()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("A  Formula variables -> SR tasks -> physics concepts", loc="left", fontweight="bold")

    if tasks.empty:
        return

    feature_order = (
        features["feature"].value_counts().sort_values(ascending=False).index.tolist()
        if not features.empty
        else []
    )
    feature_y = {feat: y for feat, y in zip(feature_order, np.linspace(0.88, 0.12, max(len(feature_order), 1)))}

    task_y: dict[str, float] = {}
    ys = np.linspace(0.88, 0.12, len(tasks))
    for i, (_, r) in enumerate(tasks.iterrows()):
        task_y[str(r["task"])] = ys[i]

    mechanism_order = list(MECHANISMS.keys())
    mech_y = {m: y for m, y in zip(mechanism_order, np.linspace(0.88, 0.12, len(mechanism_order)))}

    ax.text(0.10, 0.97, "Formula variable", ha="center", fontsize=8, color="0.35")
    ax.text(0.47, 0.97, "SR task", ha="center", fontsize=8, color="0.35")
    ax.text(0.88, 0.97, "Physics concept", ha="center", fontsize=8, color="0.35")

    for feat, y in feature_y.items():
        ax.add_patch(FancyBboxPatch((0.005, y - 0.017), 0.19, 0.034, boxstyle="round,pad=0.006", fc="#e6f2ff", ec="#4f81bd", alpha=0.95))
        ax.text(0.10, y, FEATURE_LABELS.get(feat, feat), ha="center", va="center", fontsize=6.7, color="0.15")

    for _, r in tasks.iterrows():
        y = task_y[str(r["task"])]
        c = GROUP_COLORS.get(str(r["group"]), "#888888")
        ax.add_patch(FancyBboxPatch((0.39, y - 0.017), 0.16, 0.034, boxstyle="round,pad=0.006", fc=c, ec="white", alpha=0.85))
        ax.text(0.47, y, str(r["label"]), ha="center", va="center", color="white", fontsize=6.7)

    for mech, y in mech_y.items():
        ax.add_patch(FancyBboxPatch((0.76, y - 0.02), 0.23, 0.04, boxstyle="round,pad=0.006", fc="#f6f1e6", ec="#b58900", alpha=0.95))
        ax.text(0.875, y, MECHANISMS[mech]["label"], ha="center", va="center", fontsize=6.6, color="0.15")

    # Formula variable -> SR task: weighted by structural share of xi occurrences in the formula.
    f_edges = edges[edges["edge_type"] == "formula_variable_to_task"].copy()
    for _, r in f_edges.iterrows():
        task = str(r["task"])
        feat = str(r["feature"])
        if task not in task_y or feat not in feature_y:
            continue
        group = tasks.loc[tasks["task"] == task, "group"]
        color = GROUP_COLORS.get(str(group.iloc[0]) if not group.empty else "", "#888888")
        share = float(r.get("weight_share", 0.0))
        lw = 0.6 + 7.5 * share
        alpha = 0.16 + 0.34 * min(share / 0.5, 1.0)
        _bezier(ax, 0.195, feature_y[feat], 0.39, task_y[task], color, lw, alpha)

    # SR task -> physics concept: weighted by keyword-hit share in the LLM physics explanation.
    km_edges = edges[edges["edge_type"] == "llm_keyword_evidence"].copy()
    for _, r in km_edges.iterrows():
        task = str(r["task"])
        mech = str(r["mechanism"])
        if task not in task_y or mech not in mech_y:
            continue
        group = tasks.loc[tasks["task"] == task, "group"]
        color = GROUP_COLORS.get(str(group.iloc[0]) if not group.empty else "", "#999999")
        share = float(r.get("weight_share", 0.0))
        lw = 0.35 + 8.5 * share
        alpha = 0.08 + 0.42 * min(share / 0.35, 1.0)
        _bezier(ax, 0.55, task_y[task], 0.76, mech_y[mech], color, lw, alpha)

    ax.text(
        0.02,
        0.02,
        "Line width = within-task share (variables: xi occurrence share; concepts: keyword-hit share).",
        fontsize=6.8,
        color="0.35",
        ha="left",
        va="bottom",
    )


def plot_mechanism_heatmap(tables: dict[str, pd.DataFrame], ax: plt.Axes) -> None:
    mat = tables["mechanism_matrix"].copy()
    ax.set_title("B  LLM text keyword evidence by mechanism", loc="left", fontweight="bold")
    if mat.empty:
        ax.axis("off")
        return
    mat = mat.set_index("display_label")
    ordered_cols = [MECHANISMS[m]["label"] for m in MECHANISMS if MECHANISMS[m]["label"] in mat.columns]
    mat = mat.reindex(columns=ordered_cols)
    sns.heatmap(
        np.log1p(mat.astype(float)),
        ax=ax,
        cmap="YlGnBu",
        linewidths=0.3,
        cbar_kws={"label": "log(1 + keyword hits)", "shrink": 0.8},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)


def plot_depth_diagnostic(tables: dict[str, pd.DataFrame], ax: plt.Axes) -> None:
    tasks = tables["tasks"].copy()
    curves = tasks[tasks["group"] == "Curves"].copy()
    ax.set_title("C  Curve-formula depth diagnostic", loc="left", fontweight="bold")
    if curves.empty:
        ax.axis("off")
        return

    cols = ["uses_depth", "n_features_used", "test_R2", "test_MSE"]
    y = np.arange(len(curves))
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, len(curves) - 0.5)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(curves["label"], fontsize=8)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(["uses\ndepth", "# used\nvars", "test\nR²", "test\nMSE"], fontsize=8)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, (_, r) in enumerate(curves.iterrows()):
        vals = [
            "yes" if int(r["uses_depth"]) else "no",
            str(int(r["n_features_used"])),
            f"{float(r['test_R2']):.2f}",
            f"{float(r['test_MSE']):.3g}",
        ]
        for j, text in enumerate(vals):
            if j == 0:
                fc = "#35b779" if text == "yes" else "#fdae61"
                tc = "white"
            else:
                fc = "#f7f7f7"
                tc = "0.2"
            ax.add_patch(FancyBboxPatch((j + 0.08, i - 0.28), 0.84, 0.56, boxstyle="round,pad=0.02", fc=fc, ec="0.85"))
            ax.text(j + 0.5, i, text, ha="center", va="center", color=tc, fontsize=8, fontweight="bold" if j == 0 else None)

    ax.text(
        0.0,
        len(curves) - 0.05,
        "A curve formula without depth can mainly capture sample-level offset, not depth-varying shape.",
        fontsize=7.2,
        color="0.35",
        ha="left",
        va="top",
    )


def plot_physics_meaning_figure(tables: dict[str, pd.DataFrame], out_png: Path) -> None:
    apply_plot_style()
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.55], hspace=0.35, wspace=0.24)
    ax_net = fig.add_subplot(gs[:, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_depth = fig.add_subplot(gs[1, 1])
    plot_network_panel(tables, ax_net)
    plot_mechanism_heatmap(tables, ax_heat)
    plot_depth_diagnostic(tables, ax_depth)
    fig.suptitle(
        "Physical interpretation map of symbolic formulas: formula variables + LLM keyword evidence",
        fontsize=13,
        y=0.98,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or TMP_OUT
    data_dir = out_dir / "data"
    plot_dir = out_dir / "plots"
    sr_plot_dir = SR_PLOT_OUT
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    sr_plot_dir.mkdir(parents=True, exist_ok=True)

    tables = build_physics_meaning_tables()
    outputs: list[Path] = []
    for name, df in tables.items():
        csv_path = data_dir / f"sr_physics_meaning_{name}.csv"
        save_csv(df, csv_path)
        outputs.append(csv_path)
        save_csv(df, sr_plot_dir / csv_path.name)

    png = plot_dir / "sr_physics_meaning_map.png"
    plot_physics_meaning_figure(tables, png)
    outputs.append(png)
    png2 = sr_plot_dir / png.name
    shutil.copy2(png, png2)
    outputs.append(png2)
    return outputs


if __name__ == "__main__":
    for path in run():
        print(path)
