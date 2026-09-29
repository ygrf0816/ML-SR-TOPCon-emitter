"""Rich paper figures for SR + LLM interpretation (beyond bar/scatter only)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch

from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import ATHENA_FEATURES, MODEL2_FEATURES, OUTPUT_ROOT
from topcon_experiments.exp4_symbolic.sr_equation_utils import eval_sympy_expr, sort_formulas_by_accuracy
from topcon_experiments.exp4_symbolic.sr_metrics import regression_metrics
from topcon_experiments.exp4_symbolic.sr_split_utils import split_train_test
from topcon_experiments.tmp_fig5.export_fig5_table import build_paper_table
from topcon_experiments.tmp_fig5.sr_llm_comparison import PAPER_TASK_GROUPS, _short_task_label
from topcon_experiments.tmp_fig5.utils import (
    ATHENA_DISPLAY,
    EXP4_OUT,
    TMP_OUT,
    extract_json_block,
    formula_complexity,
    formula_var_indices,
    llm_text_row,
    load_xy_for_task,
)

SR_PLOT_OUT = OUTPUT_ROOT / "exp4_symbolic" / "sr_plots"

GROUP_COLORS = {
    "IV (Athena)": "#440154",
    "IV (full model)": "#31688e",
    "Doping": "#35b779",
    "Defect": "#fde725",
    "Curves": "#fdae61",
}

OP_SPECS = [
    ("log", r"\blog\b"),
    ("exp", r"\bexp\b"),
    ("sqrt", r"\bsqrt\b"),
    ("abs", r"\babs\b"),
    ("square", r"\bsquare\b"),
    ("inv", r"\binv\b"),
    ("rational", r"/"),
]
OP_COLORS = {
    "log": "#440154",
    "exp": "#31688e",
    "sqrt": "#35b779",
    "abs": "#fde725",
    "square": "#fdae61",
    "inv": "#f46d43",
    "rational": "#d53e4f",
}


def _task_group(task: str) -> str:
    for group, tasks in PAPER_TASK_GROUPS.items():
        if task in tasks:
            return group
    return "Other"


def _llm_paths(task: str) -> tuple[Path | None, Path | None]:
    if task in ("doping_curve_tail_full", "defect_curve"):
        src = "doping" if "doping" in task else "defect"
        return EXP4_OUT / f"llm_blind_math_{src}.csv", EXP4_OUT / f"llm_physics_{src}.csv"
    blind = EXP4_OUT / f"llm_blind_math_tabular_{task}.csv"
    phys = EXP4_OUT / f"llm_physics_tabular_{task}.csv"
    return (blind if blind.exists() else None, phys if phys.exists() else None)


def _best_equation(task: str) -> str:
    if task in ("doping_curve_tail_full", "defect_curve"):
        from topcon_experiments.config import SR_DOPING_CURVE_FORMULA_PATH, SR_DEFECT_CURVE_FORMULA_PATH

        path = SR_DOPING_CURVE_FORMULA_PATH if "doping" in task else SR_DEFECT_CURVE_FORMULA_PATH
        row = sort_formulas_by_accuracy(pd.read_csv(path)).iloc[0]
    else:
        row = sort_formulas_by_accuracy(pd.read_csv(EXP4_OUT / f"sr_tabular_{task}.csv")).iloc[0]
    return str(row.get("equation", ""))


def parse_plausibility_score(text: str) -> tuple[float, str]:
    blob = (text or "").lower()
    parsed = extract_json_block(text)
    rating = ""
    if parsed:
        pa = parsed.get("plausibility_assessment")
        if isinstance(pa, dict):
            rating = str(pa.get("rating", "")).lower()
            blob += " " + str(pa.get("explanation", "")).lower()
            blob += " " + str(pa.get("comment", "")).lower()
        elif isinstance(pa, str):
            rating = pa.lower()
            blob += " " + rating

    if rating in ("high", "good"):
        return 3.0, "high"
    if rating in ("medium", "moderate", "moderately"):
        return 2.0, "medium"
    if rating in ("low", "poor", "bad"):
        return 1.0, "low"

    low_kw = (
        "not physically plausible", "physically implausible", "low plausib",
        "no predictive or physical value", "empirical artifact", "overfit",
    )
    high_kw = ("highly plausible", "high plausib", "physically plausible", "good surrogate")
    med_kw = ("moderately plausible", "moderate plausib", "plausible as a", "reasonable as a surrogate")

    if any(k in blob for k in low_kw):
        if any(k in blob for k in med_kw):
            return 2.0, "medium"
        return 1.0, "low"
    if any(k in blob for k in high_kw):
        return 3.0, "high"
    if any(k in blob for k in med_kw):
        return 2.0, "medium"
    return 2.0, "medium"


def parse_blind_math(text: str) -> dict[str, float | str]:
    parsed = extract_json_block(text)
    if not parsed:
        return {"match_score": float("nan"), "confidence": float("nan"), "relationship_type": ""}

    m = parsed.get("matches_provided_names")
    ms = str(m).lower()
    if m is True or ms in ("true", "yes"):
        match = 1.0
    elif m is False or ms in ("false", "no"):
        match = 0.0
    else:
        match = 0.5

    conf = parsed.get("confidence")
    if isinstance(conf, (int, float)):
        conf_v = float(conf)
    elif isinstance(conf, str):
        cl = conf.lower()
        if "low" in cl:
            conf_v = 0.35
        elif "high" in cl:
            conf_v = 0.85
        elif "moderate" in cl:
            conf_v = 0.6
        else:
            try:
                conf_v = float(conf)
            except ValueError:
                conf_v = float("nan")
    else:
        conf_v = float("nan")

    return {
        "match_score": match,
        "confidence": conf_v,
        "relationship_type": str(parsed.get("relationship_type", "")),
    }


def count_formula_ops(expr: str) -> dict[str, int]:
    s = str(expr)
    counts: dict[str, int] = {}
    for name, pat in OP_SPECS:
        counts[name] = len(re.findall(pat, s, flags=re.IGNORECASE))
    return counts


def _feature_labels(task: str) -> list[str]:
    if task.startswith("full_to_") or task.endswith("_curve") or task == "defect_curve":
        return [f"x{i}" for i in range(len(MODEL2_FEATURES))]
    return [ATHENA_DISPLAY[c] for c in ATHENA_FEATURES]


def build_insights_table() -> pd.DataFrame:
    paper = build_paper_table()
    if paper.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, row in paper.iterrows():
        task = str(row["task"])
        eq = str(row["sr_equation_pysr"])
        group = _task_group(task)
        n_feat_total = len(_feature_labels(task))
        used_idx = formula_var_indices(eq)
        blind_p, phys_p = _llm_paths(task)

        blind_text = ""
        phys_text = ""
        if blind_p is not None:
            blind_text = llm_text_row(pd.read_csv(blind_p), eq, "blind_math")
        if phys_p is not None:
            phys_text = llm_text_row(pd.read_csv(phys_p), eq, "physics")

        blind = parse_blind_math(blind_text)
        plaus_score, plaus_label = parse_plausibility_score(phys_text)
        ops = count_formula_ops(eq)

        rows.append({
            "group": group,
            "task": task,
            "label": _short_task_label(task),
            "sr_test_R2": float(row["sr_test_R2"]),
            "llm_test_R2": float(row.get("llm_test_R2", float("nan"))),
            "sr_complexity": int(row["sr_complexity"]),
            "llm_complexity": int(row.get("llm_complexity", float("nan"))),
            "complexity_reduction_pct": float(row.get("complexity_reduction_pct", float("nan"))),
            "n_features_used": len(used_idx),
            "n_features_total": n_feat_total,
            "feature_sparsity": len(used_idx) / n_feat_total if n_feat_total else float("nan"),
            "plausibility_score": plaus_score,
            "plausibility_label": plaus_label,
            "blind_match_score": blind["match_score"],
            "blind_confidence": blind["confidence"],
            "blind_relationship_type": blind["relationship_type"],
            **{f"op_{k}": v for k, v in ops.items()},
        })

    return pd.DataFrame(rows)


def plot_complexity_dumbbell(df: pd.DataFrame, out_png: Path) -> None:
    sub = df.dropna(subset=["llm_complexity"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values(["group", "sr_complexity"], ascending=[True, False])
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, max(5, 0.38 * len(sub))))
    y = np.arange(len(sub))

    for i, (_, r) in enumerate(sub.iterrows()):
        c = GROUP_COLORS.get(r["group"], "#888888")
        ax.plot([r["sr_complexity"], r["llm_complexity"]], [i, i], color=c, lw=2.2, alpha=0.75, zorder=1)
        ax.scatter(r["sr_complexity"], i, s=70, color="#440154", edgecolors="white", lw=0.6, zorder=2)
        ax.scatter(r["llm_complexity"], i, s=70, color="#21908C", marker="s", edgecolors="white", lw=0.6, zorder=2)
        ax.text(
            max(r["sr_complexity"], r["llm_complexity"]) + 0.4,
            i,
            f"R²={r['sr_test_R2']:.2f}",
            va="center",
            fontsize=7,
            color="0.35",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['label']} ({r['group']})" for _, r in sub.iterrows()], fontsize=8)
    ax.set_xlabel("SymPy complexity (count_ops)")
    ax.set_title("LLM simplification: complexity dumbbell (PySR → LLM)")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    for g in sub["group"].unique():
        ax.scatter([], [], color=GROUP_COLORS.get(g, "#888"), label=g, s=40)
    ax.scatter([], [], color="#440154", label="PySR", s=50)
    ax.scatter([], [], color="#21908C", marker="s", label="LLM simplified", s=50)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_plausibility_landscape(df: pd.DataFrame, out_png: Path) -> None:
    sub = df[np.isfinite(df["sr_test_R2"]) & np.isfinite(df["plausibility_score"])].copy()
    if sub.empty:
        return
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.axvspan(0.0, 0.7, ymin=0.0, ymax=0.45, color="#fee0d2", alpha=0.35, zorder=0)
    ax.axvspan(0.7, 1.02, ymin=0.0, ymax=0.45, color="#fcbba1", alpha=0.35, zorder=0)
    ax.axvspan(0.0, 0.7, ymin=0.45, ymax=1.0, color="#deebf7", alpha=0.35, zorder=0)
    ax.axvspan(0.7, 1.02, ymin=0.45, ymax=1.0, color="#c7e9c0", alpha=0.4, zorder=0)

    ax.text(0.18, 2.75, "low accuracy\nweak fit", fontsize=8, color="0.4", ha="center")
    ax.text(0.86, 2.75, "accurate &\nphysically plausible", fontsize=8, color="0.35", ha="center", fontweight="bold")
    ax.text(0.18, 1.25, "weak &\nnot physical", fontsize=8, color="0.4", ha="center")
    ax.text(0.86, 1.25, "accurate but\nempirical", fontsize=8, color="0.4", ha="center")

    for _, r in sub.iterrows():
        c = GROUP_COLORS.get(r["group"], "#888888")
        size = 80 + 25 * float(r["sr_complexity"]) if np.isfinite(r["sr_complexity"]) else 100
        ax.scatter(
            r["sr_test_R2"], r["plausibility_score"],
            s=size, c=c, alpha=0.82, edgecolors="white", linewidths=0.7, zorder=3,
        )
        if r["sr_test_R2"] > 0.85 or r["plausibility_score"] < 1.3:
            ax.annotate(
                r["label"], (r["sr_test_R2"], r["plausibility_score"]),
                textcoords="offset points", xytext=(4, 4), fontsize=7, color="0.25",
            )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.7, 3.3)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["low", "medium", "high"])
    ax.set_xlabel("test R² (PySR best formula)")
    ax.set_ylabel("LLM physics plausibility")
    ax.set_title("Prediction accuracy vs physical plausibility")
    ax.grid(alpha=0.2)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GROUP_COLORS[g], markersize=8, label=g)
        for g in GROUP_COLORS if g in sub["group"].values
    ]
    ax.legend(handles=handles, title="task group", loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_formula_fingerprint_rose(df: pd.DataFrame, out_png: Path) -> None:
    op_cols = [c for c in df.columns if c.startswith("op_")]
    if not op_cols:
        return
    tasks = df["label"].tolist()
    n = len(tasks)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    apply_plot_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows), subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(axes).ravel()

    labels = [c.replace("op_", "") for c in op_cols]
    theta = np.linspace(0, 2 * np.pi, len(op_cols), endpoint=False)
    width = 2 * np.pi / len(op_cols) * 0.85

    for ax_i, (_, row) in enumerate(df.iterrows()):
        if ax_i >= len(axes):
            break
        ax = axes[ax_i]
        vals = np.array([row[c] for c in op_cols], dtype=float)
        bottom = 0.0
        for j, (lab, val) in enumerate(zip(labels, vals)):
            if val <= 0:
                continue
            ax.bar(
                theta[j], val, width=width, bottom=bottom,
                color=OP_COLORS.get(lab, "#999999"), alpha=0.9, edgecolor="white", linewidth=0.4,
            )
            bottom += val
        ax.set_xticks(theta)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels([])
        ax.set_title(f"{row['label']}\n({row['group']})", fontsize=8, pad=12)
        ax.grid(alpha=0.2)

    for j in range(ax_i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Formula structure fingerprint (rose charts: operator counts)", fontsize=11, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_iv_feature_bipartite(out_png: Path) -> None:
    tasks = PAPER_TASK_GROUPS["IV (Athena)"]
    feat_labels = [ATHENA_DISPLAY[c] for c in ATHENA_FEATURES]
    edges: list[tuple[int, int, str]] = []
    for ti, task in enumerate(tasks):
        eq = _best_equation(task)
        for fi in formula_var_indices(eq):
            if fi < len(feat_labels):
                edges.append((fi, ti, GROUP_COLORS["IV (Athena)"]))

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    n_f, n_t = len(feat_labels), len(tasks)
    y_f = np.linspace(0, 1, n_f)
    y_t = np.linspace(0, 1, n_t)
    x_f, x_t = 0.08, 0.92

    for i, lab in enumerate(feat_labels):
        ax.scatter(x_f, y_f[i], s=120, color="#440154", zorder=3)
        ax.text(x_f - 0.03, y_f[i], lab, ha="right", va="center", fontsize=8)

    short = [_short_task_label(t) for t in tasks]
    for j, lab in enumerate(short):
        ax.scatter(x_t, y_t[j], s=140, color="#21908C", marker="s", zorder=3)
        ax.text(x_t + 0.03, y_t[j], lab, ha="left", va="center", fontsize=9, fontweight="bold")

    for fi, ti, color in edges:
        con = ConnectionPatch(
            xyA=(x_f, y_f[fi]), xyB=(x_t, y_t[ti]),
            coordsA="data", coordsB="data", axesA=ax, axesB=ax,
            color=color, alpha=0.35, linewidth=1.8,
            connectionstyle="arc3,rad=0.12",
        )
        ax.add_artist(con)

    ax.text(x_f, 1.05, "Athena process\nparameters", ha="center", fontsize=9, transform=ax.transAxes)
    ax.text(x_t, 1.05, "IV targets", ha="center", fontsize=9, transform=ax.transAxes)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.08, 1.08)
    ax.axis("off")
    ax.set_title("Bipartite graph: which parameters enter each IV formula?")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_blind_math_radar(df: pd.DataFrame, out_png: Path) -> None:
    sub = df[df["group"] == "IV (Athena)"].copy()
    if sub.empty:
        return
    categories = ["test R²", "name match", "blind conf.", "sparsity", "1−complexity/20"]
    n_cat = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
    angles += angles[:1]

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(sub)))

    for i, (_, r) in enumerate(sub.iterrows()):
        vals = [
            float(r["sr_test_R2"]),
            float(r["blind_match_score"]) if np.isfinite(r["blind_match_score"]) else 0.5,
            float(r["blind_confidence"]) if np.isfinite(r["blind_confidence"]) else 0.5,
            float(r["feature_sparsity"]) if np.isfinite(r["feature_sparsity"]) else 0.5,
            max(0.0, 1.0 - float(r["sr_complexity"]) / 20.0),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[i], lw=1.8, label=r["label"])
        ax.fill(angles, vals, color=colors[i], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Blind-math & structure profile (IV Athena)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False, fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _split_equation_terms(expr: str) -> list[tuple[str, str]]:
    """Heuristic term blocks for infographic annotation."""
    s = str(expr)
    terms: list[tuple[str, str]] = []
    patterns = [
        (r"exp\([^)]+\)", "exponential decay / activation"),
        (r"log\([^)]+\)", "logarithmic dose or thickness"),
        (r"abs\([^)]+\)", "nonlinear saturation (abs)"),
        (r"/\s*\([^)]+\)", "rational trade-off"),
        (r"square\([^)]+\)", "quadratic penalty"),
    ]
    for pat, meaning in patterns:
        for m in re.finditer(pat, s, flags=re.IGNORECASE):
            terms.append((m.group(0)[:42], meaning))
    if not terms:
        terms.append((s[:50] + ("…" if len(s) > 50 else ""), "empirical combination"))
    return terms[:5]


def plot_case_study_voc(out_png: Path) -> None:
    task = "athena_to_iv_Voc"
    paper = build_paper_table()
    row = paper[paper["task"] == task]
    if row.empty:
        return
    row = row.iloc[0]
    eq = str(row["sr_equation_pysr"])
    simp_eq = str(row.get("llm_expr_eval", "")) or eq

    blind_p, phys_p = _llm_paths(task)
    blind = parse_blind_math(llm_text_row(pd.read_csv(blind_p), eq, "blind_math") if blind_p else "")
    plaus_score, plaus_label = parse_plausibility_score(
        llm_text_row(pd.read_csv(phys_p), eq, "physics") if phys_p else ""
    )

    X, y, _ = load_xy_for_task(task)
    train_idx, test_idx = split_train_test(len(X))
    X_test = X.iloc[test_idx].values.astype(float)
    y_test = y[test_idx]
    pred = eval_sympy_expr(str(row["sr_expr_eval"]), X_test)

    apply_plot_style()
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 1], wspace=0.25, hspace=0.3)

    ax_flow = fig.add_subplot(gs[0, 0])
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 6)
    ax_flow.axis("off")
    ax_flow.set_title("A  LLM interpretation workflow (Voc case)", loc="left", fontsize=10, fontweight="bold")

    boxes = [
        (0.2, 3.8, 2.2, 1.2, "PySR\nbest formula", "#440154"),
        (2.8, 3.8, 2.2, 1.2, "LLM\nsimplify", "#21908C"),
        (5.4, 3.8, 2.0, 1.2, "Blind\nmath", "#31688e"),
        (7.7, 3.8, 2.0, 1.2, "Physics\nreview", "#35b779"),
        (0.2, 1.0, 9.5, 1.8, "", "#f7f7f7"),
    ]
    for x, y0, w, h, text, color in boxes[:4]:
        patch = FancyBboxPatch(
            (x, y0), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=color, edgecolor="white", alpha=0.9, linewidth=1.2,
        )
        ax_flow.add_patch(patch)
        ax_flow.text(x + w / 2, y0 + h / 2, text, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    for i in range(3):
        x0 = 2.4 + i * 2.6
        arr = FancyArrowPatch((x0, 4.4), (x0 + 0.35, 4.4), arrowstyle="-|>", mutation_scale=12, color="0.45", lw=1.5)
        ax_flow.add_patch(arr)

    info = FancyBboxPatch(
        (0.2, 1.0), 9.5, 1.8, boxstyle="round,pad=0.1", facecolor="#f7f7f7", edgecolor="0.8",
    )
    ax_flow.add_patch(info)
    ax_flow.text(
        0.45, 2.35,
        f"R²={row['sr_test_R2']:.3f}  |  complexity {row['sr_complexity']}→{row.get('llm_complexity', '—')}  |  "
        f"blind match={blind.get('match_score', float('nan')):.0%}  |  plausibility={plaus_label}",
        fontsize=8.5, color="0.25",
    )
    ax_flow.text(0.45, 1.55, f"PySR: {eq[:95]}…", fontsize=7.5, family="monospace", color="0.2")
    ax_flow.text(0.45, 1.15, f"LLM:  {simp_eq[:95]}…", fontsize=7.5, family="monospace", color="#21908C")

    ax_terms = fig.add_subplot(gs[1, 0])
    ax_terms.set_xlim(0, 10)
    ax_terms.set_ylim(0, len(_split_equation_terms(eq)) + 1)
    ax_terms.axis("off")
    ax_terms.set_title("B  Term-level structure (heuristic)", loc="left", fontsize=10, fontweight="bold")
    for i, (frag, meaning) in enumerate(_split_equation_terms(eq)):
        yp = len(_split_equation_terms(eq)) - i
        ax_terms.add_patch(FancyBboxPatch(
            (0.3, yp - 0.35), 3.6, 0.7, boxstyle="round,pad=0.05", facecolor="#440154", alpha=0.85,
        ))
        ax_terms.text(2.1, yp, frag, ha="center", va="center", color="white", fontsize=7, family="monospace")
        arr = FancyArrowPatch((4.0, yp), (5.0, yp), arrowstyle="-|>", mutation_scale=10, color="0.5")
        ax_terms.add_patch(arr)
        ax_terms.text(5.2, yp, meaning, va="center", fontsize=8, color="0.3")

    ax_sc = fig.add_subplot(gs[:, 1])
    m = regression_metrics(y_test, pred)
    ax_sc.scatter(y_test, pred, s=18, alpha=0.55, c="#31688e", edgecolors="none")
    lo, hi = float(np.min(y_test)), float(np.max(y_test))
    pad = 0.02 * (hi - lo + 1e-9)
    ax_sc.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5)
    ax_sc.set_xlabel("measured Voc")
    ax_sc.set_ylabel("SR predicted Voc")
    ax_sc.set_title(f"C  Test scatter (R²={m['R2']:.3f})", loc="left", fontsize=10, fontweight="bold")
    ax_sc.grid(alpha=0.25)

    fig.suptitle("Case study: Voc — accurate fit with limited physical interpretability", fontsize=12, y=0.98)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_sankey_style(df: pd.DataFrame, out_png: Path) -> None:
    """Alluvial-style summary: task groups → interpretation layers."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    groups = [g for g in PAPER_TASK_GROUPS if g in df["group"].values]
    n_g = len(groups)
    y_g = np.linspace(8.5, 1.5, n_g)
    layer_x = [1.0, 3.5, 6.0, 8.5]
    layer_names = ["SR task", "LLM simplify", "Blind math", "Physics"]

    for x, name in zip(layer_x, layer_names):
        ax.text(x, 9.5, name, ha="center", fontsize=9, fontweight="bold", color="0.3")

    for i, g in enumerate(groups):
        sub = df[df["group"] == g]
        mean_r2 = sub["sr_test_R2"].mean()
        mean_drop = sub["complexity_reduction_pct"].mean()
        mean_plaus = sub["plausibility_score"].mean()
        color = GROUP_COLORS.get(g, "#888")

        ys = y_g[i]
        for j, x in enumerate(layer_x):
            w = 0.55 + 0.15 * j
            ax.add_patch(FancyBboxPatch(
                (x - w / 2, ys - 0.35), w, 0.7, boxstyle="round,pad=0.05",
                facecolor=color, alpha=0.35 + 0.15 * j, edgecolor=color, linewidth=0.8,
            ))
            if j == 0:
                ax.text(x, ys, g.replace(" ", "\n"), ha="center", va="center", fontsize=7.5, color="0.2")
            elif j == 1:
                ax.text(x, ys, f"−{mean_drop:.0f}% C", ha="center", va="center", fontsize=7, color="0.25")
            elif j == 2:
                ax.text(x, ys, f"R²={mean_r2:.2f}", ha="center", va="center", fontsize=7, color="0.25")
            else:
                ax.text(x, ys, f"plaus.\n{mean_plaus:.1f}/3", ha="center", va="center", fontsize=7, color="0.25")

        for j in range(len(layer_x) - 1):
            ax.annotate(
                "", xy=(layer_x[j + 1] - 0.35, ys), xytext=(layer_x[j] + 0.35, ys),
                arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.2),
            )

    ax.set_title("Interpretation pipeline summary by task group (alluvial-style)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_composite_insights(plot_dir: Path, out_png: Path) -> Path | None:
    panels = [
        plot_dir / "sr_insight_dumbbell_complexity.png",
        plot_dir / "sr_insight_plausibility_landscape.png",
        plot_dir / "sr_insight_formula_rose.png",
        plot_dir / "sr_insight_iv_bipartite.png",
        plot_dir / "sr_insight_blind_math_radar.png",
        plot_dir / "sr_insight_case_voc.png",
    ]
    existing = [p for p in panels if p.exists()]
    if len(existing) < 3:
        return None

    apply_plot_style()
    n = len(existing)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    titles = [
        "Complexity dumbbell", "Plausibility landscape", "Formula fingerprints",
        "IV bipartite graph", "Blind-math radar", "Voc case study",
    ]

    for i, path in enumerate(existing):
        axes[i].imshow(plt.imread(path))
        axes[i].axis("off")
        axes[i].set_title(titles[i] if i < len(titles) else path.stem, fontsize=10, loc="left", fontweight="bold")

    for j in range(len(existing), len(axes)):
        axes[j].axis("off")

    fig.suptitle("SR + LLM interpretation — supplementary figure panels", fontsize=13, y=1.01)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def run(out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or TMP_OUT
    plot_dir = out_dir / "plots"
    data_dir = out_dir / "data"
    sr_plot_dir = SR_PLOT_OUT
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    sr_plot_dir.mkdir(parents=True, exist_ok=True)

    df = build_insights_table()
    if df.empty:
        return []

    csv_path = data_dir / "sr_llm_insights_table.csv"
    save_csv(df, csv_path)
    save_csv(df, sr_plot_dir / "sr_llm_insights_table.csv")

    jobs: list[tuple[str, callable]] = [
        ("sr_insight_dumbbell_complexity.png", lambda p: plot_complexity_dumbbell(df, p)),
        ("sr_insight_plausibility_landscape.png", lambda p: plot_plausibility_landscape(df, p)),
        ("sr_insight_formula_rose.png", lambda p: plot_formula_fingerprint_rose(df, p)),
        ("sr_insight_iv_bipartite.png", lambda p: plot_iv_feature_bipartite(p)),
        ("sr_insight_blind_math_radar.png", lambda p: plot_blind_math_radar(df, p)),
        ("sr_insight_case_voc.png", lambda p: plot_case_study_voc(p)),
        ("sr_insight_pipeline_flow.png", lambda p: plot_pipeline_sankey_style(df, p)),
    ]

    written: list[Path] = [csv_path]
    for name, fn in jobs:
        for d in (plot_dir, sr_plot_dir):
            out = d / name
            fn(out)
            written.append(out)

    comp = plot_composite_insights(plot_dir, plot_dir / "sr_insight_composite.png")
    if comp:
        written.append(comp)
        comp2 = sr_plot_dir / "sr_insight_composite.png"
        shutil.copy2(comp, comp2)
        written.append(comp2)

    return written


if __name__ == "__main__":
    for p in run():
        print(p)
