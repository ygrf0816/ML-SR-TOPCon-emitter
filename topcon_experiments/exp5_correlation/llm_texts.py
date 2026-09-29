"""Collect LLM analysis texts from exp4 outputs for semantic similarity panels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from topcon_experiments.common.variable_labels import label_for
from topcon_experiments.config import IV_TARGET_LABELS, OUTPUT_ROOT
from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy

EXP4_SYM = OUTPUT_ROOT / "exp4_symbolic"


@dataclass
class TextItem:
    panel: str
    label: str
    source: str
    analysis_type: str
    equation: str
    text: str


def _short_label(source: str) -> str:
    if source in ("doping", "defect"):
        return "doping curve" if source == "doping" else "defect curve"
    if source.startswith("athena_to_iv_"):
        key = "iv_" + source.replace("athena_to_iv_", "")
        return IV_TARGET_LABELS.get(key, label_for(key))
    if source.startswith("full_to_iv_"):
        key = "iv_" + source.replace("full_to_iv_", "")
        return f"full->{IV_TARGET_LABELS.get(key, label_for(key))}"
    if source.startswith("athena_to_"):
        raw = source.replace("athena_to_", "").replace("_log", "")
        return label_for(raw).split("[")[0].strip()
    return source


def _best_equation_for_source(source: str) -> str | None:
    if source in ("doping", "defect"):
        from topcon_experiments.config import SR_DOPING_CURVE_FORMULA_PATH, SR_DEFECT_CURVE_FORMULA_PATH

        path = SR_DOPING_CURVE_FORMULA_PATH if source == "doping" else SR_DEFECT_CURVE_FORMULA_PATH
        if not path.exists():
            path = EXP4_SYM / f"sr_formulas_{source}.csv"
    else:
        path = EXP4_SYM / f"sr_tabular_{source}.csv"
    if not path.exists():
        return None
    df = sort_formulas_by_accuracy(pd.read_csv(path))
    if df.empty:
        return None
    return str(df.iloc[0].get("equation", ""))


def _pick_llm_row(llm_df: pd.DataFrame, equation: str | None) -> pd.Series | None:
    if llm_df.empty:
        return None
    if equation:
        hits = llm_df[llm_df["equation"].astype(str) == equation]
        if not hits.empty:
            return hits.iloc[0]
    return llm_df.iloc[0]


def _load_panel_best(
    panel: str,
    sources: list[str],
    analysis_type: str,
    llm_prefix: str,
) -> list[TextItem]:
    items: list[TextItem] = []
    for source in sources:
        path = EXP4_SYM / f"{llm_prefix}_{source}.csv"
        if not path.exists():
            continue
        llm_df = pd.read_csv(path)
        if "analysis_type" in llm_df.columns:
            filtered = llm_df[llm_df["analysis_type"].astype(str) == analysis_type]
            if not filtered.empty:
                llm_df = filtered
        eq = _best_equation_for_source(source)
        row = _pick_llm_row(llm_df, eq)
        if row is None:
            continue
        text = str(row.get("response", "")).strip()
        if not text:
            continue
        label = _short_label(source)
        items.append(
            TextItem(
                panel=panel,
                label=label,
                source=source,
                analysis_type=analysis_type,
                equation=str(row.get("equation", "")),
                text=text,
            )
        )
    return items


def _load_panel_topk(
    panel: str,
    source: str,
    analysis_type: str,
    llm_prefix: str,
    k: int,
) -> list[TextItem]:
    path = EXP4_SYM / f"{llm_prefix}_{source}.csv"
    sr_path = EXP4_SYM / f"sr_tabular_{source}.csv"
    if not path.exists() or not sr_path.exists():
        return []
    llm_df = pd.read_csv(path)
    if "analysis_type" in llm_df.columns:
        filtered = llm_df[llm_df["analysis_type"].astype(str) == analysis_type]
        if not filtered.empty:
            llm_df = filtered
    sr_df = sort_formulas_by_accuracy(pd.read_csv(sr_path)).head(k)
    items: list[TextItem] = []
    for rank, (_, srow) in enumerate(sr_df.iterrows(), start=1):
        eq = str(srow.get("equation", ""))
        row = _pick_llm_row(llm_df, eq)
        if row is None:
            continue
        text = str(row.get("response", "")).strip()
        if not text:
            continue
        items.append(
            TextItem(
                panel=panel,
                label=f"#{rank}",
                source=source,
                analysis_type=analysis_type,
                equation=eq,
                text=text,
            )
        )
    return items


def collect_all_text_items(topk_within: int = 5) -> list[TextItem]:
    """Build semantic-similarity panels from exp4 LLM CSV outputs."""
    iv_sources = ["athena_to_iv_Eff", "athena_to_iv_Voc", "athena_to_iv_Jsc", "athena_to_iv_FF"]
    doping_sources = [
        "athena_to_doping_N_peak_log",
        "athena_to_doping_x_peak",
        "athena_to_doping_junction_depth",
        "athena_to_doping_FWHM",
        "athena_to_doping_gradient_max",
        "athena_to_doping_dose_log",
        "athena_to_doping_R_sheet",
    ]
    defect_sources = [
        "athena_to_defect_vac_N_peak_log",
        "athena_to_defect_vac_gradient_max",
        "athena_to_defect_vac_dose_log",
    ]
    full_iv_sources = [
        "full_to_iv_Eff",
        "full_to_iv_Voc",
        "full_to_iv_Jsc",
        "full_to_iv_FF",
    ]

    items: list[TextItem] = []
    panels = [
        ("iv_athena_physics", iv_sources, "physics", "llm_physics_tabular"),
        ("iv_athena_blind_math", iv_sources, "blind_math", "llm_blind_math_tabular"),
        ("doping_descriptor_physics", doping_sources, "physics", "llm_physics_tabular"),
        ("defect_descriptor_physics", defect_sources, "physics", "llm_physics_tabular"),
        ("full_iv_physics", full_iv_sources, "physics", "llm_physics_tabular"),
        ("curve_physics", ["doping", "defect"], "physics", "llm_physics"),
    ]
    for panel, sources, atype, prefix in panels:
        items.extend(_load_panel_best(panel, sources, atype, prefix))

    items.extend(
        _load_panel_topk(
            "iv_athena_physics_topk",
            "athena_to_iv_Eff",
            "physics",
            "llm_physics_tabular",
            topk_within,
        )
    )
    return items


def items_to_dataframe(items: list[TextItem]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "panel": it.panel,
                "label": it.label,
                "source": it.source,
                "analysis_type": it.analysis_type,
                "equation": it.equation,
                "text_len": len(it.text),
                "text": it.text,
            }
            for it in items
        ]
    )
