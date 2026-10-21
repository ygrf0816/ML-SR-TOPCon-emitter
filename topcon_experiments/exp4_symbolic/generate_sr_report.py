"""Generate per-task markdown reports from symbolic regression + LLM outputs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.variable_labels import label_for
from topcon_experiments.config import (
    IV_TARGETS,
    OUTPUT_ROOT,
    SR_DOPING_CURVE_FORMULA_PATH,
    SR_DEFECT_CURVE_FORMULA_PATH,
    SR_TOP_FORMULAS_IN_REPORT,
)
from topcon_experiments.exp4_symbolic.sr_metrics import sort_formulas_by_metrics as sort_formulas_by_accuracy
from topcon_experiments.exp4_symbolic.sr_variable_map import (
    feature_order_for_task,
    log_features_for_task,
    substitute_equation,
    variable_legend_table,
)

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
REPORT_DIR = EXP4_OUT / "sr_reports"
INDEX_PATH = EXP4_OUT / "symbolic_regression_report.md"
LEGACY_COMBINED = EXP4_OUT / "symbolic_regression_report_combined.md"

TASK_THEORY = {
    "doping_curve_tail_full": (
        "掺杂曲线尾部（0.25~2 μm，剔除 BSG 尖峰区）；"
        "Erfc/Gaussian 扩散剖面在体区的衰减；"
        "输入为 Athena + 掺杂/缺陷描述符 + depth。"
    ),
    "doping_curve": "Erfc/Gaussian 扩散剖面；峰值浓度与源剂量、√(Dt) 相关；结深 ~ √(Dt)。",
    "defect_vac_curve": "热过程引入的空位缺陷剖面；常呈深度衰减；与淬火/氧化相关。",
    "tabular_athena_to_descriptor": "工艺参数 (T, t, 剂量, 气氛) 映射到曲线描述符 (峰值、FWHM、结深)。",
    "tabular_athena_to_iv": "IV 指标遵循二极管方程：Voc ~ (kT/q)ln(Jsc/J0+1)；Eff ~ Voc·Jsc·FF。",
    "tabular_full_to_iv": "在 model2 全特征 (Athena + 掺杂/缺陷描述符) 下拟合 IV；可捕捉工艺-剖面-电性能的耦合。",
    "tabular_iv_trio_ff": "用实测 Voc/Jsc/Eff 拟合 FF；与恒等式 FF=100·Eff/(Voc·Jsc) 对比。",
}

# Per-task theory overrides (slug -> text)
SLUG_THEORY: dict[str, str] = {
    "athena_to_doping_R_sheet": (
        "前部发射极方阻 $R_{\\mathrm{sheet}}$（Ω/□）由硼扩散后的有效掺杂剂量与结深共同决定，"
        "经验上 $R_s \\propto 1/(N_{\\mathrm{dose}} \\cdot \\mu_p)$；"
        "BSG 厚度与硼源浓度决定可用剂量，扩散温度/时间控制激活与再分布。"
        "本任务从 8 个 Athena 工艺参数直接回归方阻（目标为原尺度，非 log）。"
    ),
}


def _extract_json_block(text: str) -> dict | None:
    if not isinstance(text, str) or not text.strip():
        return None
    for pattern in (
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
    ):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


def _esc_cell(text: str, max_len: int = 100) -> str:
    s = str(text).replace("\n", " ").replace("|", "\\|").strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _fmt_num(val) -> str:
    if pd.isna(val):
        return "N/A"
    try:
        f = float(val)
        if abs(f) >= 1e4 or (abs(f) < 1e-3 and f != 0):
            return f"{f:.6g}"
        return f"{f:.6f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(val)


def _preprocess_block(sr_df: pd.DataFrame) -> str:
    if sr_df.empty:
        return ""
    note = str(sr_df["preprocess_note"].iloc[0]).strip() if "preprocess_note" in sr_df.columns else ""
    key = str(sr_df["preprocess_key"].iloc[0]) if "preprocess_key" in sr_df.columns else ""
    lines = ["### 预处理说明（与 exp1 前向模型对齐）", ""]
    lines.append(note if note else "_未记录预处理说明_")
    if "n_train" in sr_df.columns and "n_test" in sr_df.columns:
        lines += [
            "",
            f"- 切分：`TEST_SIZE=0.2`, `RANDOM_STATE=42`；"
            f"n_train={int(sr_df['n_train'].iloc[0])}, n_test={int(sr_df['n_test'].iloc[0])}",
        ]
    if key:
        lines.append(f"- 预处理键：`{key}`")
    lines.append("")
    return "\n".join(lines)


def _rank_formula_table(sr_df: pd.DataFrame, n: int) -> tuple[str, list[str]]:
    """Return markdown table and list of full equations for appendix."""
    if sr_df.empty:
        return "_无公式数据_\n", []

    cols = [
        c
        for c in ["complexity", "loss", "train_MSE", "test_MSE", "train_MAE", "test_MAE", "train_R2", "test_R2", "dataset_R2", "score", "equation"]
        if c in sr_df.columns
    ]
    show = sort_formulas_by_accuracy(sr_df).head(n).reset_index(drop=True)
    header = "| rank | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    full_eqs: list[str] = []
    for i, row in show.iterrows():
        full_eq = str(row.get("equation", ""))
        full_eqs.append(full_eq)
        vals = []
        for c in cols:
            if c == "equation":
                vals.append(_esc_cell(full_eq, 90))
            else:
                vals.append(_esc_cell(_fmt_num(row[c])))
        rows.append("| " + " | ".join([str(i + 1)] + vals) + " |")
    return "\n".join([header, sep] + rows) + "\n", full_eqs


def _llm_rows_by_equation(llm_df: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if llm_df is None or llm_df.empty:
        return out
    for _, row in llm_df.iterrows():
        eq = str(row.get("equation", ""))
        out[eq] = row.to_dict()
    return out


def _llm_lookup(llm_df: pd.DataFrame, equation: str, rank: int) -> dict:
    """Match LLM row by exact equation; fall back to row order (rank) if SR CSV was refreshed."""
    if llm_df is None or llm_df.empty:
        return {}
    by_eq = _llm_rows_by_equation(llm_df)
    if equation in by_eq:
        return by_eq[equation]
    if 1 <= rank <= len(llm_df):
        return llm_df.iloc[rank - 1].to_dict()
    return {}


def _blind_math_table(sr_df: pd.DataFrame, blind_df: pd.DataFrame, n: int) -> str:
    if sr_df.empty:
        return "_无数据_\n"
    blind_map = _llm_rows_by_equation(blind_df)
    header = "| rank | equation | relationship_type | confidence | notes |"
    sep = "| --- | --- | --- | --- | --- |"
    rows = []
    ranked = sort_formulas_by_accuracy(sr_df).head(n)
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        eq = str(row.get("equation", ""))
        parsed = _extract_json_block(str(_llm_lookup(blind_df, eq, rank).get("response", "")))
        if parsed:
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        _esc_cell(eq, 70),
                        _esc_cell(parsed.get("relationship_type", "")),
                        _esc_cell(parsed.get("confidence", "")),
                        _esc_cell(parsed.get("notes", ""), 80),
                    ]
                )
                + " |"
            )
        else:
            rows.append(f"| {rank} | {_esc_cell(eq, 70)} | — | — | _无 LLM 结果_ |")
    return "\n".join([header, sep] + rows) + "\n"


def _simplify_table(sr_df: pd.DataFrame, simp_df: pd.DataFrame, n: int) -> str:
    if sr_df.empty:
        return "_无数据_\n"
    simp_map = _llm_rows_by_equation(simp_df)
    header = "| rank | equation | simplified_formula | notes |"
    sep = "| --- | --- | --- | --- |"
    rows = []
    ranked = sort_formulas_by_accuracy(sr_df).head(n)
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        eq = str(row.get("equation", ""))
        parsed = _extract_json_block(str(_llm_lookup(simp_df, eq, rank).get("response", "")))
        if parsed:
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        _esc_cell(eq, 70),
                        _esc_cell(parsed.get("simplified_formula", ""), 80),
                        _esc_cell(parsed.get("simplification_notes", ""), 80),
                    ]
                )
                + " |"
            )
        else:
            rows.append(f"| {rank} | {_esc_cell(eq, 70)} | — | _无 LLM 结果_ |")
    return "\n".join([header, sep] + rows) + "\n"


def _physical_formula_table(
    sr_df: pd.DataFrame,
    simp_df: pd.DataFrame,
    task_type: str,
    n: int,
) -> str:
    if sr_df.empty:
        return "_无数据_\n"
    log_feats = log_features_for_task(task_type)
    feat_order = feature_order_for_task(task_type)
    simp_map = _llm_rows_by_equation(simp_df)
    header = "| rank | 物理符号公式 | 简化物理公式 |"
    sep = "| --- | --- | --- |"
    rows = []
    ranked = sort_formulas_by_accuracy(sr_df).head(n)
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        eq = str(row.get("equation", ""))
        phys = substitute_equation(eq, feat_order, log_features=log_feats)
        parsed = _extract_json_block(str(_llm_lookup(simp_df, eq, rank).get("response", "")))
        simp_phys = "—"
        if parsed and parsed.get("simplified_formula"):
            simp_phys = substitute_equation(str(parsed["simplified_formula"]), feat_order, log_features=log_feats)
        rows.append(
            "| "
            + " | ".join([str(rank), _esc_cell(phys, 90), _esc_cell(simp_phys, 90)])
            + " |"
        )
    return "\n".join([header, sep] + rows) + "\n"


def _physics_section(sr_df: pd.DataFrame, phys_df: pd.DataFrame, n: int) -> str:
    if sr_df.empty:
        return "_无数据_\n"
    phys_map = _llm_rows_by_equation(phys_df)
    lines: list[str] = []
    ranked = sort_formulas_by_accuracy(sr_df).head(n)
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        eq = str(row.get("equation", ""))
        resp = str(_llm_lookup(phys_df, eq, rank).get("response", ""))
        parsed = _extract_json_block(resp)
        lines.append(f"#### 公式 {rank}")
        lines.append("")
        lines.append(f"```text\n{eq}\n```")
        lines.append("")
        if parsed:
            if parsed.get("overall_physics"):
                lines.append(f"**物理含义**：{parsed['overall_physics']}")
                lines.append("")
            interp = parsed.get("term_interpretations")
            if isinstance(interp, dict):
                lines.append("**各项解释**：")
                for k, v in interp.items():
                    lines.append(f"- {k}: {v}")
                lines.append("")
            if parsed.get("plausibility_assessment"):
                lines.append(f"**与现有理论关系**：{parsed['plausibility_assessment']}")
                lines.append("")
        else:
            snippet = _esc_cell(resp, 600)
            lines.append(f"**分析摘要**：{snippet}")
            lines.append("")
    return "\n".join(lines)


def _evaluation_appendix(slug: str, best_row: pd.Series) -> str:
    """Optional §5: scatter plot, AutoGluon baseline, dataset correlation."""
    lines: list[str] = []
    scatter_metrics = EXP4_OUT / "sr_plots" / f"regression_scatter_sr_{slug}_metrics.csv"
    scatter_png = EXP4_OUT / "sr_plots" / f"regression_scatter_sr_{slug}.png"
    if scatter_metrics.exists():
        m = pd.read_csv(scatter_metrics).iloc[0]
        lines += [
            "### 5.1 测试集回归散点图（最优公式）",
            "",
            f"- 图像：`sr_plots/regression_scatter_sr_{slug}.png`",
            f"- test R²（重算）: `{_fmt_num(m.get('R2'))}`",
            f"- test RMSE: `{_fmt_num(m.get('RMSE'))}` Ω/□" if "R_sheet" in slug else f"- test RMSE: `{_fmt_num(m.get('RMSE'))}`",
            f"- test MAE: `{_fmt_num(m.get('MAE'))}`",
            f"- n_test: `{int(m.get('n', 0))}`",
            "",
        ]
    elif scatter_png.exists():
        lines += [
            "### 5.1 测试集回归散点图（最优公式）",
            "",
            f"- 图像：`sr_plots/regression_scatter_sr_{slug}.png`",
            "",
        ]

    target = slug.replace("athena_to_", "").replace("full_to_", "").replace("_log", "")
    if slug.startswith("athena_to_doping_"):
        ag_target = "doping_" + slug.replace("athena_to_doping_", "").replace("_log", "")
    elif slug.startswith("athena_to_defect_vac_"):
        ag_target = "defect_vac_" + slug.replace("athena_to_defect_vac_", "").replace("_log", "")
    elif slug.startswith("athena_to_iv_"):
        ag_target = "iv_" + slug.replace("athena_to_iv_", "")
    else:
        ag_target = target

    metrics_path = OUTPUT_ROOT / "exp1_forward" / "metrics_model1.csv"
    if slug.startswith("athena_to_iv_") or slug.startswith("full_to_iv_"):
        metrics_path = OUTPUT_ROOT / "exp1_forward" / "metrics_model2.csv"
    if metrics_path.exists() and ag_target:
        ag = pd.read_csv(metrics_path)
        hit = ag[(ag["target"] == ag_target) & (ag["split"] == "test")]
        if not hit.empty:
            row = hit.iloc[0]
            lines += [
                "### 5.2 AutoGluon 基线（同一切分）",
                "",
                f"- 模型：exp1 `{'model1' if 'model1' in str(metrics_path) else 'model2'}` / `autogluon_{ag_target}`",
                f"- test R²: `{_fmt_num(row.get('R2'))}`",
                f"- test RMSE (primary): `{_fmt_num(row.get('RMSE_primary'))}`",
                f"- test MAE (primary): `{_fmt_num(row.get('MAE_primary'))}`",
                f"- nRMSE%: `{_fmt_num(row.get('nRMSE_pct'))}`",
                "",
            ]
            sr_r2 = float(best_row.get("test_R2", float("nan")))
            ag_r2 = float(row.get("R2", float("nan")))
            if pd.notna(sr_r2) and pd.notna(ag_r2):
                lines.append(
                    f"- **对比**：PySR 最优 test R² = {sr_r2:.4f}；AutoGluon test R² = {ag_r2:.4f}。"
                )
                lines.append("")

    if slug == "athena_to_doping_R_sheet":
        pearson = OUTPUT_ROOT / "exp5_correlation" / "data" / "pearson_descriptors_iv.csv"
        if pearson.exists():
            corr = pd.read_csv(pearson, index_col=0)
            rs_label = "Front emitter sheet resistance (ohm/sq)"
            if rs_label in corr.index:
                lines += [
                    "### 5.3 数据集 Pearson 相关（方阻 vs IV）",
                    "",
                    "| 变量 | Pearson r |",
                    "| --- | --- |",
                ]
                for iv in ("Voc", "Jsc", "FF", "PCE"):
                    if iv in corr.columns:
                        lines.append(f"| {iv} | {_fmt_num(corr.loc[rs_label, iv])} |")
                lines.append("")
                lines.append(
                    "方阻与 PCE/Jsc 呈负相关（|r|≈0.55~0.57），与“方阻升高→发射极电阻增大→电性能下降”一致。"
                )
                lines.append("")

    if not lines:
        return ""
    return "\n".join(["---", "", "## 5. 独立评估与基线对比", ""] + lines)


def build_task_report(
    title: str,
    task_type: str,
    slug: str,
    sr_path: Path,
    blind_path: Path | None,
    simp_path: Path | None,
    phys_path: Path | None,
) -> str:
    blind_df = pd.read_csv(blind_path) if blind_path and blind_path.exists() else pd.DataFrame()
    simp_df = pd.read_csv(simp_path) if simp_path and simp_path.exists() else pd.DataFrame()
    phys_df = pd.read_csv(phys_path) if phys_path and phys_path.exists() else pd.DataFrame()

    lines = [
        f"# {title}",
        "",
        f"- **任务类型**：`{task_type}`",
        f"- **报告文件**：`sr_reports/{slug}.md`",
        f"- **理论参考**：{SLUG_THEORY.get(slug, TASK_THEORY.get(task_type, '见 TCAD / 半导体工艺文献'))}",
        "",
        "---",
        "",
        "## 1. PySR 符号公式（x0, x1, …）",
        "",
        "以下公式使用 PySR 内部编号变量，含义见下表。",
        "",
        "### 1.1 变量编号对照",
        "",
        variable_legend_table(task_type, log_features=log_features_for_task(task_type)),
        "",
    ]

    if not sr_path.exists():
        lines += ["### 1.2 公式排名（按 test_MSE 升序，MAE 次之，R² 参考）", "", "_符号回归结果文件不存在_\n"]
        return "\n".join(lines)

    sr_df = pd.read_csv(sr_path)
    sr_df = sort_formulas_by_accuracy(sr_df)

    lines += [_preprocess_block(sr_df), "### 1.2 公式排名（按 test_MSE 升序，MAE 次之，R² 参考）", ""]
    rank_tbl, full_eqs = _rank_formula_table(sr_df, SR_TOP_FORMULAS_IN_REPORT)
    lines.append(rank_tbl)

    if full_eqs:
        lines += ["<details>", "<summary>完整公式（展开）</summary>", ""]
        for i, eq in enumerate(full_eqs[:SR_TOP_FORMULAS_IN_REPORT], start=1):
            lines += [f"**#{i}**", "", f"```text\n{eq}\n```", ""]
        lines += ["</details>", ""]

    if len(sr_df):
        best = sr_df.iloc[0]
        lines += [
            "### 1.3 最优公式（符号编号）",
            "",
            f"- test_MSE: `{_fmt_num(best.get('test_MSE'))}`",
            f"- test_MAE: `{_fmt_num(best.get('test_MAE'))}`",
            f"- test_R²: `{_fmt_num(best.get('test_R2'))}`",
            f"- loss (PySR MSE): `{_fmt_num(best.get('loss'))}`",
            f"- complexity: `{_fmt_num(best.get('complexity'))}`",
            "",
            f"```text\n{best.get('equation', '')}\n```",
            "",
            "---",
            "",
            "## 2. Blind-math 与公式简化",
            "",
            "### 2.1 Blind-math 分析",
            "",
            _blind_math_table(sr_df, blind_df, SR_TOP_FORMULAS_IN_REPORT),
            "",
            "### 2.2 公式简化",
            "",
            _simplify_table(sr_df, simp_df, SR_TOP_FORMULAS_IN_REPORT),
            "",
            "---",
            "",
            "## 3. 物理变量形式",
            "",
            "将 x0… 替换为 `parameter_name.md` 中的物理符号（见 §1.1）。",
            "",
            _physical_formula_table(sr_df, simp_df, task_type, SR_TOP_FORMULAS_IN_REPORT),
            "",
            "---",
            "",
            "## 4. 物理含义与理论评述",
            "",
            _physics_section(sr_df, phys_df, SR_TOP_FORMULAS_IN_REPORT),
        ]
        appendix = _evaluation_appendix(slug, best)
        if appendix:
            lines.append(appendix)

    return "\n".join(lines)


def _task_jobs() -> list[tuple[str, str, str, Path, Path | None, Path | None, Path | None]]:
    jobs: list[tuple[str, str, str, Path, Path | None, Path | None, Path | None]] = [
        (
            "掺杂曲线 SR（尾部窗口 tail_full）",
            "doping_curve_tail_full",
            "doping_curve",
            SR_DOPING_CURVE_FORMULA_PATH,
            EXP4_OUT / "llm_blind_math_doping.csv",
            EXP4_OUT / "llm_simplified_doping.csv",
            EXP4_OUT / "llm_physics_doping.csv",
        ),
        (
            "缺陷曲线 SR",
            "defect_vac_curve",
            "defect_curve",
            SR_DEFECT_CURVE_FORMULA_PATH,
            EXP4_OUT / "llm_blind_math_defect.csv",
            EXP4_OUT / "llm_simplified_defect.csv",
            EXP4_OUT / "llm_physics_defect.csv",
        ),
    ]

    for path in sorted(EXP4_OUT.glob("sr_tabular_athena_to_*.csv")):
        if path.name in {"sr_tabular_formulas_summary.csv"}:
            continue
        target = path.stem.replace("sr_tabular_", "")
        raw = target.replace("athena_to_", "").replace("_log", "")
        ttype = "tabular_athena_to_iv" if raw in IV_TARGETS else "tabular_athena_to_descriptor"
        title = f"表格 SR: Athena → {label_for(raw)}"
        jobs.append(
            (
                title,
                ttype,
                target,
                path,
                EXP4_OUT / f"llm_blind_math_tabular_{target}.csv",
                EXP4_OUT / f"llm_simplified_tabular_{target}.csv",
                EXP4_OUT / f"llm_physics_tabular_{target}.csv",
            )
        )

    for path in sorted(EXP4_OUT.glob("sr_tabular_full_to_*.csv")):
        target = path.stem.replace("sr_tabular_", "")
        raw = target.replace("full_to_", "")
        title = f"表格 SR (全特征): 工艺+剖面 → {label_for(raw)}"
        jobs.append(
            (
                title,
                "tabular_full_to_iv",
                target,
                path,
                EXP4_OUT / f"llm_blind_math_tabular_{target}.csv",
                EXP4_OUT / f"llm_simplified_tabular_{target}.csv",
                EXP4_OUT / f"llm_physics_tabular_{target}.csv",
            )
        )

    trio_path = EXP4_OUT / "sr_tabular_iv_trio_to_FF.csv"
    if trio_path.exists():
        jobs.append(
            (
                "表格 SR: Voc+Jsc+Eff → FF",
                "tabular_iv_trio_ff",
                "iv_trio_to_FF",
                trio_path,
                EXP4_OUT / "llm_blind_math_tabular_iv_trio_to_FF.csv",
                EXP4_OUT / "llm_simplified_tabular_iv_trio_to_FF.csv",
                EXP4_OUT / "llm_physics_tabular_iv_trio_to_FF.csv",
            )
        )

    return jobs


def build_ff_comparison_report() -> str:
    cmp_path = EXP4_OUT / "sr_ff_comparison.csv"
    if not cmp_path.exists():
        return ""
    df = pd.read_csv(cmp_path)
    lines = [
        "# FF 预测方案对比",
        "",
        "比较三种方式预测填充因子 iv_FF（test_R² 越高越好）：",
        "",
        "| 方法 | train_R² | test_R² | 说明 |",
        "| --- | --- | --- | --- |",
    ]
    labels = {
        "identity_FF_calc": "直接计算 FF=100·Eff/(Voc·Jsc)（实测三参数）",
        "sr_iv_trio": "SR：实测 Voc+Jsc+Eff → FF",
        "sr_full_features": "SR：全特征 → FF",
        "FF_from_SR_Voc_Jsc_Eff": "SR预测 Voc/Jsc/Eff 后代入公式算 FF",
    }
    for _, row in df.iterrows():
        method = str(row.get("method", ""))
        lines.append(
            f"| {labels.get(method, method)} | {_fmt_num(row.get('train_R2'))} | "
            f"{_fmt_num(row.get('test_R2'))} | {_esc_cell(str(row.get('preprocess_note', ''))[:80])} |"
        )
    lines += ["", "## 推荐", ""]
    lines.append(
        "- **工程采用**：FF 由 SR 预测的 Voc/Jsc/Eff 代入恒等式 "
        "`FF = 100 × Eff_pred / (Voc_pred × Jsc_pred)` 计算；不对 FF 单独做全特征 SR。"
    )
    lines.append(
        "- 有实测 IV 时，恒等式与实测值一致（test_R²≈1）；"
        "仅有工艺/曲线特征时，用各 IV 的 SR 公式预测后再代入。"
    )
    pred_row = df[df["method"] == "FF_from_SR_Voc_Jsc_Eff"]
    if not pred_row.empty and "test_MAE" in pred_row.columns:
        mae = pred_row.iloc[0].get("test_MAE")
        if pd.notna(mae):
            lines.append(f"- SR 预测 IV 后代入 FF：测试集 MAE ≈ {float(mae):.2f}%（见 `sr_plots/ff_methods_comparison.png`）。")
    lines.append("- 全特征直接 SR→FF（test_R²≈0.31）精度远低于上述恒等式路径，仅作对照。")
    lines.append("")
    return "\n".join(lines)


def build_index(job_entries: list[tuple[str, str]]) -> str:
    lines = [
        "# TOPCon 符号回归分析报告（索引）",
        "",
        "各任务已拆分为独立 Markdown，便于阅读。详细报告目录：`outputs/exp4_symbolic/sr_reports/`",
        "",
        f"- 每个任务展示前 **{SR_TOP_FORMULAS_IN_REPORT}** 条公式",
        "- 单报告结构：符号公式 → blind-math/简化 → 物理变量公式 → 物理评述",
        "",
        "## 报告列表",
        "",
    ]
    for title, slug in job_entries:
        lines.append(f"- [{title}](sr_reports/{slug}.md)")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    setup_runtime()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    job_entries: list[tuple[str, str]] = []

    for title, ttype, slug, sr_path, blind_p, simp_p, phys_p in _task_jobs():
        report = build_task_report(title, ttype, slug, sr_path, blind_p, simp_p, phys_p)
        out_path = REPORT_DIR / f"{slug}.md"
        out_path.write_text(report, encoding="utf-8")
        job_entries.append((title, slug))
        log(f"SR report -> {out_path}")

    ff_report = build_ff_comparison_report()
    if ff_report:
        ff_path = REPORT_DIR / "ff_comparison.md"
        ff_path.write_text(ff_report, encoding="utf-8")
        job_entries.append(("FF 预测方案对比", "ff_comparison"))
        log(f"SR report -> {ff_path}")

    pred_iv_path = EXP4_OUT / "sr_reports" / "ff_from_predicted_iv.md"
    if pred_iv_path.exists():
        job_entries.append(("FF 由 SR 预测 IV 后代入计算", "ff_from_predicted_iv"))

    index = build_index(job_entries)
    INDEX_PATH.write_text(index, encoding="utf-8")
    log(f"SR index -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
