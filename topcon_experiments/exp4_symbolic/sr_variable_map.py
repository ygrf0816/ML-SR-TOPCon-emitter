"""PySR x-index to dataset column and physical-symbol mapping (parameter_name.md)."""

from __future__ import annotations

import re

from topcon_experiments.config import (
    ATHENA_FEATURES,
    CURVE_DESCRIPTORS,
    DOPING_DESCRIPTORS,
    DEFECT_DESCRIPTORS,
    MODEL2_FEATURES,
)

# Physical symbols from outputs/exp4_symbolic/parameter_name.md
PHYSICAL_SYMBOL: dict[str, str] = {
    "athena_thick": r"d_{BSG}",
    "athena_c_boron": r"N_{as}",
    "athena_temp1": r"T_{1}",
    "athena_time1": r"t_{1}",
    "athena_temp2": r"T_{2}",
    "athena_time2": r"t_{2}",
    "athena_F_N2": r"F_{N2}",
    "athena_F_O2": r"F_{O2}",
    "depth_um": r"x",
    "doping_N_peak": r"N_{a,peak}",
    "doping_x_peak": r"d_{a,peak}",
    "doping_junction_depth": r"d_{junc}",
    "doping_FWHM": r"FWHM",
    "doping_gradient_max": r"N_{a,grad}",
    "doping_dose": r"N_{a,eff}",
    "doping_R_sheet": r"R_{sheet}",
    "defect_vac_N_peak": r"N_{t,peak}",
    "defect_vac_gradient_max": r"N_{t,grad}",
    "defect_vac_dose": r"N_{t,eff}",
    "iv_Voc": r"V_{oc}",
    "iv_Jsc": r"J_{sc}",
    "iv_Eff": r"Eff",
    "iv_FF": r"FF",
}

PHYSICAL_LABEL_ZH: dict[str, str] = {
    "athena_thick": "BSG 厚度 (μm)",
    "athena_c_boron": "硼源浓度 (cm⁻³)",
    "athena_temp1": "一次扩散温度 (℃)",
    "athena_time1": "一次扩散时间 (min)",
    "athena_temp2": "二次扩散温度 (℃)",
    "athena_time2": "二次扩散时间 (min)",
    "athena_F_N2": "二次扩散 N₂ 流量 (L/min)",
    "athena_F_O2": "二次扩散 O₂ 流量 (L/min)",
    "depth_um": "深度 (μm)",
    "doping_N_peak": "掺杂峰值浓度 (cm⁻³)",
    "doping_x_peak": "掺杂峰值位置 (μm)",
    "doping_junction_depth": "结深 (μm)",
    "doping_FWHM": "半高全宽 (μm)",
    "doping_gradient_max": "掺杂曲线最大梯度 (cm⁻³/μm)",
    "doping_dose": "积分掺杂浓度 (cm⁻³)",
    "doping_R_sheet": "前部发射极方阻 (Ω/□)",
    "defect_vac_N_peak": "缺陷峰值浓度 (cm⁻³)",
    "defect_vac_gradient_max": "缺陷曲线最大梯度 (cm⁻³/μm)",
    "defect_vac_dose": "积分缺陷浓度 (cm⁻³)",
    "iv_Voc": "开路电压 (V)",
    "iv_Jsc": "短路电流密度 (mA/cm²)",
    "iv_Eff": "效率 / PCE (%)",
    "iv_FF": "填充因子 (%)",
}

LOG_TRANSFORMED_FEATURES = {
    "athena_c_boron",
    "doping_N_peak",
    "doping_dose",
    "defect_vac_N_peak",
    "defect_vac_dose",
}


def feature_order_for_task(task_type: str) -> list[str]:
    if task_type == "doping_curve_tail_full":
        return list(MODEL2_FEATURES) + ["depth_um"]
    if task_type == "curve":
        return ATHENA_FEATURES + ["depth_um"]
    if task_type == "tabular_iv_trio_ff":
        return ["iv_Voc", "iv_Jsc", "iv_Eff"]
    if task_type in ("tabular_athena_to_iv", "tabular_athena_to_descriptor"):
        return list(ATHENA_FEATURES)
    if task_type == "tabular_full_to_iv":
        return list(MODEL2_FEATURES)
    return list(ATHENA_FEATURES)


def physical_symbol(col: str, *, log_space: bool = False) -> str:
    sym = PHYSICAL_SYMBOL.get(col, col)
    if log_space and col in LOG_TRANSFORMED_FEATURES:
        return rf"\log({sym})"
    return sym


def substitute_equation(equation: str, feature_order: list[str], *, log_features: set[str] | None = None) -> str:
    """Replace x0..xN with physical symbols (highest index first)."""
    if not isinstance(equation, str):
        return ""
    log_features = log_features or set()
    out = equation
    for i in range(len(feature_order) - 1, -1, -1):
        col = feature_order[i]
        sym = physical_symbol(col, log_space=col in log_features)

        def _repl(_m: re.Match, replacement: str = sym) -> str:
            return replacement

        out = re.sub(rf"\bx{i}\b", _repl, out)
    return out


def variable_legend_table(task_type: str, *, log_features: set[str] | None = None) -> str:
    log_features = log_features or set()
    features = feature_order_for_task(task_type)
    lines = [
        "| PySR | 列名 | 物理符号 | 含义 |",
        "| --- | --- | --- | --- |",
    ]
    for i, col in enumerate(features):
        log_note = " (训练取 log)" if col in log_features else ""
        sym = physical_symbol(col, log_space=col in log_features)
        zh = PHYSICAL_LABEL_ZH.get(col, col)
        lines.append(f"| x{i} | `{col}`{log_note} | ${sym}$ | {zh} |")
    return "\n".join(lines) + "\n"


def log_features_for_task(task_type: str) -> set[str]:
    if task_type in ("tabular_full_to_iv", "curve", "doping_curve_tail_full"):
        return LOG_TRANSFORMED_FEATURES
    return set()
