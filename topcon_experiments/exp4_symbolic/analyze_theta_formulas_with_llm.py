"""LLM simplification + physics review for the process->theta SR formulas.

For each DG parameter (ln_N_p, z_p_um, z_f1, z_f2) the best PySR formula is
sent to DeepSeek twice: once for symbolic simplification, once for a physics
review (Arrhenius / diffusion-length consistency). A final call reviews the
assembled fully-symbolic chain N(z) = DG(z; theta(process)) * exp(r_SR).

Outputs (under outputs/exp4_symbolic/process_to_theta_sr/):
  * theta_llm_analysis.csv
  * theta_llm_analysis.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.log_utils import log, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.exp4_symbolic.llm_analysis import (
    chat,
    check_api_connectivity,
    create_client,
)
from topcon_experiments.exp4_symbolic.run_process_to_dg_chain import DG_TARGETS
from topcon_experiments.exp4_symbolic.run_process_to_theta_sr import OUT_DIR
from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy

OUT_CSV = OUT_DIR / "theta_llm_analysis.csv"
OUT_MD = OUT_DIR / "theta_llm_analysis.md"

TARGET_MEANING = {
    "ln_N_p": "双高斯峰值浓度的自然对数 ln(N_p[cm^-3])",
    "z_p_um": "双高斯峰位深度 z_p [um]",
    "z_f1": "峰前(表面侧)高斯宽度 z_f1 [um]",
    "z_f2": "峰后(体内侧)高斯宽度 z_f2 [um]，物理上对应扩散长度 sqrt(Dt)",
}


def _feature_legend() -> str:
    cols = json.loads((OUT_DIR / "feature_columns.json").read_text())
    notes = {
        "athena_thick": "BSG 淀积厚度 [um]",
        "athena_c_boron": "ln(硼源浓度)",
        "athena_temp1": "第1步扩散温度 [C]",
        "athena_time1": "第1步时间 [min]",
        "athena_temp2": "第2步扩散温度 [C]",
        "athena_time2": "第2步时间 [min]",
        "athena_F_N2": "N2 流量",
        "athena_F_O2": "O2 流量",
    }
    lines = []
    for i, c in enumerate(cols):
        extra = notes.get(c, "")
        if c.startswith("ln_dt"):
            ea = c.split("_ea")[-1]
            step = "1" if "dt1" in c else ("2" if "dt2" in c else "1+2总和")
            extra = f"ln(t*exp(-Ea/kT)) 热预算, 步骤{step}, Ea={ea} eV"
        elif c.startswith("frac_dt2"):
            extra = f"第2步热预算占比, Ea={c.split('_ea')[-1]} eV"
        elif c == "ln_thick_cboron":
            extra = "ln(thick)+ln(c_boron)，硼源剂量代理"
        lines.append(f"x{i} = {c}" + (f"（{extra}）" if extra else ""))
    return "\n".join(lines)


def main() -> None:
    setup_runtime()
    legend = _feature_legend()

    cached = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame()
    cached_tgt = set(cached["target"].astype(str)) if not cached.empty else set()
    rows = cached.to_dict("records") if not cached.empty else []

    pending = [t for t in DG_TARGETS if t not in cached_tgt]
    client = None
    if pending or "combined_chain" not in cached_tgt:
        check_api_connectivity()
        client = create_client()

    best_eqs = {}
    for tgt in DG_TARGETS:
        df = pd.read_csv(OUT_DIR / f"sr_formulas_theta_{tgt}.csv")
        best = sort_formulas_by_accuracy(df).iloc[0]
        best_eqs[tgt] = (str(best["equation"]), float(best["test_R2"]),
                         int(best["complexity"]))

    for tgt in pending:
        eq, r2, cx = best_eqs[tgt]
        simplify_prompt = f"""你是符号数学专家。请简化下面由 PySR 得到的 TOPCon 硼扩散工艺公式。

目标量：{TARGET_MEANING[tgt]}
公式（变量为 x0..x16）：{eq}
测试集 R² = {r2:.4f}，复杂度 = {cx}

变量定义：
{legend}

要求：
1. 将数值常数合并为少量符号参数 a,b,c...，给出紧凑形式；
2. 用物理变量名重写公式（替换 x 索引）；
3. 说明简化是严格等价还是近似；
4. 若公式结构可进一步化简（如 square(u)+c 形式），给出最简形式。
请用中文回答。"""
        physics_prompt = f"""你是硼在硅中扩散工艺的专家。请审查下面的经验公式的物理合理性。

目标量：{TARGET_MEANING[tgt]}
公式（变量为 x0..x16）：{eq}
测试集 R² = {r2:.4f}

变量定义：
{legend}

背景：该公式由符号回归从 ~4100 组 TCAD(Athena) 硼扩散仿真中拟合，
输入已包含 Arrhenius 型热预算特征 ln(t*exp(-Ea/kT))。
双高斯参数是对 N>=1e18 cm^-3 的硅内分布段拟合得到的。

请分项回答：
1. 公式依赖哪些物理量，方向性(单调性)是否符合扩散物理；
2. 与 Fick 扩散 / Arrhenius 定律的联系与偏离；
3. 哪些项只能视为经验修正；
4. 论文中可以如何表述这个公式（可接受/不可接受的说法）。
请用中文回答，避免过度解读。"""
        log(f"LLM analyzing theta[{tgt}] ...")
        rows.append({
            "target": tgt,
            "equation": eq,
            "test_R2": r2,
            "complexity": cx,
            "simplify_response": chat(
                client, [{"role": "user", "content": simplify_prompt}],
                f"theta/{tgt}/simplify"),
            "physics_response": chat(
                client, [{"role": "user", "content": physics_prompt}],
                f"theta/{tgt}/physics"),
        })
        save_csv(pd.DataFrame(rows), OUT_CSV)

    if "combined_chain" not in cached_tgt:
        theta_block = "\n".join(
            f"{t} = {best_eqs[t][0]}   (test R²={best_eqs[t][1]:.3f})"
            for t in DG_TARGETS
        )
        combined_prompt = f"""你是 TOPCon 太阳能电池仿真与符号回归专家。以下是一条全符号化的
"工艺参数 -> 掺杂曲线" 预测链，请给出整体评述与论文表述建议。

第一层（形状先验，文献双高斯）：
N(z) = N_p * exp(-((z-z_p)/z_f1)^2)  当 z <  z_p
N(z) = N_p * exp(-((z-z_p)/z_f2)^2)  当 z >= z_p
（可选残差修正：N(z) *= exp(r_SR(z, z_p, z_f2))，r_SR 为已有 PySR 公式）

第二层（工艺 -> 双高斯参数，PySR 公式，变量 x0..x16）：
{theta_block}

变量定义：
{legend}

在 1034 个独立测试样本上，全符号链整曲线中位 R²_log≈0.89
（对比：黑箱GBM同链条 0.945；曲线自拟合参数上限 0.99）。

请回答：
1. 这条链的科学价值与局限该如何客观表述；
2. z_p、z_f1 公式精度低（0.63/0.29）的可能物理与数据原因；
3. 论文中推荐的公式呈现方式（哪些放正文、哪些放SI）。
请用中文回答。"""
        log("LLM analyzing combined chain ...")
        rows.append({
            "target": "combined_chain",
            "equation": "N(z)=DG(z;theta(process)) [* exp(r_SR)]",
            "test_R2": float("nan"),
            "complexity": 0,
            "simplify_response": "",
            "physics_response": chat(
                client, [{"role": "user", "content": combined_prompt}],
                "theta/combined/review"),
        })
        save_csv(pd.DataFrame(rows), OUT_CSV)

    # Markdown report
    lines = ["# 工艺→双高斯参数 SR 公式：LLM 简化与物理审查", ""]
    for row in rows:
        lines += [f"## {row['target']}", "",
                  f"- 公式: `{row['equation']}`",
                  f"- test R² = {row['test_R2']}", ""]
        if row.get("simplify_response"):
            lines += ["### 简化", "", str(row["simplify_response"]), ""]
        lines += ["### 物理审查", "", str(row["physics_response"]), ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log(f"LLM analysis -> {OUT_CSV}")
    log(f"Markdown -> {OUT_MD}")


if __name__ == "__main__":
    main()
