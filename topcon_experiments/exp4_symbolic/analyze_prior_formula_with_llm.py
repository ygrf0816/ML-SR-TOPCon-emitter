"""LLM simplification and physics review for the DG-prior residual SR formulas."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import OUTPUT_ROOT
from topcon_experiments.exp4_symbolic.llm_analysis import (
    chat,
    check_api_connectivity,
    create_client,
)
from topcon_experiments.exp4_symbolic.sr_equation_utils import sort_formulas_by_accuracy

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"
FORMULA_PATH = EXP4_OUT / "curve_prior_experiment" / "sr_formulas_doping_residual_full.csv"
OUT_CSV = EXP4_OUT / "curve_prior_experiment" / "prior_residual_full_llm_analysis.csv"
OUT_MD = EXP4_OUT / "curve_prior_experiment" / "prior_residual_full_llm_analysis.md"
TOP_N = 5


def _physical_equation(equation: str) -> str:
    """Replace only variables used by the winning prior-residual formulas."""
    mapping = {
        18: "z",
        20: "z_p",
        22: "z_f2",
        19: "N_p",
        21: "z_f1",
    }
    out = str(equation)
    for idx in sorted(mapping, reverse=True):
        out = re.sub(rf"\bx{idx}\b", mapping[idx], out)
    return out


def _metrics(row: pd.Series) -> dict[str, float | int]:
    keys = ("complexity", "loss", "train_R2", "test_R2", "train_MSE", "test_MSE")
    return {k: row[k] for k in keys if k in row and pd.notna(row[k])}


def _load_cached() -> pd.DataFrame:
    return pd.read_csv(OUT_CSV) if OUT_CSV.exists() else pd.DataFrame()


def _write_markdown(df: pd.DataFrame) -> None:
    lines = [
        "# 双高斯先验残差 SR：LLM 简化与物理含义分析",
        "",
        "完整模型：$N(z)=N_{DG}(z)\\exp[r_{SR}(z)]$；下列公式均为对数残差 $r_{SR}$，",
        "不是掺杂浓度 $N$ 本身。变量：$z$=绝对物理深度，$z_p$=双高斯峰位，",
        "$z_{f2}$=峰右侧宽度。当前最优公式不直接使用 Athena 工艺变量。",
        "",
        "> 注意：仿真检验中 residual-SR 相对纯双高斯的中位 ΔR² 接近 0 或略负，",
        "因此物理解读只能视为经验形状修正，不应宣称发现了新的扩散定律。",
        "",
    ]
    for _, row in df.sort_values("rank").iterrows():
        lines += [
            f"## Rank {int(row['rank'])}",
            "",
            f"- complexity: {row['complexity']}",
            f"- test_R²（残差空间）: {row['test_R2']}",
            f"- 原式：`{row['equation']}`",
            f"- 物理变量式：`{row['physical_equation']}`",
            "",
            "### LLM 简化",
            "",
            str(row["simplify_response"]),
            "",
            "### LLM 物理解读",
            "",
            str(row["physics_response"]),
            "",
        ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not FORMULA_PATH.exists():
        raise FileNotFoundError(FORMULA_PATH)
    formulas = sort_formulas_by_accuracy(pd.read_csv(FORMULA_PATH)).head(TOP_N)
    cached = _load_cached()
    cached_eq = set(cached["equation"].astype(str)) if not cached.empty else set()
    rows = cached.to_dict("records") if not cached.empty else []

    pending = [row for _, row in formulas.iterrows() if str(row["equation"]) not in cached_eq]
    if pending:
        check_api_connectivity()
        client = create_client()
    else:
        client = None

    for rank, (_, row) in enumerate(formulas.iterrows(), start=1):
        equation = str(row["equation"])
        if equation in cached_eq:
            continue
        physical = _physical_equation(equation)
        metrics = _metrics(row)
        simplify_prompt = f"""你是符号数学专家。请简化下面的 TOPCon 掺杂曲线对数残差公式。

完整重建关系：N(z) = N_DG(z) * exp(r_SR(z))
当前公式是 r_SR，不是 N。
变量：z=绝对物理深度(um)，z_p=双高斯峰位(um)，z_f2=右侧宽度(um)。
原始 PySR 公式：{equation}
替换物理变量后：{physical}
指标：{json.dumps(metrics, ensure_ascii=False)}

要求：
1. 用少量符号参数 a,b,c... 合并数值常数，给出紧凑公式；
2. 明确每个新参数与原常数的对应关系；
3. 不得删除 Abs 或四次幂等会改变函数形状的结构；
4. 说明简化是严格等价还是近似；若近似，说明可能损失。
请用中文回答。"""
        physics_prompt = f"""你是 TOPCon 硼扩散与符号回归专家，请审查以下经验残差公式的物理含义。

完整关系：N(z)=N_DG(z;N_p,z_p,z_f1,z_f2)*exp(r_SR(z))
残差公式：{physical}
指标：{json.dumps(metrics, ensure_ascii=False)}
已知事实：该候选式主要只依赖 z、z_p、z_f2，不直接依赖温度、时间、硼源浓度；
在严格去除 BSG 后的仿真测试中，最终模型 R² 很高，但相对纯 DG 的 ΔR² 中位接近 0 或略负；
文献参考曲线本身也是双高斯合成数据。

请分项回答：
1. 数学结构和无量纲组合；
2. 对峰后尾部/峰位附近的作用；
3. 与 Fick/高斯扩散先验是否相容；
4. 哪些部分只能视为经验修正、不能作物理定律；
5. 论文中可接受与不可接受的表述。
请用中文回答，避免过度解读。"""
        rows.append({
            "rank": rank,
            "complexity": row.get("complexity"),
            "test_R2": row.get("test_R2"),
            "equation": equation,
            "sympy_format": row.get("sympy_format", ""),
            "physical_equation": physical,
            "simplify_response": chat(
                client, [{"role": "user", "content": simplify_prompt}],
                f"prior_residual_full/rank{rank}/simplify",
            ),
            "physics_response": chat(
                client, [{"role": "user", "content": physics_prompt}],
                f"prior_residual_full/rank{rank}/physics",
            ),
        })
        save_csv(pd.DataFrame(rows), OUT_CSV)

    result = pd.DataFrame(rows)
    # Keep only current top-N formulas and refresh ranks after cached reruns.
    rank_by_eq = {str(row["equation"]): i for i, (_, row) in enumerate(formulas.iterrows(), start=1)}
    result = result[result["equation"].astype(str).isin(rank_by_eq)].copy()
    result["rank"] = result["equation"].astype(str).map(rank_by_eq)
    result = result.sort_values("rank").reset_index(drop=True)
    save_csv(result, OUT_CSV)
    _write_markdown(result)
    print(f"LLM analysis -> {OUT_CSV}")
    print(f"Markdown report -> {OUT_MD}")


if __name__ == "__main__":
    main()
