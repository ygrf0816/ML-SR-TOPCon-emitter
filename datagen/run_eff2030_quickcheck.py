"""
run_eff2030_quickcheck.py

用途:
- 读取项目根下 `generated_eff_20_30_curves.csv`(效率 20-30)，每个效率(`eff_target`)只取两个样本，
  将曲线导出为一批 `.txt` 掺杂曲线(两列: x 与浓度，x 在 [0,2] 均匀取样)，
  使用 `topcon_n_cd` 进行快速仿真，输出 `.log/.str/.csv` 到新目录，
  参考 `analyze_scans.py` 计算四个指标(Voc, Jsc, FF, Eff)，保存汇总 CSV，
  并画出输入目标效率 vs 仿真得到效率(Eff_sim) 的对比图(含 y=x 参考线)。

功能特性:
- 单次仿真超时控制：当仿真超过2分钟时自动跳过，避免长时间卡住
- 使用 threading 实现超时机制，不依赖 signal 模块
- 跳过已存在的仿真结果，支持断点续跑

说明:
- 可直接在 IDE 中运行，不依赖命令行参数。按需修改脚本内路径常量。
"""

import os
import time
import threading
import subprocess
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from python_tcad import pythontcad
import tcadmodel
from analyze_scans import read_iv_from_csv, compute_metrics


# 路径常量(按需修改)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
CSV_PATH = os.path.join(PROJECT_ROOT, "generated_eff_20_30_curves.csv")
OUT_TXT_DIR = os.path.join(PROJECT_ROOT, "eff2030_txt")
OUT_SIM_DIR = os.path.join(PROJECT_ROOT, "eff2030_jv")
OUT_SUMMARY_CSV = os.path.join(PROJECT_ROOT, "eff2030_summary.csv")
OUT_FIG = os.path.join(PROJECT_ROOT, "eff2030_eff_compare.png")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def select_two_per_eff(df: "pd.DataFrame") -> "pd.DataFrame":
    # 每个 eff_target 取前两个(按 idx_in_group 升序)
    if "idx_in_group" in df.columns:
        df_sorted = df.sort_values(["eff_target", "idx_in_group"])  # 保持稳定顺序
    else:
        df_sorted = df.sort_values(["eff_target"]).copy()
        df_sorted["idx_in_group"] = df_sorted.groupby("eff_target").cumcount()
    picked = (
        df_sorted.groupby("eff_target", as_index=False)
        .head(20)
        .reset_index(drop=True)
    )
    return picked


def row_to_txt(row: "pd.Series", conc_cols: List[str], out_dir: str) -> str:
    eff = float(row.get("eff_target", row.get("Eff", 0.0)))
    idx = int(row.get("idx_in_group", 0))
    base = f"eff{eff:.2f}_idx{idx:03d}"
    txt_path = os.path.join(out_dir, f"{base}.txt")

    N = row[conc_cols].astype(float).to_numpy()
    npts = int(N.size)
    x = np.linspace(0.0, 2.0, npts, dtype=float)  # 均匀 0-2

    # 写两列: x \t N
    with open(txt_path, "w", encoding="utf-8") as f:
        for xi, Ni in zip(x, N):
            f.write(f"{xi:.15g}\t{Ni:.6g}\n")
    return txt_path


def run_simulation_with_timeout(a: pythontcad, timeout_seconds: int = 120) -> bool:
    """
    在指定超时时间内运行仿真
    
    Args:
        a: pythontcad 实例
        timeout_seconds: 超时时间（秒），默认120秒（2分钟）
    
    Returns:
        bool: True表示仿真成功完成，False表示超时或失败
    """
    result = {"success": False, "error": None}
    
    def run_simulation():
        try:
            a.tcad.updatecommand()
            a.updateprocess()
            for cmd in a.tcad.get_allcommandlist():
                a.run_atlas_command(cmd)
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
    
    # 创建线程运行仿真
    sim_thread = threading.Thread(target=run_simulation)
    sim_thread.daemon = True
    sim_thread.start()
    
    # 等待仿真完成或超时
    sim_thread.join(timeout=timeout_seconds)
    
    if sim_thread.is_alive():
        # 仿真超时，线程仍在运行
        print(f"  仿真超时（{timeout_seconds}秒），跳过此仿真")
        return False
    
    if result["error"]:
        print(f"  仿真执行出错: {result['error']}")
        return False
    
    return result["success"]


def simulate_one(txt_path: str, sim_dir: str, exe_dir: str) -> Dict:
    base = os.path.splitext(os.path.basename(txt_path))[0]
    outlog = os.path.join(sim_dir, f"{base}.log")
    outstr = os.path.join(sim_dir, f"{base}.str")
    outcsv = os.path.join(sim_dir, f"{base}.csv")

    a = pythontcad()
    if exe_dir:
        a.set_tcad(exe_dir)
    a.tcad = tcadmodel.topcon_n_cd()

    nk_tpl_dir = os.path.dirname(__file__)
    a.tcad.nkpath = nk_tpl_dir + os.sep
    a.tcad.template_lib = os.path.join(nk_tpl_dir, "template.lib")

    a.tcad.cdpath = txt_path
    a.tcad.outlog = outlog
    a.tcad.outstr = outstr
    a.tcad.outcsv = outcsv

    # 使用超时机制运行仿真
    success = run_simulation_with_timeout(a, timeout_seconds=120)
    
    if not success:
        raise TimeoutError("仿真超时或执行失败")

    return {
        "txt_path": txt_path,
        "outcsv": outcsv,
        "outlog": outlog,
        "outstr": outstr,
    }


def main():
    ensure_dir(OUT_TXT_DIR)
    ensure_dir(OUT_SIM_DIR)

    # 读取数据集
    df = pd.read_csv(CSV_PATH)
    # 识别浓度列
    conc_cols = [c for c in df.columns if str(c).startswith("conc_")]
    if not conc_cols:
        raise ValueError("未在 CSV 中找到 conc_* 列。")

    # 选择每个效率两个样本
    picked = select_two_per_eff(df)

    exe_dir = os.environ.get("TCAD_EXE_DIR", "D:/silvaco/exe")

    records: List[Dict] = []
    print(f"计划仿真条目: {len(picked)} (每个效率 2 条)")

    # 先导出 txt，再仿真
    for i, row in picked.iterrows():
        eff_target = float(row.get("eff_target", row.get("Eff", 0.0)))
        idx = int(row.get("idx_in_group", i))
        try:
            txt_path = row_to_txt(row, conc_cols, OUT_TXT_DIR)
        except Exception as e:
            print(f"导出TXT失败 eff={eff_target} idx={idx}: {e}")
            continue

        # 若已存在仿真结果，直接跳过执行，仅做结果读取
        base = os.path.splitext(os.path.basename(txt_path))[0]
        outcsv = os.path.join(OUT_SIM_DIR, f"{base}.csv")
        if os.path.isfile(outcsv):
            print(f"跳过仿真(已存在): {base}")
            sim_info = {"txt_path": txt_path, "outcsv": outcsv, "outlog": os.path.join(OUT_SIM_DIR, f"{base}.log"), "outstr": os.path.join(OUT_SIM_DIR, f"{base}.str")}
        else:
            print(f"运行仿真: eff={eff_target:.2f} idx={idx:03d}")
            try:
                sim_info = simulate_one(txt_path, OUT_SIM_DIR, exe_dir)
            except TimeoutError as e:
                print(f"  仿真超时: {e}")
                continue
            except Exception as e:
                print(f"  仿真失败: {e}")
                continue

        # 读取仿真 IV 并计算指标
        try:
            V, I = read_iv_from_csv(sim_info["outcsv"])
            met = compute_metrics(V, I)
            rec = {
                "eff_target": eff_target,
                "idx_in_group": idx,
                "base": base,
                "txt_path": sim_info["txt_path"],
                "outcsv": sim_info["outcsv"],
                "Voc_sim": met["Voc"],
                "Jsc_sim": met["Jsc"],
                "FF_sim": met["FF"],
                "Eff_sim": met["Eff"],
            }
            records.append(rec)
        except Exception as e:
            print(f"  结果解析失败: {e}")
            continue

    if not records:
        print("无有效结果，已结束。")
        return

    # 导出汇总 CSV
    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"已输出汇总: {OUT_SUMMARY_CSV}")

    # 对比图：eff_target vs Eff_sim
    plt.figure(figsize=(7, 6))
    x = df_out["eff_target"].astype(float).to_numpy()
    y = df_out["Eff_sim"].astype(float).to_numpy()
    plt.scatter(x, y, s=18, label="simulation")
    # 画 y=x
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    lo = min(x_min, float(np.nanmin(y)))
    hi = max(x_max, float(np.nanmax(y)))
    xs = np.linspace(lo, hi, 100)
    plt.plot(xs, xs, "k--", lw=1, label="y=x")
    plt.xlabel("eff_target")
    plt.ylabel("Eff_sim")
    plt.title("Efficiency target vs simulated")
    plt.grid(True, ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"已输出对比图: {OUT_FIG}")


if __name__ == "__main__":
    main()


