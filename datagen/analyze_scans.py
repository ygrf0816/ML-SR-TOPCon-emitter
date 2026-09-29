import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.use_mathtext'] = False
try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    print("需要 pandas 库: pip install pandas", file=sys.stderr)
    raise

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print("需要 matplotlib 库: pip install matplotlib", file=sys.stderr)
    raise


def find_column(df: "pd.DataFrame", keywords: List[str]) -> str:
    """在列名中查找包含所有关键字的列，返回列名。大小写不敏感。"""
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    for name_lower, original in lower.items():
        if all(k.lower() in name_lower for k in keywords):
            return original
    raise KeyError(f"未找到列，需包含关键字: {keywords}，已存在列: {cols}")


def read_iv_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 CSV，返回 (V, I) 数组。优先取 anode Voltage 与 cathode Current。"""
    df = pd.read_csv(csv_path)
    # 去除列名前后空格与引号
    df.columns = [str(c).strip().strip('"\'') for c in df.columns]

    # 电压列（优先 anode Voltage，否则 cathode Voltage）
    try:
        v_col = find_column(df, [ "anode Voltage (V)"])
    except KeyError:
        v_col = find_column(df, ["cathode", "voltage"])

    # 电流列（优先 cathode Current，否则 anode Current）
    try:
        i_col = find_column(df, [ "cathode Int. Voltage (V)"])  # 与仿真脚本一致
    except KeyError:
        i_col = find_column(df, ["anode", "current"])

    V = df[v_col].astype(float).to_numpy()
    I = df[i_col].astype(float).to_numpy()
    return V, I


def interp_at_x(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """在线性插值下计算 y(x0)。假设 x 单调。"""
    # 找到最接近 x0 的区间
    idx = np.searchsorted(x, x0)
    if idx == 0:
        return float(y[0])
    if idx >= len(x):
        return float(y[-1])
    x1, x2 = x[idx - 1], x[idx]
    y1, y2 = y[idx - 1], y[idx]
    if x2 == x1:
        return float(y1)
    t = (x0 - x1) / (x2 - x1)
    return float(y1 + t * (y2 - y1))


def find_zero_crossing_x(x: np.ndarray, y: np.ndarray) -> float:
    """寻找 y(x)=0 的零点（线性插值）。如无符号变化，则返回 NaN。"""
    s = np.sign(y)
    sign_changes = np.where(np.diff(s) != 0)[0]
    if len(sign_changes) == 0:
        return float("nan")
    i = sign_changes[0]
    x1, x2 = x[i], x[i + 1]
    y1, y2 = y[i], y[i + 1]
    if y2 == y1:
        return float(x1)
    return float(x1 - y1 * (x2 - x1) / (y2 - y1))


def compute_metrics(
    V: np.ndarray,
    I: np.ndarray,
    area_cm2: float = 1.0,
    pin_W_per_cm2: float = 0.1,
    scale_v: float = 1,
    scale_i: float = 1.16,
    use_empirical: bool = True,
) -> Dict[str, float]:
    """
    按经验公式计算：
    - V_emp = V * 1.06524
    - I_emp = I * 1.31358
    - Pm(mW/cm^2) = max(V_emp * I_emp) * 1e11
    - Vm = 对应 Pmax 的 V_emp
    - Im(mA/cm^2) = Pm / Vm
    - Voc = I_emp=0 时的 V_emp
    - Jsc(mA/cm^2) = I_emp(V=0) * 1e11
    - FF(%) = Pm / (Jsc * Voc) * 100
    - Eff：若 use_empirical=True，则返回经验式中的 `Pm`（与原提取式一致）；否则按标准口径 Eff(%) = Pm/100mW·cm^-2×100
    """
    # 标度
    V_emp = V.astype(float) * scale_v
    I_emp = I.astype(float) * scale_i

    # 按 V 升序，便于插值与查找
    order = np.argsort(V_emp)
    V_emp = V_emp[order]
    I_emp = I_emp[order]

    # Voc：I_emp=0 的电压
    Voc = find_zero_crossing_x(V_emp, I_emp)

    # Jsc：V=0 的电流（转 mA/cm^2）
    Isc_at_v0 = interp_at_x(V_emp, I_emp, 0.0)
    Jsc = Isc_at_v0 * 1e11

    # 功率曲线：P = V_emp * I_emp（转 mW/cm^2）
    P = V_emp * I_emp
    idx_pmax = int(np.nanargmax(P))
    Pm = float(P[idx_pmax]) * 1e11
    Vm = float(V_emp[idx_pmax])
    Im = float(Pm / Vm) if Vm != 0 else float("nan")  # mA/cm^2

    # FF：
    FF = float("nan") if (np.isnan(Voc) or Jsc == 0) else Pm / (Jsc * Voc) * 100.0

    # Eff：经验式返回 Pm，本质与原提取式一致；否则用标准口径
    if use_empirical:
        Eff = Pm
    else:
        Eff = (Pm / (pin_W_per_cm2 * 1e3)) * 100.0  # 若 Pm 单位为 mW/cm^2

    return {
        "Voc": Voc,
        "Isc": Isc_at_v0,   # 原始电流（A），便于参考
        "Jsc": Jsc,         # mA/cm^2
        "Pm": Pm,           # mW/cm^2
        "Vm": Vm,
        "Im": Im,           # mA/cm^2
        "FF": FF,
        "Eff": Eff,         # 经验式：与 Pm 相同口径
    }


def parse_param_from_filename(fname: str) -> Tuple[str, float]:
    """从文件名解析扫描参数与取值，返回 (param_name, value)。大小写不敏感。
    例：topcon_n_resist_rear_0.01.csv -> (resist_rear, 0.01)
        Topcon_n_Nt_top_N_10000000000.0.csv -> (nt_top_n, 1.0e10)
    """
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)
    name_lower = name.lower()
    # 去掉前缀
    if name_lower.startswith("topcon_n_"):
        name_lower = name_lower[len("topcon_n_") :]
    parts = name_lower.split("_")
    if len(parts) < 2:
        return ("unknown", float("nan"))
    # 参数名可能包含下划线，数值为最后一段
    try:
        value = float(parts[-1])
    except ValueError:
        # 尝试提取科学计数或混合
        m = re.search(r"(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", parts[-1])
        value = float(m.group(1)) if m else float("nan")
    param = "_".join(parts[:-1])
    return (param, value)


def collect_scan_groups(scan_dir: str) -> Dict[str, List[Tuple[float, str]]]:
    groups: Dict[str, List[Tuple[float, str]]] = {}
    for root, _, files in os.walk(scan_dir):
        for f in files:
            if f.lower().endswith(".csv") and f.lower().startswith("topcon_n_"):
                full = os.path.join(root, f)
                param, val = parse_param_from_filename(f)
                groups.setdefault(param, []).append((val, full))
    # 排序
    for k in list(groups.keys()):
        groups[k].sort(key=lambda x: x[0])
    return groups


def summarize_and_plot(groups: Dict[str, List[Tuple[float, str]]], area_cm2: float, pin_W_per_cm2: float, save: bool, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    for param, items in groups.items():
        metrics_rows = []
        plt.figure(figsize=(10, 7))
        for val, path in items:
            try:
                V, I = read_iv_from_csv(path)
                met = compute_metrics(V, I, area_cm2=area_cm2, pin_W_per_cm2=pin_W_per_cm2)
                metrics_rows.append({"param": param, "value": val, **met, "file": path})
                # 画 IV
                plt.plot(V, -I, label=f"{param}={val}")  # 画 -I 方便正向视图
            except Exception as e:
                print(f"跳过 {path}: {e}")
                continue

        if not metrics_rows:
            plt.close()
            continue

        # 保存/显示 IV 曲线
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title(f"IV curves - scan {param}")
        plt.legend(fontsize=8)
        plt.grid(True, ls=":", alpha=0.4)
        if save:
            iv_png = os.path.join(outdir, f"iv_{param}.png")
            plt.savefig(iv_png, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

        # 汇总表与趋势图
        dfm = pd.DataFrame(metrics_rows).sort_values("value")
        if save:
            csv_out = os.path.join(outdir, f"summary_{param}.csv")
            dfm.to_csv(csv_out, index=False)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        ax = axes.ravel()
        ax[0].plot(dfm["value"], dfm["Voc"], marker="o")
        ax[0].set_title("Voc")
        ax[0].set_xlabel(param)
        ax[0].set_xscale('log')
        ax[0].set_ylabel("Voc (V)")
        ax[0].grid(True, ls=":", alpha=0.4)

        ax[1].plot(dfm["value"], abs(dfm["Jsc"]), marker="o")
        ax[1].set_title("Jsc")
        ax[1].set_xlabel(param)
        ax[1].set_xscale('log')
        ax[1].set_ylabel("Jsc mA/cm^2")
        ax[1].grid(True, ls=":", alpha=0.4)

        ax[2].plot(dfm["value"], dfm["FF"], marker="o")
        ax[2].set_title("FF")
        ax[2].set_xlabel(param)
        ax[2].set_xscale('log')
        ax[2].set_ylabel("FF (%)")
        ax[2].grid(True, ls=":", alpha=0.4)

        ax[3].plot(dfm["value"], dfm["Eff"], marker="o")
        ax[3].set_title("Efficiency")
        ax[3].set_xlabel(param)
        ax[3].set_xscale('log')
        ax[3].set_ylabel("Eff (%)")
        ax[3].grid(True, ls=":", alpha=0.4)

        plt.suptitle(f"Scan summary - {param}")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        if save:
            png_out = os.path.join(outdir, f"summary_{param}.png")
            plt.savefig(png_out, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()


def main():

    groups=collect_scan_groups("D:\\code\\silvaco\\tcadproject\\saomiao")
    summarize_and_plot(groups,area_cm2=1.0, pin_W_per_cm2=0.1, save=True, outdir="D:\\code\\silvaco\\tcadproject\\saomiao\\analysis")
    print("完成。")


if __name__ == "__main__":
    main()


