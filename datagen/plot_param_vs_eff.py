"""工艺输入参数 vs 转换效率散点图（数据质量快速检查）。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "dataset" / "topcon_dataset.csv"
DEFAULT_OUT = SCRIPT_DIR / "dataset" / "plots"

# 14 个核心工艺输入参数（athena_* 列）
INPUT_PARAMS: dict[str, str] = {
    "athena_sub_resis": "衬底电阻率 (Ω·cm)",
    "athena_light_dose": "轻掺剂量 (cm⁻²)",
    "athena_light_energy": "轻掺能量 (keV)",
    "athena_light_drive_time": "轻掺推扩时间 (min)",
    "athena_light_drive_temp": "轻掺推扩温度 (°C)",
    "athena_light_oxide_thick": "轻掺氧化层厚度 (μm)",
    "athena_heavy_dose": "重掺剂量 (cm⁻²)",
    "athena_heavy_energy": "重掺能量 (keV)",
    "athena_mask_oxide_thick": "掩膜氧化层厚度 (μm)",
    "athena_select_boundary_x": "选择边界 x (μm)",
    "athena_anneal_temp": "退火温度 (°C)",
    "athena_anneal_time": "退火时间 (min)",
    "athena_repair_temp": "修复温度 (°C)",
    "athena_repair_time": "修复时间 (min)",
}

EFF_COL = "iv_Eff"


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if EFF_COL not in df.columns:
        raise KeyError(f"数据集中缺少 {EFF_COL} 列")
    df = df[df[EFF_COL].notna() & (df[EFF_COL] > 0)].copy()
    return df


def plot_all(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(INPUT_PARAMS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes_flat = axes.flatten()

    for ax, (col, label) in zip(axes_flat, INPUT_PARAMS.items()):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        x = df[col]
        y = df[EFF_COL]
        ax.scatter(x, y, alpha=0.75, edgecolors="white", linewidths=0.4, s=36)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("转换效率 Eff (%)", fontsize=9)
        short = col.replace("athena_", "")
        ax.set_title(short, fontsize=10)
        ax.grid(True, alpha=0.3)
        if x.max() / max(x.min(), 1e-30) > 100:
            ax.set_xscale("log")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"工艺输入参数 vs 转换效率（n={len(df)}）",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    overview = out_dir / "param_vs_eff_overview.png"
    fig.savefig(overview, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 总览图: {overview}")

    for col, label in INPUT_PARAMS.items():
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        x = df[col]
        y = df[EFF_COL]
        ax.scatter(x, y, alpha=0.8, edgecolors="white", linewidths=0.5, s=50)
        ax.set_xlabel(label)
        ax.set_ylabel("转换效率 Eff (%)")
        ax.set_title(f"{label} vs Eff")
        ax.grid(True, alpha=0.3)
        if x.max() / max(x.min(), 1e-30) > 100:
            ax.set_xscale("log")
        fname = out_dir / f"{col.replace('athena_', '')}_vs_eff.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] 单参数图目录: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制工艺参数-效率散点图")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    df = load_data(args.csv)
    print(f"[INFO] 有效样本数: {len(df)}（已排除 Eff≤0）")
    plot_all(df, args.out)


if __name__ == "__main__":
    main()
