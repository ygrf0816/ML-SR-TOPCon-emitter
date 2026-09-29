"""
从 Athena 结构 CSV 中提取点缺陷曲线与统计特征，并生成可选的器件缺陷映射表。

用途：
1) 导出可用于后续建模的缺陷曲线长表（Traps/Vacancies/Interstitials vs depth）
2) 计算每条样本的缺陷统计特征（峰值、积分、前表面平均等）
3) 生成用于 cd_txt_scan.py 的缺陷参数映射 CSV（DEFECT_MAP_CSV）

默认目录（可改）：
- ATHENA_CSV_DIR: 项目根/saomiao3csv
- OUT_DIR: 项目根/defect_exports
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _detect_project_root() -> Path:
    """
    自动探测项目根目录。
    优先使用环境变量 PROJECT_ROOT；否则从当前文件向上查找包含 TOPCon/pycode 的目录。
    """
    env_root = os.environ.get("PROJECT_ROOT", "").strip()
    if env_root:
        p = Path(env_root).resolve()
        if p.exists():
            return p

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "TOPCon" / "pycode").is_dir():
            return parent

    # 兜底：按当前文件位于 <root>/TOPCon/pycode 的常见结构回退
    return here.parents[2]


PROJECT_ROOT = _detect_project_root()
ATHENA_CSV_DIR = Path(os.environ.get("ATHENA_CSV_DIR", str(PROJECT_ROOT / "saomiao3csv"))).resolve()
OUT_DIR = Path(os.environ.get("DEFECT_OUT_DIR", str(PROJECT_ROOT / "defect_exports"))).resolve()

# 前表面窗口（um），用于提取表面损伤相关统计
FRONT_WINDOW_UM = 0.02


def _find_col(cols: list[str], keywords: Tuple[str, ...]) -> Optional[str]:
    low_to_raw = {c.lower(): c for c in cols}
    for low, raw in low_to_raw.items():
        if all(k in low for k in keywords):
            return raw
    return None


def _read_athena_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    return df


def _safe_trapz(x: np.ndarray, y: np.ndarray) -> float:
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))


def _normalize(v: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    z = (v - vmin) / (vmax - vmin)
    return max(0.0, min(1.0, z))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted([p for p in ATHENA_CSV_DIR.glob("*.csv") if p.is_file()])
    if not csv_files:
        print(f"[WARN] 未找到 Athena CSV: {ATHENA_CSV_DIR}")
        return

    long_rows = []
    feat_rows = []

    for path in csv_files:
        df = _read_athena_csv(path)
        if df is None:
            continue

        cols = [str(c).strip() for c in df.columns]
        x_col = _find_col(cols, ("x",))
        traps_col = _find_col(cols, ("traps",))
        vac_col = _find_col(cols, ("vacancies",))
        int_col = _find_col(cols, ("interstitials",))
        boron_col = _find_col(cols, ("boron",))

        if x_col is None or traps_col is None or vac_col is None or int_col is None:
            continue

        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        traps = pd.to_numeric(df[traps_col], errors="coerce").to_numpy(dtype=float)
        vac = pd.to_numeric(df[vac_col], errors="coerce").to_numpy(dtype=float)
        inter = pd.to_numeric(df[int_col], errors="coerce").to_numpy(dtype=float)
        boron = (
            pd.to_numeric(df[boron_col], errors="coerce").to_numpy(dtype=float)
            if boron_col is not None
            else np.full_like(x, np.nan)
        )

        valid = np.isfinite(x) & np.isfinite(traps) & np.isfinite(vac) & np.isfinite(inter)
        if np.sum(valid) < 2:
            continue

        x = x[valid]
        traps = traps[valid]
        vac = vac[valid]
        inter = inter[valid]
        boron = boron[valid] if boron.shape == x.shape else np.full_like(x, np.nan)

        order = np.argsort(x)
        x = x[order]
        traps = traps[order]
        vac = vac[order]
        inter = inter[order]
        boron = boron[order]

        file_base = path.stem
        front_mask = x <= FRONT_WINDOW_UM
        if not np.any(front_mask):
            front_mask = x <= np.nanpercentile(x, 5)

        traps_front = float(np.nanmean(traps[front_mask]))
        vac_front = float(np.nanmean(vac[front_mask]))
        inter_front = float(np.nanmean(inter[front_mask]))

        feat_rows.append(
            {
                "file_base": file_base,
                "traps_peak": float(np.nanmax(traps)),
                "traps_mean": float(np.nanmean(traps)),
                "traps_front_mean": traps_front,
                "traps_integral": _safe_trapz(x, traps),
                "vacancies_peak": float(np.nanmax(vac)),
                "vacancies_mean": float(np.nanmean(vac)),
                "vacancies_front_mean": vac_front,
                "vacancies_integral": _safe_trapz(x, vac),
                "interstitials_peak": float(np.nanmax(inter)),
                "interstitials_mean": float(np.nanmean(inter)),
                "interstitials_front_mean": inter_front,
                "interstitials_integral": _safe_trapz(x, inter),
                "x_min": float(np.nanmin(x)),
                "x_max": float(np.nanmax(x)),
                "n_points": int(len(x)),
            }
        )

        for xi, ti, vi, ii, bi in zip(x, traps, vac, inter, boron):
            long_rows.append(
                {
                    "file_base": file_base,
                    "x_um": float(xi),
                    "traps_cm3": float(ti),
                    "vacancies_cm3": float(vi),
                    "interstitials_cm3": float(ii),
                    "boron_cm3": float(bi) if np.isfinite(bi) else np.nan,
                }
            )

    if not feat_rows:
        print("[WARN] 未提取到有效缺陷特征。")
        return

    feat_df = pd.DataFrame(feat_rows)
    long_df = pd.DataFrame(long_rows)
    feat_df.to_csv(OUT_DIR / "athena_defect_features.csv", index=False)
    long_df.to_csv(OUT_DIR / "athena_defect_curves_long.csv", index=False)

    # 生成用于器件 IV 的缺陷参数映射（经验映射，可按实验再标定）
    tmin, tmax = float(feat_df["traps_front_mean"].min()), float(feat_df["traps_front_mean"].max())
    vmin, vmax = float(feat_df["vacancies_front_mean"].min()), float(feat_df["vacancies_front_mean"].max())
    imin, imax = float(feat_df["interstitials_front_mean"].min()), float(feat_df["interstitials_front_mean"].max())

    map_rows = []
    for _, r in feat_df.iterrows():
        ntop_scale = _normalize(float(r["traps_front_mean"]), tmin, tmax)
        nbulk_scale = _normalize(float(r["vacancies_front_mean"]), vmin, vmax)
        life_scale = _normalize(float(r["interstitials_front_mean"]), imin, imax)

        map_rows.append(
            {
                "file_base": r["file_base"],
                # 界面态：1e10 ~ 5e12
                "Nt_top_N": 1e10 * (10 ** (ntop_scale * 2.7)),
                # 体缺陷：1e14 ~ 3e15
                "Nt_polySi_top": 1e14 * (10 ** (nbulk_scale * 1.5)),
                "Nt_polySi_rear": 1e14 * (10 ** (nbulk_scale * 1.2)),
                # 背面界面态：1e13 ~ 3e13
                "Nt_Si_SiOx": 1e13 * (1.0 + 2.0 * ntop_scale),
                "Nt_SiOx_Poly": 1e13 * (1.0 + 2.0 * ntop_scale),
                # 寿命：插值到 [1e-3, 1e-1]，缺陷越重寿命越低
                "taun_Si": 1e-1 - (1e-1 - 1e-3) * life_scale,
            }
        )

    map_df = pd.DataFrame(map_rows)
    map_df.to_csv(OUT_DIR / "defect_map_for_iv.csv", index=False)

    print(f"[OK] 特征表: {OUT_DIR / 'athena_defect_features.csv'}")
    print(f"[OK] 曲线表: {OUT_DIR / 'athena_defect_curves_long.csv'}")
    print(f"[OK] IV映射: {OUT_DIR / 'defect_map_for_iv.csv'}")
    print("后续可设置环境变量 DEFECT_MAP_CSV 指向该 IV 映射文件后运行 cd_txt_scan.py")


if __name__ == "__main__":
    main()
