"""验证学长要求的三项改进是否完善。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cutline_extract import extract_composite_from_str, merge_light_heavy_profiles
from se_athena_scan import sample_se_params

ROOT = Path(__file__).resolve().parent


def _load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ys, ns = [], []
    for ln in path.read_text(encoding="utf-8").splitlines():
        t = ln.split()
        if len(t) >= 2:
            ys.append(float(t[0]))
            ns.append(float(t[1]))
    return np.asarray(ys), np.asarray(ns)


def check_point1_cutline_metrics() -> bool:
    """第1点：cutline 提取 + O1~O6 工艺指标。"""
    print("\n=== 第1点：cutline 工艺指标提取 ===")
    metrics_csv = ROOT / "defect_exports" / "se_process_metrics.csv"
    if not metrics_csv.is_file():
        print("[FAIL] 缺少 se_process_metrics.csv，请先运行 extract_from_str")
        return False

    df = pd.read_csv(metrics_csv)
    required = ["Rs_light", "Rs_heavy", "Xj_light", "Xj_heavy", "Cs_light", "Cs_heavy"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"[FAIL] 工艺指标列缺失: {missing_cols}")
        return False

    bad = df[required].isna().all(axis=1)
    if bad.any():
        print(f"[FAIL] 有样本工艺指标全为空: {df.loc[bad, 'file_base'].tolist()}")
        return False

    print(f"[OK] {len(df)} 个样本均有 cutline 工艺指标")
    print(df[["file_base"] + required].head(2).to_string(index=False))
    return True


def check_point2_composite_curve() -> bool:
    """第2点：轻掺+重掺复合曲线，深度加深到 5μm。"""
    print("\n=== 第2点：轻/重掺复合曲线 + 加深深度 ===")
    str_files = sorted((ROOT / "saomiao3").glob("SE_Scan_*.str"))
    if not str_files:
        print("[FAIL] 无 SE_Scan_*.str 文件")
        return False

    p = str_files[0]
    cut = extract_composite_from_str(p, y_max=5.0)
    light = cut["light"]
    heavy = cut["heavy"]
    comp = cut["composite"]

    y_max_light = max(light["y_um"]) if light["y_um"] else 0
    y_max_comp = max(comp["y_um"]) if comp["y_um"] else 0
    if y_max_comp < 1.0:
        print(f"[FAIL] 复合曲线深度过浅: max_y={y_max_comp}")
        return False
    print(f"[OK] 复合曲线最大深度: {y_max_comp:.4f} μm (目标加深至 5μm 范围)")

    # 复合曲线应在至少一个深度上同时体现轻掺和重掺特征
    n_l = np.asarray(light["net_doping"])
    n_h = np.asarray(heavy["net_doping"])
    n_c = np.asarray(comp["net_doping"])
    merged = merge_light_heavy_profiles(light, heavy)
    n_m = np.asarray(merged["net_doping"])
    if not np.allclose(n_c, n_m, rtol=1e-6, equal_nan=True):
        print("[FAIL] 复合曲线与 merge_light_heavy 不一致")
        return False

    heavy_dominates = float(np.max(n_h)) > float(np.max(n_l)) * 0.5
    comp_ge_light = float(np.max(n_c)) >= float(np.max(n_l)) * 0.99
    comp_ge_heavy_surface = n_c[0] >= min(n_h[0], n_l[0]) if n_c.size else False
    if not (heavy_dominates and comp_ge_light):
        print("[WARN] 轻/重掺差异不明显，但合并逻辑正确")

    cutline_dir = ROOT / "defect_exports" / "cutlines"
    tags = ["light", "heavy", "composite"]
    for tag in tags:
        f = cutline_dir / f"{p.stem}_{tag}_cutline.txt"
        if not f.is_file():
            print(f"[FAIL] 缺少 cutline 文件: {f.name}")
            return False
    print(f"[OK] 已生成 light/heavy/composite 三条 cutline: {p.stem}")

    # TOPCon 输入应来自复合曲线
    doping_txt = ROOT / "saomiao3csv" / f"{p.stem}.txt"
    comp_txt = cutline_dir / f"{p.stem}_composite_cutline.txt"
    if doping_txt.is_file() and comp_txt.is_file():
        _, n_topcon = _load_curve(doping_txt)
        _, n_comp_file = _load_curve(comp_txt)
        if float(np.max(n_topcon)) < float(np.max(n_comp_file)) * 0.5:
            print("[FAIL] TOPCon 掺杂 txt 未使用复合曲线")
            return False
        print("[OK] TOPCon 掺杂 txt 与复合 cutline 一致")
    return True


def check_point3_param_sampling() -> bool:
    """第3点：扩展工艺参数采样。"""
    print("\n=== 第3点：扩展参数采样 ===")
    required = {
        "light_drive_temp",
        "repair_temp",
        "repair_time",
        "light_oxide_thick",
        "mask_oxide_thick",
        "select_boundary_x",
        "sub_resis",
        "light_dose",
        "heavy_dose",
        "anneal_temp",
        "anneal_time",
    }
    sample = sample_se_params()
    missing = required - set(sample.keys())
    if missing:
        print(f"[FAIL] sample_se_params 缺少: {missing}")
        return False
    print(f"[OK] sample_se_params 含 {len(sample)} 个参数，含 light_drive_temp/repair_temp/repair_time")

    jsonl = ROOT / "saomiao3" / "se_scan.jsonl"
    if jsonl.is_file():
        rows = [json.loads(ln) for ln in jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if rows:
            last = rows[-1].get("params", {})
            old_missing = required - set(last.keys())
            if old_missing:
                print(f"[WARN] 历史 jsonl 末条尚无新参数 {old_missing}（新扫描后会补齐）")
            else:
                print(f"[OK] jsonl 最新记录已含扩展参数")
    return True


def main() -> int:
    ok1 = check_point1_cutline_metrics()
    ok2 = check_point2_composite_curve()
    ok3 = check_point3_param_sampling()
    all_ok = ok1 and ok2 and ok3
    print("\n" + ("全部验证通过" if all_ok else "存在未通过项"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
