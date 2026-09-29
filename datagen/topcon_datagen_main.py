"""
topcon_datagen_main.py

统一主入口（IDE 一键运行）：
1) Athena 工艺扫描（可选）
2) 从 Athena .str 批量提取掺杂曲线与缺陷曲线/特征
3) 批量运行 TOPCon IV（掺杂 + 前部缺陷参数）
4) 汇总成训练/分析数据集

使用方式：
- 直接运行本文件
- 在下方 `RUN_MODE` 选择模式：
  - "athena_scan_only"
  - "extract_from_str"
  - "run_iv"
  - "build_dataset"
  - "full_pipeline"
"""

from __future__ import annotations

import json
import math
import os
from time import perf_counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================
# 可配置区（主入口设置）
# =========================
# 运行模式说明（把下面 RUN_MODE 改成对应值即可）：
# 1) 从零开始（没有现成 .str）：
#    RUN_MODE = "from_scratch"
#    -> 先跑 Athena 扫描，再自动执行提取 -> IV -> 数据集
#
# 2) Athena .str 已经跑完，只做后处理与器件计算（推荐常用）：
#    RUN_MODE = "full_pipeline"   # 等价于 extract_from_str + run_iv + build_dataset
#
# 3) 分步运行：
#    RUN_MODE = "extract_from_str"  # 仅从 .str 提取掺杂/缺陷及参数
#    RUN_MODE = "run_iv"            # 仅批量算 IV（读取 saomiao3csv + defect_map_for_iv.csv）
#    RUN_MODE = "build_dataset"     # 仅汇总成 dataset/topcon_dataset.csv
#
# 4) 仅 Athena 扫描：
#    RUN_MODE = "athena_scan_only"
#
# 你也可以不改代码，直接用环境变量：
#   DATAGEN_MODE=full_pipeline
#   DATAGEN_MODE=from_scratch
RUN_MODE = os.environ.get("DATAGEN_MODE", "from_scratch").strip().lower()

# 项目根目录 = se-TOPCon 文件夹本身
PROJECT_ROOT = Path(__file__).resolve().parent
ATHENA_STR_DIR = PROJECT_ROOT / "saomiao3"
ATHENA_JSONL = ATHENA_STR_DIR / "se_scan.jsonl"

DOPING_TXT_DIR = PROJECT_ROOT / "saomiao3csv"
IV_OUT_DIR = PROJECT_ROOT / "saomiao3jv"
DEFECT_EXPORT_DIR = PROJECT_ROOT / "defect_exports"
DATASET_DIR = PROJECT_ROOT / "dataset"

TCAD_EXE_DIR = os.environ.get("TCAD_EXE_DIR", r"D:\silvaco\exe")

# 缺陷输入模式：
# - "mapped": 维持当前积分映射法（Nt_top_N / Nt_polySi_top）
# - "trap_curve": 使用 DOPING TRAP 导入外部缺陷曲线（前表面）
DEFECT_INPUT_MODE = os.environ.get("DEFECT_INPUT_MODE", "trap_curve").strip().lower()

# Athena 扫描（仅 athena_scan_only 模式使用）
ATHENA_SCAN_RUNS = int(os.environ.get("ATHENA_SCAN_RUNS", "10000"))
ATHENA_SCAN_PARALLEL = os.environ.get("ATHENA_SCAN_PARALLEL", "0").strip() in ("1", "true", "yes")
ATHENA_SCAN_WORKERS = int(os.environ.get("ATHENA_SCAN_WORKERS", "2"))

# IV 运行设置
FORCE_RERUN_IV = os.environ.get("FORCE_RERUN_IV", "0").strip() in ("1", "true", "yes")
MAX_FILES: int | None = None  # None 表示全部；调试时可设为小整数
# 掺杂曲线下限（衬底背景浓度）：提取时将 n<该值 的点抬到该值
DOPING_SUBSTRATE_FLOOR = float(os.environ.get("DOPING_SUBSTRATE_FLOOR", "2e15"))
# cutline 提取深度范围（μm），默认加深到 5.0 以覆盖更深扩散尾部
CUTLINE_Y_MIN = float(os.environ.get("CUTLINE_Y_MIN", "0.0"))
CUTLINE_Y_MAX = float(os.environ.get("CUTLINE_Y_MAX", "5.0"))
# 轻掺 / 重掺 cutline 位置（与 se.in 中 x_light_extract / x_heavy_extract 一致）
CUTLINE_X_LIGHT = float(os.environ.get("CUTLINE_X_LIGHT", "50.0"))
CUTLINE_X_HEAVY = float(os.environ.get("CUTLINE_X_HEAVY", "7.5"))


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _safe_trapz(x: np.ndarray, y: np.ndarray) -> float:
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _normalize_cutline_y(y: np.ndarray) -> np.ndarray:
    """将原始 y 轴平移：把 BSG 起点(最小 y)映射到 0。"""
    if y.size == 0:
        return y
    valid = y[np.isfinite(y)]
    if valid.size == 0:
        return y
    y0 = float(np.min(valid))
    return y - y0


def _replace_one_with_nearby_max(arr: np.ndarray, window: int = 10) -> np.ndarray:
    """
    缺陷曲线修复：
    - 将数值为 1（含浮点近似）的点替换为“附近窗口内最大缺陷值”
    - 若附近无可用值，则使用全局最大非 1 值
    """
    out = np.asarray(arr, dtype=float).copy()
    if out.size == 0:
        return out

    finite = np.isfinite(out)
    is_one = finite & np.isclose(out, 1.0, rtol=0.0, atol=1e-12)
    valid_non_one = finite & (~is_one)
    if not np.any(is_one):
        return out
    if not np.any(valid_non_one):
        return out

    global_max = float(np.nanmax(np.abs(out[valid_non_one])))
    idxs = np.where(is_one)[0]
    n = len(out)
    w = max(1, int(window))
    for i in idxs:
        l = max(0, i - w)
        r = min(n, i + w + 1)
        local = out[l:r]
        local_finite = np.isfinite(local)
        local_is_one = local_finite & np.isclose(local, 1.0, rtol=0.0, atol=1e-12)
        local_valid = local_finite & (~local_is_one)
        if np.any(local_valid):
            out[i] = float(np.nanmax(np.abs(local[local_valid])))
        else:
            out[i] = global_max
    return out


def _build_doping_curve(y: List[float], conc: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(y, dtype=float)
    n = np.asarray(conc, dtype=float)
    x = _normalize_cutline_y(x)
    mask = (
        np.isfinite(x)
        & np.isfinite(n)
        & (x >= CUTLINE_Y_MIN)
        & (x <= CUTLINE_Y_MAX)
        & (n > 0)
    )
    x = x[mask]
    n = n[mask]
    if x.size == 0:
        return x, n
    # 低于衬底背景浓度的掺杂值按下限处理，便于后续建模/学习使用
    n = np.maximum(n, DOPING_SUBSTRATE_FLOOR)
    order = np.argsort(x)
    x = x[order]
    n = n[order]
    # 去重：x 重复时取该点最大浓度
    ux = []
    un = []
    i = 0
    while i < len(x):
        j = i + 1
        vmax = n[i]
        while j < len(x) and x[j] == x[i]:
            vmax = max(vmax, n[j])
            j += 1
        ux.append(x[i])
        un.append(vmax)
        i = j
    return np.asarray(ux), np.asarray(un)


def _doping_descriptors(x: np.ndarray, n: np.ndarray) -> Dict[str, float]:
    if x.size < 2:
        return {
            "N_peak": float("nan"),
            "x_peak": float("nan"),
            "junction_depth": float("nan"),
            "FWHM": float("nan"),
            "gradient_max": float("nan"),
            "dose": float("nan"),
            "n_points": int(x.size),
        }
    idx_peak = int(np.nanargmax(n))
    n_peak = float(n[idx_peak])
    x_peak = float(x[idx_peak])
    mask = n > 1e16
    junction_depth = float(np.max(x[mask])) if np.any(mask) else float("nan")
    half = 0.5 * n_peak
    ids = np.where(n >= half)[0]
    fwhm = float(x[ids[-1]] - x[ids[0]]) if ids.size >= 2 else float("nan")
    grad = np.gradient(n, x)
    gradient_max = float(np.nanmax(np.abs(grad)))
    dose = _safe_trapz(x, n)
    return {
        "N_peak": n_peak,
        "x_peak": x_peak,
        "junction_depth": junction_depth,
        "FWHM": fwhm,
        "gradient_max": gradient_max,
        "dose": dose,
        "n_points": int(x.size),
    }


def extract_from_str_batch() -> None:
    t0 = perf_counter()
    DOPING_TXT_DIR.mkdir(parents=True, exist_ok=True)
    DEFECT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    str_files = sorted(ATHENA_STR_DIR.glob("*.str"))
    if MAX_FILES is not None:
        str_files = str_files[: max(0, int(MAX_FILES))]
    if not str_files:
        print(f"[WARN] 未找到 .str 文件: {ATHENA_STR_DIR}")
        return

    from cutline_extract import extract_composite_from_str

    defect_long_rows: List[Dict[str, Any]] = []
    defect_feat_rows: List[Dict[str, Any]] = []
    defect_map_rows: List[Dict[str, Any]] = []
    doping_desc_rows: List[Dict[str, Any]] = []
    process_metric_rows: List[Dict[str, Any]] = []
    trap_curve_dir = DEFECT_EXPORT_DIR / "trap_curves"
    cutline_dir = DEFECT_EXPORT_DIR / "cutlines"
    trap_curve_dir.mkdir(parents=True, exist_ok=True)
    cutline_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(str_files, desc="Extract Cutline", unit="file"):
        base = p.stem
        try:
            cut = extract_composite_from_str(
                p,
                x_light=CUTLINE_X_LIGHT,
                x_heavy=CUTLINE_X_HEAVY,
                y_min=CUTLINE_Y_MIN,
                y_max=CUTLINE_Y_MAX,
            )
        except Exception as e:
            print(f"[WARN] cutline 提取失败: {p.name} -> {e}")
            continue

        composite = cut["composite"]
        conc = composite.get("net_doping", [])
        x, n = _build_doping_curve(composite.get("y_um", []), conc)
        if x.size == 0:
            continue

        # 保存轻/重/复合 cutline 便于检查
        for tag, profile in (("light", cut["light"]), ("heavy", cut["heavy"]), ("composite", composite)):
            out_cut = cutline_dir / f"{base}_{tag}_cutline.txt"
            with out_cut.open("w", encoding="utf-8", newline="") as f:
                for yy, nn in zip(profile.get("y_um", []), profile.get("net_doping", [])):
                    f.write(f"{float(yy):.15g}\t{float(nn):.8e}\n")

        proc_row = {"file_base": base, **cut["metrics"]}
        process_metric_rows.append(proc_row)

        # 写 TOPCon 用复合掺杂 txt
        doping_txt = DOPING_TXT_DIR / f"{base}.txt"
        with doping_txt.open("w", encoding="utf-8", newline="") as f:
            for xx, nn in zip(x, n):
                f.write(f"{xx:.15g}\t{nn:.8e}\n")

        desc = _doping_descriptors(x, n)
        desc["file_base"] = base
        desc["doping_txt"] = str(doping_txt)
        doping_desc_rows.append(desc)

        # 缺陷长表 + 特征（基于复合 cutline）
        y_raw = np.asarray(composite.get("y_um", []), dtype=float)
        y = _normalize_cutline_y(y_raw)
        vac = np.asarray(composite.get("vacancies", []), dtype=float)
        inter = np.asarray(composite.get("interstitials", []), dtype=float)
        traps = np.asarray(composite.get("traps", []), dtype=float)
        phos = np.asarray(composite.get("phosphorus", []), dtype=float)
        m = (
            np.isfinite(y)
            & (y >= CUTLINE_Y_MIN)
            & (y <= CUTLINE_Y_MAX)
        )
        y, vac, inter, traps, phos = y[m], vac[m], inter[m], traps[m], phos[m]

        order = np.argsort(y)
        y, vac, inter, traps, phos = (
            y[order],
            vac[order],
            inter[order],
            traps[order],
            phos[order],
        )
        # 修复 BSG 区缺陷“占位值=1”：替换为附近最大缺陷值
        vac = _replace_one_with_nearby_max(vac)
        inter = _replace_one_with_nearby_max(inter)
        traps = _replace_one_with_nearby_max(traps)

        for yy, vv, ii, tt, pp in zip(y, vac, inter, traps, phos):
            defect_long_rows.append(
                {
                    "file_base": base,
                    "y_um": float(yy),
                    "phosphorus_cm3": float(pp),
                    "vacancies_0_cm3": float(vv),
                    "interstitials_1_cm3": float(ii),
                    "traps_12": float(tt),
                }
            )

        front = (y <= 0.02)
        if not np.any(front):
            front = np.zeros_like(y, dtype=bool)
            front[: min(10, len(y))] = True

        vac_front = float(np.nanmean(np.abs(vac[front])))
        inter_front = float(np.nanmean(np.abs(inter[front])))
        traps_front = float(np.nanmean(np.abs(traps[front])))

        defect_feat_rows.append(
            {
                "file_base": base,
                "vac_front_mean": vac_front,
                "inter_front_mean": inter_front,
                "traps_front_mean": traps_front,
                "vac_peak": float(np.nanmax(np.abs(vac))) if vac.size else float("nan"),
                "inter_peak": float(np.nanmax(np.abs(inter))) if inter.size else float("nan"),
                "n_points": int(len(y)),
            }
        )

        # 前部可用版：仅前部参数映射
        nt_top_n = _clip(vac_front * 1e-4, 1e10, 5e12)
        nt_poly_top = _clip(inter_front * 1e1, 1e13, 5e15)
        # 导出可直接给 DOPING TRAP 使用的缺陷曲线（默认采用 vacancies 列）
        # 两列格式：depth(um), trap_density(cm^-3)
        trap_txt = trap_curve_dir / f"{base}_trap_vacancies.txt"
        with trap_txt.open("w", encoding="utf-8", newline="") as f:
            for yy, vv in zip(y, vac):
                if np.isfinite(yy) and np.isfinite(vv) and yy >= 0:
                    f.write(f"{float(yy):.15g}\t{abs(float(vv)):.8e}\n")

        rec = {
            "file_base": base,
            "Nt_top_N": nt_top_n,
            "Nt_polySi_top": nt_poly_top,
            "defect_mode": DEFECT_INPUT_MODE,
            "trap_cdpath": str(trap_txt),
        }
        defect_map_rows.append(rec)

    if process_metric_rows:
        pd.DataFrame(process_metric_rows).to_csv(DEFECT_EXPORT_DIR / "se_process_metrics.csv", index=False)
    pd.DataFrame(doping_desc_rows).to_csv(DEFECT_EXPORT_DIR / "athena_doping_curve_descriptors.csv", index=False)
    pd.DataFrame(defect_long_rows).to_csv(DEFECT_EXPORT_DIR / "athena_defect_curve_long.csv", index=False)
    pd.DataFrame(defect_feat_rows).to_csv(DEFECT_EXPORT_DIR / "athena_defect_feature_summary.csv", index=False)
    pd.DataFrame(defect_map_rows).to_csv(DEFECT_EXPORT_DIR / "defect_map_for_iv.csv", index=False)

    dt = perf_counter() - t0
    print(f"[OK] 提取完成，样本数: {len(doping_desc_rows)}")
    print(f"[OK] 掺杂txt目录: {DOPING_TXT_DIR}")
    print(f"[OK] 缺陷导出目录: {DEFECT_EXPORT_DIR}")
    print(f"[TIME] extract_from_str: {dt:.2f}s, {len(str_files)/dt:.2f} file/s" if dt > 0 else "[TIME] extract_from_str: <1ms")


def run_iv_batch() -> None:
    from cd_txt_scan import list_txt_files, load_defect_map, run_one_with_cd

    t0 = perf_counter()
    IV_OUT_DIR.mkdir(parents=True, exist_ok=True)
    txt_files = list_txt_files(str(DOPING_TXT_DIR))
    if MAX_FILES is not None:
        txt_files = txt_files[: max(0, int(MAX_FILES))]
    if not txt_files:
        print(f"[WARN] 无掺杂 txt 文件: {DOPING_TXT_DIR}")
        return

    defect_map_file = DEFECT_EXPORT_DIR / "defect_map_for_iv.csv"
    defect_map = load_defect_map(str(defect_map_file)) if defect_map_file.is_file() else {}
    print(f"[INFO] 缺陷映射样本数: {len(defect_map)}")

    run_rows: List[Dict[str, Any]] = []
    for i, txt in enumerate(tqdm(txt_files, desc="Run IV", unit="file"), 1):
        base = Path(txt).stem
        out_csv = IV_OUT_DIR / f"{base}.csv"
        if out_csv.is_file() and not FORCE_RERUN_IV:
            print(f"[{i}/{len(txt_files)}] skip: {base}")
            continue
        defects = defect_map.get(base.lower())
        print(f"[{i}/{len(txt_files)}] run:  {base}  defects={'yes' if defects else 'no'}")
        rec = run_one_with_cd(
            txt_path=txt,
            exe_dir=TCAD_EXE_DIR,
            out_dir=str(IV_OUT_DIR),
            defect_values=defects,
        )
        status = str(rec.get("status", "error"))
        if status == "ok":
            run_rows.append(rec)
        elif status.startswith("timeout"):
            print(f"  timeout skip: {status}")
        else:
            print(f"  failed: {status}")

    if run_rows:
        out_jsonl = IV_OUT_DIR / "iv_run_records.jsonl"
        with out_jsonl.open("a", encoding="utf-8") as f:
            for r in run_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[OK] IV 运行记录追加到: {out_jsonl}")
    else:
        print("[INFO] 没有新的 IV 任务执行。")
    dt = perf_counter() - t0
    print(f"[TIME] run_iv: {dt:.2f}s, {len(txt_files)/dt:.2f} file/s(含skip)" if dt > 0 else "[TIME] run_iv: <1ms")


def build_dataset() -> None:
    from analyze_scans import compute_metrics, read_iv_from_csv

    t0 = perf_counter()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Athena 参数与工艺指标（来自 SE 扫描 jsonl）
    athena_rows = _load_jsonl(ATHENA_JSONL)
    athena_map: Dict[str, Dict[str, Any]] = {}
    metrics_map: Dict[str, Dict[str, Any]] = {}
    for r in athena_rows:
        fb = str(r.get("file_base", "")).strip()
        if not fb:
            continue
        key = fb.lower()
        athena_map[key] = r.get("params", {}) or {}
        metrics_map[key] = r.get("metrics", {}) or {}

    # cutline 计算的工艺指标（优先于 jsonl 中空缺的 process 字段）
    proc_p = DEFECT_EXPORT_DIR / "se_process_metrics.csv"
    if proc_p.is_file():
        proc_df = pd.read_csv(proc_p)
        for _, r in proc_df.iterrows():
            fb = str(r.get("file_base", "")).strip().lower()
            if not fb:
                continue
            metrics_map[fb] = {**metrics_map.get(fb, {}), **{k: r[k] for k in r.index if k != "file_base"}}

    # 额外特征
    desc_df = pd.DataFrame()
    feat_df = pd.DataFrame()
    map_df = pd.DataFrame()
    p = DEFECT_EXPORT_DIR / "athena_doping_curve_descriptors.csv"
    if p.is_file():
        desc_df = pd.read_csv(p)
    p = DEFECT_EXPORT_DIR / "athena_defect_feature_summary.csv"
    if p.is_file():
        feat_df = pd.read_csv(p)
    p = DEFECT_EXPORT_DIR / "defect_map_for_iv.csv"
    if p.is_file():
        map_df = pd.read_csv(p)

    desc_map = {str(r["file_base"]).lower(): dict(r) for _, r in desc_df.iterrows()} if not desc_df.empty else {}
    feat_map = {str(r["file_base"]).lower(): dict(r) for _, r in feat_df.iterrows()} if not feat_df.empty else {}
    dmap = {str(r["file_base"]).lower(): dict(r) for _, r in map_df.iterrows()} if not map_df.empty else {}

    rows: List[Dict[str, Any]] = []
    iv_files = sorted(IV_OUT_DIR.glob("*.csv"))
    if MAX_FILES is not None:
        iv_files = iv_files[: max(0, int(MAX_FILES))]
    for p in tqdm(iv_files, desc="Build Dataset", unit="file"):
        base = p.stem
        try:
            v, i = read_iv_from_csv(str(p))
            met = compute_metrics(v, i)
        except Exception as e:
            print(f"[WARN] 指标计算失败: {p.name} -> {e}")
            continue

        rec: Dict[str, Any] = {"file_base": base, "iv_csv": str(p)}
        rec.update({f"iv_{k}": float(vv) for k, vv in met.items()})
        for k, vv in athena_map.get(base.lower(), {}).items():
            rec[f"athena_{k}"] = vv
        for k, vv in metrics_map.get(base.lower(), {}).items():
            rec[f"process_{k}"] = vv
        for k, vv in desc_map.get(base.lower(), {}).items():
            if k != "file_base":
                rec[f"doping_{k}"] = vv
        for k, vv in feat_map.get(base.lower(), {}).items():
            if k != "file_base":
                rec[f"defect_{k}"] = vv
        for k, vv in dmap.get(base.lower(), {}).items():
            if k != "file_base":
                rec[f"map_{k}"] = vv
        rows.append(rec)

    out_csv = DATASET_DIR / "topcon_dataset.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    dt = perf_counter() - t0
    print(f"[OK] 数据集已输出: {out_csv}")
    print(f"[OK] 样本数: {len(rows)}")
    print(f"[TIME] build_dataset: {dt:.2f}s, {len(iv_files)/dt:.2f} file/s" if dt > 0 else "[TIME] build_dataset: <1ms")


def run_athena_scan() -> None:
    from se_athena_scan import run_scan

    t0 = perf_counter()
    ATHENA_STR_DIR.mkdir(parents=True, exist_ok=True)
    print("[INFO] 开始 SE Athena 扫描（基于 se.in）...")
    run_scan(
        runs=ATHENA_SCAN_RUNS,
        out_dir=str(ATHENA_STR_DIR),
        exe_dir=TCAD_EXE_DIR,
        jsonl=str(ATHENA_JSONL),
        parallel=ATHENA_SCAN_PARALLEL,
        workers=ATHENA_SCAN_WORKERS,
    )
    dt = perf_counter() - t0
    print(f"[TIME] athena_scan: {dt:.2f}s, target_runs={ATHENA_SCAN_RUNS}")


def main() -> None:
    modes = {
        "athena_scan_only": run_athena_scan,
        "extract_from_str": extract_from_str_batch,
        "run_iv": run_iv_batch,
        "build_dataset": build_dataset,
        "from_scratch": None,  # 由下方特殊流程处理
    }

    if RUN_MODE == "from_scratch":
        # 从零开始：Athena -> 提取 -> IV -> 数据集
        run_athena_scan()
        extract_from_str_batch()
        run_iv_batch()
        build_dataset()
        return

    if RUN_MODE == "full_pipeline":
        # .str 已经存在时：执行后三步
        extract_from_str_batch()
        run_iv_batch()
        build_dataset()
        return

    fn = modes.get(RUN_MODE)
    if fn is None:
        raise ValueError(
            f"未知 RUN_MODE={RUN_MODE}，可选：{list(modes.keys()) + ['full_pipeline']}"
        )
    fn()


if __name__ == "__main__":
    main()

