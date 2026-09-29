"""
summarize_boron_scan.py

用途:
- 读取项目根下 `saomiao3/athena_boron_scan.jsonl`，根据其中的 `file_base`，
  到 `saomiao3csv` 中查找对应的 `.txt` 掺杂曲线文件，计算指定的五个指标：
  1) 峰值浓度
  2) 结深(>1e16 的最大 x 坐标)
  3) 半高宽(峰值一半处的两个 x 的差)
  4) 结面突变度(峰值梯度)
  5) 有效剂量(浓度对 x 的积分)
  并将 `jsonl` 参数与这些指标整合到一个 CSV 文件中。

使用:
- 直接在 IDE 中运行，不依赖命令行参数。如需修改路径，调整脚本内常量即可。

更新:
- 2025-09: 用 `np.trapezoid` 替代 `np.trapz`（对旧版 NumPy 回退），以消除弃用警告。
- 2025-09: 增加 `SimpleProfiler` 分段计时与可选 `cProfile` 汇总，打印耗时排行。
- 2025-09: 当 `OUT_CSV` 指向目录时，自动写为 `summary_boron_scan.csv` 防止权限错误。
- 2025-09: 为 `TXT_DIR` 构建一次性索引，避免每次查找都扫描目录。
"""

import json
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager
from time import perf_counter
import cProfile
import pstats
from bisect import bisect_left, bisect_right

# 路径常量(可按需修改)
JSONL_PATH = "D:\\code\\silvaco\\tcadproject\\saomiao3\\athena_boron_scan.jsonl"
TXT_DIR = "D:\\code\\silvaco\\tcadproject\\saomiao3csv"
#TXT_DIR = "D:\\code\\silvaco\\tcadproject\\test\\test"
OUT_CSV = "D:\\code\\silvaco\\tcadproject\\test"


class SimpleProfiler:
    """轻量级分段计时器：按标签累计耗时，并输出排行。"""

    def __init__(self) -> None:
        self._totals: Dict[str, float] = {}

    @contextmanager
    def time_block(self, label: str):
        start = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - start
            self._totals[label] = self._totals.get(label, 0.0) + dt

    def report(self, top_n: int = 20) -> None:
        if not self._totals:
            print("[Profiler] 无分段计时数据。")
            return
        total_time = sum(self._totals.values())
        print("\n分段计时(累计)排行：")
        print("标签	累计耗时(s)	占比")
        for label, t in sorted(self._totals.items(), key=lambda kv: kv[1], reverse=True)[:top_n]:
            ratio = (t / total_time * 100.0) if total_time > 0 else 0.0
            print(f"{label}\t{t:.4f}\t{ratio:.1f}%")


# 计时/分析配置
ENABLE_SECTION_PROFILING = False
ENABLE_CPROFILE = False
CPROFILE_SORT_KEY = "cumtime"  # 可选：'cumtime', 'tottime', 'calls'
CPROFILE_TOP_N = 20

# 全局分段计时器
PROF = SimpleProfiler()


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


class TxtIndex:
    """对 `search_dir` 下所有 `.txt` 文件建立一次性索引，并支持前缀查找。
    - 以文件名(无扩展、小写)作为键，值为完整路径。
    - 前缀查找采用二分范围定位；若存在完全匹配，优先返回；否则返回名称最短者。
    """

    def __init__(self, search_dir: str) -> None:
        self._entries: List[Tuple[str, str]] = []
        for root, _, files in os.walk(search_dir):
            for f in files:
                if not f.lower().endswith(".txt"):
                    continue
                name_noext = os.path.splitext(f)[0].lower()
                self._entries.append((name_noext, os.path.join(root, f)))
        self._entries.sort(key=lambda t: t[0])
        self._names: List[str] = [n for n, _ in self._entries]

    def find_best_path(self, file_base: str) -> Optional[str]:
        if not self._entries:
            return None
        prefix = (file_base or "").lower()
        left = bisect_left(self._names, prefix)
        right = bisect_right(self._names, prefix + "\uffff")
        if left >= right:
            return None
        for i in range(left, right):
            if self._names[i] == prefix:
                return self._entries[i][1]
        subset = self._entries[left:right]
        best = min(subset, key=lambda t: len(os.path.basename(t[1])))
        return best[1]


def find_txt_for_filebase(file_base: str, search_dir: str) -> Optional[str]:
    """在 `search_dir` 下查找以 file_base 开头的 .txt 文件。返回最匹配的路径或 None。"""
    prefix_lower = file_base.lower()
    candidates: List[str] = []
    for root, _, files in os.walk(search_dir):
        for f in files:
            if not f.lower().endswith(".txt"):
                continue
            name_noext = os.path.splitext(f)[0].lower()
            if name_noext.startswith(prefix_lower):
                candidates.append(os.path.join(root, f))
    if not candidates:
        return None
    # 若有多个，选择与 file_base 完全相等优先，否则取最短名称的
    exact = [p for p in candidates if os.path.splitext(os.path.basename(p))[0].lower() == prefix_lower]
    if exact:
        return exact[0]
    return sorted(candidates, key=lambda p: len(os.path.basename(p)))[0]


def read_txt_curve(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 txt 掺杂曲线，假设两列: x 和 N，任意空白分隔。返回 (x, N)。"""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    x = np.asarray(data[:, 0], dtype=float)
    N = np.asarray(data[:, 1], dtype=float)
    # 去重并按 x 升序
    order = np.argsort(x)
    x = x[order]
    N = N[order]
    return x, N


def interp_x_at_y(x: np.ndarray, y: np.ndarray, y0: float) -> List[float]:
    """在线性插值下求解 y(x)=y0 的 x。返回所有交点列表。"""
    xs: List[float] = []
    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i + 1]
        if (y1 - y0) == 0:
            xs.append(float(x[i]))
            continue
        if (y1 - y0) * (y2 - y0) <= 0 and y2 != y1:
            t = (y0 - y1) / (y2 - y1)
            xi = float(x[i] + t * (x[i + 1] - x[i]))
            xs.append(xi)
    return xs


def compute_curve_metrics(x: np.ndarray, N: np.ndarray) -> Dict[str, float]:
    """计算五个指标。"""
    # 1. 峰值浓度
    idx_peak = int(np.nanargmax(N))
    N_peak = float(N[idx_peak])
    x_peak = float(x[idx_peak])

    # 2. 结深: >1e16 的最大 x
    mask = N > 1e16
    x_jiedepth = float(np.max(x[mask])) if np.any(mask) else float("nan")


    half = 0.5 * N_peak
    ba = x[np.where(N > half)[0][-1]]

    # 4. 结面突变度: 峰值梯度
    dNdx = np.gradient(N, x)
    grad_max = abs(float(dNdx[idx_peak]))

    # 5. 有效剂量: 浓度的积分
    try:
        dose = float(np.trapezoid(N, x))  # 优先使用新 API
    except AttributeError:
        dose = float(np.trapz(N, x))      # 兼容旧版 NumPy

    return {
        "N_peak": N_peak,
        "x_peak": x_peak,
        "junction_depth": x_jiedepth,
        "FWHM": ba,
        "gradient_max": grad_max,
        "dose": dose,
    }


def build_summary() -> pd.DataFrame:
    if ENABLE_SECTION_PROFILING:
        with PROF.time_block("load_jsonl"):
            records = load_jsonl(JSONL_PATH)
        with PROF.time_block("build_txt_index"):
            txt_index = TxtIndex(TXT_DIR)
    else:
        records = load_jsonl(JSONL_PATH)
        txt_index = TxtIndex(TXT_DIR)
    rows: List[Dict] = []
    for rec in tqdm(records):
        file_base = rec.get("file_base", "")
        params = rec.get("params", {}) or {}
        if ENABLE_SECTION_PROFILING:
            with PROF.time_block("find_txt_for_filebase"):
                txt_path = txt_index.find_best_path(file_base)
        else:
            txt_path = txt_index.find_best_path(file_base)

        metrics: Dict[str, float] = {
            "N_peak": float("nan"),
            "x_peak": float("nan"),
            "junction_depth": float("nan"),
            "FWHM": float("nan"),
            "gradient_max": float("nan"),
            "dose": float("nan"),
        }

        if txt_path and os.path.isfile(txt_path):
            try:
                if ENABLE_SECTION_PROFILING:
                    with PROF.time_block("read_txt_curve"):
                        x, N = read_txt_curve(txt_path)
                    with PROF.time_block("compute_curve_metrics"):
                        metrics = compute_curve_metrics(x, N)
                else:
                    x, N = read_txt_curve(txt_path)
                    metrics = compute_curve_metrics(x, N)
            except Exception as e:
                pass

        row = {
            "file_base": file_base,
            "txt_path": txt_path or "",
            **{f"param_{k}": v for k, v in params.items()},
            **metrics,
        }
        rows.append(row)

    if ENABLE_SECTION_PROFILING:
        with PROF.time_block("build_dataframe"):
            df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(rows)
    return df


def _resolve_out_csv_path(path_spec: str) -> str:
    """将用户给定的输出规范解析为实际 CSV 文件路径。
    - 若为已存在的目录或无扩展名，则在其下生成 `summary_boron_scan.csv`。
    - 否则按文件路径使用，其父目录若不存在会被创建。
    """
    try:
        if os.path.isdir(path_spec) or os.path.splitext(path_spec)[1] == "":
            target_dir = path_spec
            os.makedirs(target_dir, exist_ok=True)
            return os.path.join(target_dir, "summary_boron_scan.csv")
        parent = os.path.dirname(path_spec) or "."
        os.makedirs(parent, exist_ok=True)
        return path_spec
    except Exception:
        parent = os.path.dirname(path_spec) or "."
        os.makedirs(parent, exist_ok=True)
        return path_spec


def main():
    prof: Optional[cProfile.Profile] = None
    if ENABLE_CPROFILE:
        prof = cProfile.Profile()
        prof.enable()

    if ENABLE_SECTION_PROFILING:
        with PROF.time_block("build_summary"):
            df = build_summary()
    else:
        df = build_summary()

    out_path = _resolve_out_csv_path(OUT_CSV)
    if ENABLE_SECTION_PROFILING:
        with PROF.time_block("write_csv"):
            df.to_csv(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"已输出: {out_path}")

    if ENABLE_CPROFILE and prof is not None:
        prof.disable()
        print("\n[cProfile 汇总]")
        pstats.Stats(prof).strip_dirs().sort_stats(CPROFILE_SORT_KEY).print_stats(CPROFILE_TOP_N)

    if ENABLE_SECTION_PROFILING:
        PROF.report()


if __name__ == "__main__":
    main()


