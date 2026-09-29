"""Curve reading, fitting, and resampling utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.config import (
    CURVE_NUM_POINTS,
    CURVE_SMOOTH_METHOD,
    CURVE_TRUNCATION_DEPTH_MARGIN,
    CURVE_TRUNCATION_DROP_RATIO,
    DEPTH_MAX,
)


def read_curve_txt(txt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    depths: list[float] = []
    values: list[float] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", ".").split()
            if len(parts) < 2:
                continue
            try:
                d = float(parts[0])
                v = float(parts[1].replace("E", "e"))
            except ValueError:
                continue
            depths.append(d)
            values.append(v)
    if not depths:
        return np.array([]), np.array([])
    depth = np.asarray(depths, dtype=np.float64)
    value = np.asarray(values, dtype=np.float64)
    order = np.argsort(depth)
    depth = depth[order]
    value = value[order]
    depth, idx = np.unique(depth, return_index=True)
    value = value[idx]
    return depth, value


def resample_uniform(
    depth: np.ndarray,
    value: np.ndarray,
    num_points: int = CURVE_NUM_POINTS,
    depth_max: float = DEPTH_MAX,
    *,
    extrapolate_right: str = "zero_below_max",
) -> Tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, depth_max, num_points, dtype=np.float64)
    if depth.size == 0:
        return grid, np.zeros_like(grid)
    depth = depth - np.min(depth)
    dmax = float(np.max(depth))
    if dmax <= 0:
        return grid, np.full_like(grid, float(value[0]) if value.size else 0.0)
    depth = np.clip(depth, 0.0, depth_max)
    if extrapolate_right == "last":
        right = float(value[-1])
    else:
        right = float(value[-1]) if depth[-1] >= depth_max else 0.0
    v_grid = np.interp(grid, depth, value, left=value[0], right=right)
    return grid, v_grid


def trim_curve_truncation_tail(
    depth: np.ndarray,
    value_raw: np.ndarray,
    value_fitted: np.ndarray,
    *,
    raw_depth_max: float | None = None,
    depth_max: float = DEPTH_MAX,
    near_depth_margin: float = CURVE_TRUNCATION_DEPTH_MARGIN,
    drop_ratio_threshold: float = CURVE_TRUNCATION_DROP_RATIO,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove artificial cliff / extrapolated tail near depth_max (e.g. 2 um boundary)."""
    depth = np.asarray(depth, dtype=float)
    value_raw = np.asarray(value_raw, dtype=float)
    value_fitted = np.asarray(value_fitted, dtype=float)
    n = len(depth)
    if n < 2:
        return depth, value_raw, value_fitted

    if raw_depth_max is not None:
        keep = depth <= float(raw_depth_max) + 1e-6
        depth = depth[keep]
        value_raw = value_raw[keep]
        value_fitted = value_fitted[keep]
        n = len(depth)
        if n < 2:
            return depth, value_raw, value_fitted

    cut = n
    for i in range(n - 1, 0, -1):
        near_boundary = float(depth[i]) >= depth_max - near_depth_margin
        sharp_drop = (
            value_fitted[i] <= 0
            or value_raw[i] <= 0
            or (
                value_fitted[i - 1] > 0
                and value_fitted[i] / value_fitted[i - 1] < drop_ratio_threshold
            )
            or (
                value_raw[i - 1] > 0
                and value_raw[i] / value_raw[i - 1] < drop_ratio_threshold
            )
        )
        if near_boundary and sharp_drop:
            cut = i
        else:
            break
    return depth[:cut], value_raw[:cut], value_fitted[:cut]


def fit_curve_smooth(depth: np.ndarray, value: np.ndarray, method: str = CURVE_SMOOTH_METHOD) -> np.ndarray:
    value = np.clip(value, 1e-10, None)
    logv = np.log10(value)
    n = len(logv)
    if n < 5:
        return value.copy()
    if method == "savgol":
        win = min(n if n % 2 == 1 else n - 1, 51)
        win = max(win, 5)
        if win % 2 == 0:
            win -= 1
        smoothed_log = savgol_filter(logv, window_length=win, polyorder=3)
    else:
        k = min(3, n - 1)
        spl = UnivariateSpline(depth, logv, k=k, s=max(0.01 * n, 1.0))
        smoothed_log = spl(depth)
    return np.power(10.0, smoothed_log)


def process_curve_file(
    txt_path: Path,
    method: str = CURVE_SMOOTH_METHOD,
    *,
    curve_type: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_raw, value_raw = read_curve_txt(txt_path)
    if depth_raw.size == 0:
        grid = np.linspace(0.0, DEPTH_MAX, CURVE_NUM_POINTS)
        return grid, np.zeros_like(grid), np.zeros_like(grid)
    depth_raw = depth_raw - np.min(depth_raw)
    raw_depth_max = float(np.max(depth_raw))
    value_fitted = fit_curve_smooth(depth_raw, value_raw, method=method)
    grid, value_raw_grid = resample_uniform(depth_raw, value_raw, extrapolate_right="last")
    _, value_fitted_grid = resample_uniform(depth_raw, value_fitted, extrapolate_right="last")
    if curve_type in ("doping", "defect"):
        grid, value_raw_grid, value_fitted_grid = trim_curve_truncation_tail(
            grid, value_raw_grid, value_fitted_grid, raw_depth_max=raw_depth_max,
        )
    return grid, value_raw_grid, value_fitted_grid
