"""Adaptive depth sampling and fit weights for curve symbolic regression."""

from __future__ import annotations

import numpy as np

from topcon_experiments.config import (
    SR_CURVE_GRAD_POWER_DOPING,
    SR_CURVE_UNIFORM_FRAC,
    SR_CURVE_UNIFORM_FRAC_DOPING,
)


def _uniform_frac_for(curve_type: str | None) -> float:
    if curve_type == "doping":
        return SR_CURVE_UNIFORM_FRAC_DOPING
    return SR_CURVE_UNIFORM_FRAC


def curve_change_score(depth_um: np.ndarray, conc: np.ndarray, *, curve_type: str | None = None) -> np.ndarray:
    """Per-point change score from |d ln(c) / d depth|; higher at transition regions."""
    conc = np.maximum(np.asarray(conc, dtype=float), 1e-30)
    depth = np.asarray(depth_um, dtype=float)
    y = np.log(conc)
    if len(depth) < 2:
        return np.ones(len(depth), dtype=float)
    grad = np.abs(np.gradient(y, depth))
    if len(grad) >= 5:
        kernel = np.ones(5, dtype=float) / 5.0
        grad = np.convolve(grad, kernel, mode="same")
    floor = float(np.percentile(grad, 10)) * (0.1 if curve_type == "doping" else 0.25) + 1e-12
    if curve_type == "doping":
        grad = np.power(grad, SR_CURVE_GRAD_POWER_DOPING)
    return floor + grad


def adaptive_sample_indices(
    depth_um: np.ndarray,
    conc: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
    *,
    uniform_frac: float | None = None,
    curve_type: str | None = None,
) -> np.ndarray:
    """More samples where concentration changes fast; fewer on flat tails."""
    n = min(int(n_points), len(depth_um))
    if n <= 0:
        return np.array([], dtype=int)
    if n >= len(depth_um):
        return np.arange(len(depth_um), dtype=int)

    if uniform_frac is None:
        uniform_frac = _uniform_frac_for(curve_type)
    score = curve_change_score(depth_um, conc, curve_type=curve_type)
    n_uniform = max(1, int(round(n * uniform_frac)))
    n_adaptive = max(0, n - n_uniform)

    uniform_idx = np.linspace(0, len(depth_um) - 1, n_uniform).astype(int)
    chosen: set[int] = set(uniform_idx.tolist())

    if n_adaptive > 0:
        prob = score / score.sum()
        cdf = np.cumsum(prob)
        targets = (np.arange(n_adaptive, dtype=float) + 0.5) / n_adaptive
        adapt_idx = np.searchsorted(cdf, targets, side="left")
        adapt_idx = np.clip(adapt_idx, 0, len(depth_um) - 1)
        chosen.update(int(i) for i in adapt_idx)

    idx = np.array(sorted(chosen), dtype=int)
    remaining = np.setdiff1d(np.arange(len(depth_um)), idx)
    while len(idx) < n and len(remaining) > 0:
        pick = int(remaining[np.argmax(score[remaining])])
        idx = np.sort(np.append(idx, pick))
        remaining = np.setdiff1d(remaining, pick)

    if len(idx) > n:
        # keep highest-change points
        keep_score = score[idx]
        top = idx[np.argsort(keep_score)[::-1][:n]]
        idx = np.sort(top)
    return idx


def fit_sample_weights(
    depth_um: np.ndarray,
    conc: np.ndarray,
    indices: np.ndarray,
    *,
    curve_type: str | None = None,
) -> np.ndarray:
    """Normalized weights for PySR fit (mean=1); emphasize transition regions."""
    score = curve_change_score(depth_um[indices], conc[indices], curve_type=curve_type)
    w = score / max(float(score.mean()), 1e-12)
    return w.astype(float)
