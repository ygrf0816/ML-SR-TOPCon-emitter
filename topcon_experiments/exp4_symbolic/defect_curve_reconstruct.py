"""Reconstruct defect vacancy curves from predicted scalar descriptors (scheme B)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def solve_exponential_decay_length(
    n_peak: float,
    dose: float,
    depth_max: float,
    *,
    min_l: float = 1e-4,
    max_l: float = 50.0,
) -> float:
    """Find L such that ∫₀^depth_max N_peak·exp(-x/L) dx = dose."""
    n_peak = max(float(n_peak), 1e-30)
    dose = max(float(dose), 1e-30)
    depth_max = max(float(depth_max), 1e-6)

    def residual(L: float) -> float:
        return n_peak * L * (1.0 - np.exp(-depth_max / L)) - dose

    try:
        lo, hi = min_l, max_l
        if residual(lo) * residual(hi) > 0:
            return depth_max
        return float(brentq(residual, lo, hi))
    except (ValueError, RuntimeError):
        return depth_max


def reconstruct_defect_exponential(
    depth_um: np.ndarray,
    n_peak: float,
    dose: float,
    *,
    depth_max: float | None = None,
) -> np.ndarray:
    """N(x) = N_peak · exp(-x/L), L from dose integral constraint."""
    depth_um = np.asarray(depth_um, dtype=float)
    dmax = float(depth_max if depth_max is not None else np.max(depth_um))
    L = solve_exponential_decay_length(n_peak, dose, dmax)
    return np.maximum(n_peak * np.exp(-depth_um / L), 1e-30)


def reconstruct_defect_flat_plus_decay(
    depth_um: np.ndarray,
    n_peak: float,
    dose: float,
    *,
    depth_max: float | None = None,
) -> np.ndarray:
    """Blend flat tail (dose/depth) with surface peak: robust when exp model misfits."""
    depth_um = np.asarray(depth_um, dtype=float)
    dmax = max(float(depth_max if depth_max is not None else np.max(depth_um)), 1e-6)
    n_flat = dose / dmax
    exp_prof = reconstruct_defect_exponential(depth_um, n_peak, dose, depth_max=dmax)
    # Weight toward exponential near surface, flat at depth
    w = np.exp(-depth_um / (0.3 * dmax + 1e-6))
    return np.maximum(w * exp_prof + (1.0 - w) * n_flat, 1e-30)
