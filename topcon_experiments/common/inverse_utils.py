"""Efficiency tier definitions and inverse-design helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from topcon_experiments.config import ATHENA_FEATURES, EFF_TIER_ANCHOR_SHIFT, EFF_TIER_COUNT, EFF_TIER_STEP


def get_efficiency_tiers(df: pd.DataFrame, eff_col: str = "iv_Eff") -> list[dict]:
    """Build 4 efficiency tiers, 1% step, anchor shifted down (e.g. >=25, 24~25, 23~24, 22~23)."""
    max_eff = float(df[eff_col].max())
    anchor = int(np.floor(max_eff)) - int(EFF_TIER_ANCHOR_SHIFT)
    tiers = []

    for i in range(EFF_TIER_COUNT):
        if i == 0:
            label = f">={anchor}%"
            mask = df[eff_col] >= anchor
            eff_min, eff_max = float(anchor), max_eff
        else:
            hi = anchor - (i - 1) * EFF_TIER_STEP
            lo = hi - EFF_TIER_STEP
            label = f"{lo}%~{hi}%"
            mask = (df[eff_col] >= lo) & (df[eff_col] < hi)
            eff_min, eff_max = lo, hi

        tiers.append({
            "tier_id": i,
            "label": label,
            "eff_min": eff_min,
            "eff_max": eff_max,
            "count": int(mask.sum()),
            "mask": mask,
        })
    return tiers


def get_top_efficiency_bounds(
    df: pd.DataFrame,
    top_frac: float = 0.05,
    margin_frac: float = 0.15,
) -> tuple[list[tuple[float, float]], pd.DataFrame]:
    """Tight bounds from top-efficiency samples with margin."""
    n_top = max(20, int(len(df) * top_frac))
    top = df.nlargest(n_top, "iv_Eff")
    bounds = []
    for col in ATHENA_FEATURES:
        lo = float(top[col].min())
        hi = float(top[col].max())
        span = hi - lo
        lo = max(float(df[col].min()), lo - margin_frac * span)
        hi = min(float(df[col].max()), hi + margin_frac * span)
        if lo >= hi:
            lo, hi = float(df[col].min()), float(df[col].max())
        bounds.append((lo, hi))
    return bounds, top


def get_de_init_population(
    df: pd.DataFrame,
    bounds: list[tuple[float, float]],
    popsize: int,
    n_seed: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Initial DE population seeded with top-efficiency athena params."""
    rng = np.random.default_rng(seed)
    n_var = len(ATHENA_FEATURES)
    pop = np.zeros((popsize, n_var))
    top_rows = df.nlargest(n_seed, "iv_Eff")
    for i in range(min(n_seed, popsize)):
        pop[i] = top_rows[ATHENA_FEATURES].iloc[i % len(top_rows)].values.astype(float)
    for i in range(n_seed, popsize):
        for j, (lo, hi) in enumerate(bounds):
            pop[i, j] = rng.uniform(lo, hi)
    return pop


def get_high_efficiency_init_population(
    df: pd.DataFrame,
    bounds: list[tuple[float, float]],
    popsize: int,
    n_top: int = 50,
    n_band: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """Seed DE from top-N samples plus random draws from the 25%~26% band."""
    rng = np.random.default_rng(seed)
    n_var = len(ATHENA_FEATURES)
    pop = np.zeros((popsize, n_var))

    top_rows = df.nlargest(n_top, "iv_Eff")
    anchor = int(np.floor(float(df["iv_Eff"].max())))
    band_lo = anchor - EFF_TIER_STEP
    band = df[(df["iv_Eff"] >= band_lo) & (df["iv_Eff"] < anchor)]
    if len(band) == 0:
        band = top_rows

    idx = 0
    for i in range(min(len(top_rows), popsize)):
        pop[idx] = top_rows[ATHENA_FEATURES].iloc[i].values.astype(float)
        idx += 1
        if idx >= popsize:
            return pop

    band_sample_n = min(n_band, max(0, popsize - idx), len(band))
    if band_sample_n > 0:
        sampled = band.sample(n=band_sample_n, random_state=seed)
        for _, row in sampled.iterrows():
            pop[idx] = row[ATHENA_FEATURES].values.astype(float)
            idx += 1
            if idx >= popsize:
                return pop

    for i in range(idx, popsize):
        for j, (lo, hi) in enumerate(bounds):
            pop[i, j] = rng.uniform(lo, hi)
    return pop


def get_de_iv_scale_factors() -> dict[str, float]:
    """Per-metric multipliers derived from calibration reference/target."""
    from topcon_experiments.config import (
        DE_IV_CALIB_REFERENCE,
        DE_IV_CALIB_TARGET,
        DE_IV_SCALE_ENABLED,
        IV_TARGETS,
    )

    if not DE_IV_SCALE_ENABLED:
        return {m: 1.0 for m in IV_TARGETS}
    scales: dict[str, float] = {}
    for m in IV_TARGETS:
        ref = float(DE_IV_CALIB_REFERENCE.get(m, 1.0))
        tgt = float(DE_IV_CALIB_TARGET.get(m, ref))
        scales[m] = (tgt / ref) if ref != 0 else 1.0
    return scales


def scale_iv_for_de(metric: str, raw: float) -> float:
    return raw * get_de_iv_scale_factors().get(metric, 1.0)


def apply_iv_scale_to_dict(iv: dict[str, float]) -> dict[str, float]:
    """Return copy with IV keys scaled for inverse-design reporting."""
    from topcon_experiments.config import IV_TARGETS

    out = dict(iv)
    for m in IV_TARGETS:
        if m in out:
            out[m] = scale_iv_for_de(m, float(out[m]))
    return out


def multistart_refine_efficiency(
    chain,
    df: pd.DataFrame,
    bounds: list[tuple[float, float]],
    n_starts: int = 10,
) -> tuple[np.ndarray, float]:
    """Local refine from top dataset athena params; return best x and scaled iv_Eff."""
    from scipy.optimize import minimize

    def neg_eff(x: np.ndarray) -> float:
        raw = float(chain.predict_iv_single(x, "iv_Eff"))
        return -scale_iv_for_de("iv_Eff", raw)

    best_x = None
    best_val = -np.inf
    starts = df.nlargest(n_starts, "iv_Eff")

    for _, row in starts.iterrows():
        x0 = row[ATHENA_FEATURES].values.astype(float)
        res = minimize(
            neg_eff,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200},
        )
        val = -float(res.fun)
        if val > best_val:
            best_val = val
            best_x = res.x

    if best_x is None:
        raise RuntimeError("multistart_refine_efficiency found no valid start points")
    return best_x, best_val

