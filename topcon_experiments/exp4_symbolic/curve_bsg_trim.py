"""Depth-window trim for tail-only curve SR experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from topcon_experiments.config import CURVE_TAIL_DEPTH_MIN, DEPTH_MAX

# Legacy adaptive BSG detection (kept for reference; tail experiment uses fixed window)
BSG_DOPING_SEARCH_DEPTH_UM = 0.15
BSG_DOPING_PEAK_FRAC = 0.5
BSG_DEFECT_PLATEAU_DEPTH_UM = 0.25
BSG_DEFECT_PLATEAU_DROP = 0.03


def apply_tail_depth_window(
    df: pd.DataFrame,
    curve_type: str,
    *,
    depth_min: float = CURVE_TAIL_DEPTH_MIN,
    depth_max: float = DEPTH_MAX,
) -> pd.DataFrame:
    """Keep only depth in [depth_min, depth_max]; drop shallow BSG (0 ~ depth_min um)."""
    rows: list[dict] = []
    for fb in df["file_base"].unique():
        sub = df[df["file_base"] == fb].sort_values("depth_um")
        d = sub["depth_um"].values.astype(float)
        keep = (d >= float(depth_min) - 1e-9) & (d <= float(depth_max) + 1e-9)
        for i in np.where(keep)[0]:
            rows.append({
                "file_base": fb,
                "curve_type": curve_type,
                "depth_um": float(d[i]),
                "value_raw": float(sub.iloc[i]["value_raw"]),
                "value_fitted": float(sub.iloc[i]["value_fitted"]),
                "tail_depth_min_um": float(depth_min),
            })
    return pd.DataFrame(rows)


def tail_depth_mask(
    depth_um: np.ndarray,
    *,
    depth_min: float = CURVE_TAIL_DEPTH_MIN,
    depth_max: float = DEPTH_MAX,
) -> np.ndarray:
    d = np.asarray(depth_um, dtype=float)
    return (d >= float(depth_min) - 1e-9) & (d <= float(depth_max) + 1e-9)


# --- Adaptive BSG-valley trimming ---
#
# Raw profile structure (surface -> bulk):
#   (1) high, roughly flat BSG plateau;
#   (2) steep log-concentration cliff dropping to a minimum (the valley);
#   (3) the in-silicon double-Gaussian segment (rise to subsurface peak, then
#       diffusion decay) — this is the ONLY part we keep.
# The correct cut is exactly the valley between (2) and (3). When the DG peak
# has merged into the cliff (monotonic decay, z_p ~ 0), there is no valley and
# we cut at the cliff "knee" (where the steep slope has relaxed).

ADAPTIVE_SEARCH_MAX_UM = 1.0   # cliff/valley must lie within this depth
ADAPTIVE_KNEE_FRAC = 0.10      # cliff considered ended when |slope| < frac*|max cliff slope|
ADAPTIVE_MIN_RISE_LN = 0.05    # ln-units; min valley->peak rise to count as a real hump
ADAPTIVE_SURFACE_SKIP_UM = 0.01  # ignore the first few surface points (grid noise)


def detect_bsg_cliff_end(
    depth_um: np.ndarray,
    value_fitted: np.ndarray,
    athena_thick: float | None = None,
    *,
    search_max_um: float = ADAPTIVE_SEARCH_MAX_UM,
    knee_frac: float = ADAPTIVE_KNEE_FRAC,
    min_rise_ln: float = ADAPTIVE_MIN_RISE_LN,
) -> int:
    """Return index (into the ORIGINAL array) of the post-cliff valley.

    Algorithm (all in ln-concentration vs depth):
      1. Anchor at the steepest descent point (the BSG cliff). The cliff slope
         is one to two orders steeper than any diffusion falloff, so the global
         ``argmin`` of the slope inside ``search_max_um`` is unambiguous.
      2. Walk forward until the slope relaxes to ``knee_frac`` of the cliff
         slope — the knee, i.e. the earliest point where the cliff is over.
      3. Look for the dominant post-knee peak. If concentration rises by at
         least ``min_rise_ln`` from the minimum before that peak, the profile
         has a real hump: cut at the valley = argmin over [cliff, peak].
      4. Otherwise the profile decays monotonically: cut at the knee.

    ``athena_thick`` is accepted for backward compatibility but no longer
    constrains the cut: forcing ``cut >= thick + margin`` was observed to
    over-trim samples whose valley sits right at the interface, while the old
    absolute slope threshold (20/um) under-trimmed thick-BSG samples.
    """
    depth = np.asarray(depth_um, dtype=float)
    c = np.asarray(value_fitted, dtype=float)
    if depth.size < 5:
        return 0
    order = np.argsort(depth)
    d = depth[order]
    lc = np.log(np.maximum(c[order], 1e-30))
    slope = np.gradient(lc, d)

    i_end = int(np.searchsorted(d, float(search_max_um)))
    i_end = min(max(i_end, 3), d.size)
    i_start = int(np.searchsorted(d, float(ADAPTIVE_SURFACE_SKIP_UM)))
    i_start = min(max(i_start, 1), i_end - 2)

    # Step 1: steepest descent = cliff anchor.
    i_cliff = i_start + int(np.argmin(slope[i_start:i_end]))
    cliff_slope = float(slope[i_cliff])
    if cliff_slope >= 0.0:
        return int(order[0])  # no cliff at all; keep everything

    # Step 2: knee = first point after the cliff where the slope has relaxed.
    relax_threshold = float(knee_frac) * cliff_slope  # negative, closer to 0
    i_knee = i_end - 1
    for i in range(i_cliff + 1, i_end):
        if float(slope[i]) > relax_threshold:
            i_knee = i
            break

    # Step 3: dominant peak after the knee; valley = minimum before it.
    seg = lc[i_knee:i_end]
    i_peak = i_knee + int(np.argmax(seg))
    if i_peak > i_knee:
        i_valley = i_cliff + int(np.argmin(lc[i_cliff:i_peak + 1]))
        if float(lc[i_peak] - lc[i_valley]) >= float(min_rise_ln):
            return int(order[i_valley])

    # Step 4: monotonic decay — cut at the knee.
    return int(order[i_knee])


def apply_adaptive_trim(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    curve_type: str,
    depth_max: float = DEPTH_MAX,
    floor: float | None = None,
) -> pd.DataFrame:
    """Per-sample adaptive BSG-cliff trim, keeping absolute physical depth.

    Parameters
    ----------
    floor
        Optional concentration floor (cm^-3). When provided, additionally
        keeps only points with ``value_fitted >= floor`` (used by the
        ``adaptive_floor1e18`` experiment variant).
    """
    rows: list[dict] = []
    for fb in df["file_base"].unique():
        sub = df[df["file_base"] == fb].sort_values("depth_um")
        d = sub["depth_um"].values.astype(float)
        c = sub["value_fitted"].values.astype(float)
        if d.size == 0:
            continue
        if fb in meta.index:
            thick = float(meta.loc[fb, "athena_thick"])
        else:
            thick = 0.1
        ic = detect_bsg_cliff_end(d, c, thick)
        keep = np.zeros(d.size, dtype=bool)
        keep[ic:] = True
        keep &= d <= float(depth_max) + 1e-9
        if floor is not None:
            keep &= c >= float(floor) - 1e-9
        cut_depth = float(d[ic]) if ic < d.size else float(d[-1])
        for i in np.where(keep)[0]:
            rows.append({
                "file_base": fb,
                "curve_type": curve_type,
                "depth_um": float(d[i]),
                "value_raw": float(sub.iloc[i]["value_raw"]),
                "value_fitted": float(sub.iloc[i]["value_fitted"]),
                "adaptive_depth_min_um": cut_depth,
            })
    return pd.DataFrame(rows)


def apply_keep_bsg(
    df: pd.DataFrame,
    *,
    curve_type: str,
    depth_max: float = DEPTH_MAX,
    floor: float | None = None,
) -> pd.DataFrame:
    """Keep the full curve (BSG included); optionally truncate deep tail below floor.

    ``floor`` only filters the deep low-concentration tail (no shallow BSG cut).
    Used by the ``keep_bsg`` and ``keep_bsg_floor1e18`` experiment variants to
    test whether keeping the BSG region yields SR formulas closer to the
    literature double-Gaussian shape.
    """
    rows: list[dict] = []
    for fb in df["file_base"].unique():
        sub = df[df["file_base"] == fb].sort_values("depth_um")
        d = sub["depth_um"].values.astype(float)
        c = sub["value_fitted"].values.astype(float)
        if d.size == 0:
            continue
        keep = d <= float(depth_max) + 1e-9
        if floor is not None:
            keep &= c >= float(floor) - 1e-9
        for i in np.where(keep)[0]:
            rows.append({
                "file_base": fb,
                "curve_type": curve_type,
                "depth_um": float(d[i]),
                "value_raw": float(sub.iloc[i]["value_raw"]),
                "value_fitted": float(sub.iloc[i]["value_fitted"]),
                "adaptive_depth_min_um": 0.0,
            })
    return pd.DataFrame(rows)
