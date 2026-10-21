"""Extract doping curve scalar descriptors from N(z) profiles."""

from __future__ import annotations

import numpy as np

N_FLOOR_DEFAULT = 4.0e18


def extract_doping_descriptors(
    depth_um: np.ndarray,
    N_cm3: np.ndarray,
    *,
    N_floor: float = N_FLOOR_DEFAULT,
) -> dict[str, float]:
    """Compute dataset-aligned doping descriptors from a 1D profile."""
    depth = np.asarray(depth_um, dtype=float)
    N = np.asarray(N_cm3, dtype=float)
    if depth.size < 2:
        raise ValueError("profile needs at least 2 points")

    order = np.argsort(depth)
    depth = depth[order]
    N = N[order]

    peak_idx = int(np.argmax(N))
    N_peak = float(N[peak_idx])
    x_peak = float(depth[peak_idx])

    # junction: last depth with N >= floor
    above = N >= N_floor
    if not np.any(above):
        junction_depth = float(depth[0])
    else:
        junction_depth = float(depth[np.where(above)[0][-1]])

    # FWHM at half maximum (within valid region)
    half = 0.5 * N_peak
    mask = (depth <= junction_depth) & (N >= half)
    if np.count_nonzero(mask) >= 2:
        d_sub = depth[mask]
        fwhm = float(d_sub.max() - d_sub.min())
    else:
        fwhm = float("nan")

    grad = np.gradient(N, depth)
    gradient_max = float(np.max(np.abs(grad)))

    # dose integral (cm^-3 * um) up to junction
    j_idx = np.searchsorted(depth, junction_depth, side="right")
    dose = float(np.trapz(N[:j_idx], depth[:j_idx]))

    return {
        "doping_N_peak": N_peak,
        "doping_x_peak": x_peak,
        "doping_junction_depth": junction_depth,
        "doping_FWHM": fwhm,
        "doping_gradient_max": gradient_max,
        "doping_dose": dose,
    }
