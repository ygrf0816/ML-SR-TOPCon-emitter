"""Weak shape priors for defect (vacancy) depth profiles.

Why the old model produced negative L
------------------------------------
The single-exponential form ``N(z) = N0 * exp(-z/L)`` forces a *monotonic
decay from the surface*. About half of the TCAD vacancy profiles instead
*rise* into the bulk (broad subsurface peak / surface depletion). The only
way that formula can rise is ``L < 0``, which is not a physical length and
broke plotting when ``L`` was clamped to ``1e-6``.

New prior (L always > 0)
------------------------
  constant : N(z) = N0
  offset   : N(z) = Nb + (Ns - Nb) * exp(-z / L),   L > 0

- Ns = surface concentration, Nb = deep bulk asymptote
- Ns > Nb : surface-high, decays into bulk (classic)
- Ns < Nb : surface-low, rises toward bulk (former "negative L" cases)
- Ns ≈ Nb : nearly flat

Process-layer regression targets: ln_Ns, ln_Nb, ln_L.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

L_MIN = 0.05   # um
L_MAX = 50.0   # um
L_FLAT = 1.0e6  # sentinel for constant prior


def _r2_ln(N_obs: np.ndarray, N_pred: np.ndarray) -> float:
    mask = (N_obs > 0) & (N_pred > 0) & np.isfinite(N_pred)
    if mask.sum() < 4:
        return float("nan")
    y = np.log(N_obs[mask])
    yhat = np.log(N_pred[mask])
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot


def predict_constant(z: np.ndarray, ln_N0: float) -> np.ndarray:
    return np.full(np.asarray(z, dtype=float).shape, float(np.exp(ln_N0)), dtype=float)


def predict_offset(z: np.ndarray, ln_Ns: float, ln_Nb: float, L: float) -> np.ndarray:
    """N(z) = Nb + (Ns - Nb) * exp(-z/L) with L > 0."""
    z = np.asarray(z, dtype=float)
    Ns = float(np.exp(ln_Ns))
    Nb = float(np.exp(ln_Nb))
    L = float(np.clip(L, L_MIN, L_MAX)) if np.isfinite(L) and L > 0 else L_FLAT
    if L >= 0.5 * L_FLAT:
        # effectively constant at geometric mean / Nb
        return np.full_like(z, Nb, dtype=float)
    out = Nb + (Ns - Nb) * np.exp(-z / L)
    return np.maximum(out, 1e-30)


def predict_prior(
    z: np.ndarray,
    prior_type: str,
    ln_Ns: float,
    ln_Nb: float | None = None,
    L: float = L_FLAT,
    **_legacy,
) -> np.ndarray:
    """Rebuild curve from prior parameters.

    Accepts legacy kwargs ``ln_N0`` for constant-only callers.
    """
    if "ln_N0" in _legacy and ln_Nb is None:
        ln_Ns = float(_legacy["ln_N0"])
    if prior_type == "constant":
        return predict_constant(z, float(ln_Ns))
    if ln_Nb is None:
        ln_Nb = float(ln_Ns)
    return predict_offset(z, float(ln_Ns), float(ln_Nb), float(L))


def fit_one(z: np.ndarray, N: np.ndarray) -> dict:
    """Fit constant vs offset-exp (L>0); return the better one."""
    z = np.asarray(z, dtype=float)
    N = np.asarray(N, dtype=float)
    keep = np.isfinite(z) & np.isfinite(N) & (N > 0)
    z, N = z[keep], N[keep]
    if z.size < 6:
        return {"error": "too few points"}

    lnN = np.log(N)
    ln_mean = float(np.mean(lnN))
    N_c = predict_constant(z, ln_mean)
    r2_c = _r2_ln(N, N_c)

    # Seeds: surface / bulk medians over first/last 10% of depth span
    z0, z1 = float(z.min()), float(z.max())
    span = max(z1 - z0, 1e-9)
    Ns0 = float(np.median(N[z <= z0 + 0.1 * span]))
    Nb0 = float(np.median(N[z >= z1 - 0.1 * span]))
    Ns0 = max(Ns0, 1e10)
    Nb0 = max(Nb0, 1e10)
    L0 = float(np.clip(0.5 * span, L_MIN, L_MAX))

    def model_ln(zz, ln_ns, ln_nb, L):
        pred = predict_offset(zz, ln_ns, ln_nb, L)
        return np.log(np.maximum(pred, 1e-30))

    r2_o = float("nan")
    ln_Ns_o = ln_Nb_o = ln_mean
    L_o = L_FLAT
    try:
        popt, _ = curve_fit(
            model_ln, z, lnN,
            p0=(np.log(Ns0), np.log(Nb0), L0),
            bounds=(
                (np.log(Ns0) - 5.0, np.log(Nb0) - 5.0, L_MIN),
                (np.log(Ns0) + 5.0, np.log(Nb0) + 5.0, L_MAX),
            ),
            maxfev=8000,
        )
        ln_Ns_o, ln_Nb_o, L_o = (float(v) for v in popt)
        L_o = float(np.clip(L_o, L_MIN, L_MAX))
        # Enforce L > 0 strictly (numerical guard)
        if L_o <= 0:
            L_o = L_MIN
        N_o = predict_offset(z, ln_Ns_o, ln_Nb_o, L_o)
        r2_o = _r2_ln(N, N_o)
    except Exception:
        pass

    # Prefer offset only if it clearly beats constant (avoid overfitting noise)
    use_offset = np.isfinite(r2_o) and (
        (not np.isfinite(r2_c) and r2_o > 0.05)
        or (np.isfinite(r2_c) and r2_o > r2_c + 0.02)
    )
    if use_offset:
        return {
            "prior_type": "offset",
            "ln_Ns": ln_Ns_o,
            "ln_Nb": ln_Nb_o,
            "ln_L": float(np.log(L_o)),
            "L_um": L_o,
            "ln_N0": ln_Ns_o,  # alias: surface
            "r2_ln": float(r2_o),
            "r2_constant": float(r2_c) if np.isfinite(r2_c) else float("nan"),
            "r2_offset": float(r2_o),
            "n_fit": int(z.size),
            "depth_min_um": float(z.min()),
            "depth_max_um": float(z.max()),
            "Ns_over_Nb": float(np.exp(ln_Ns_o - ln_Nb_o)),
        }
    return {
        "prior_type": "constant",
        "ln_Ns": ln_mean,
        "ln_Nb": ln_mean,
        "ln_L": float(np.log(L_FLAT)),
        "L_um": L_FLAT,
        "ln_N0": ln_mean,
        "r2_ln": float(r2_c) if np.isfinite(r2_c) else float("nan"),
        "r2_constant": float(r2_c) if np.isfinite(r2_c) else float("nan"),
        "r2_offset": float(r2_o) if np.isfinite(r2_o) else float("nan"),
        "n_fit": int(z.size),
        "depth_min_um": float(z.min()),
        "depth_max_um": float(z.max()),
        "Ns_over_Nb": 1.0,
    }
