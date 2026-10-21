"""Parse partial process parameters from literature extraction table."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from topcon_experiments.config import ATHENA_FEATURES

EXTRACT_DIR = Path(__file__).resolve().parents[1] / "extract_literature_info"
FULL_TABLE = EXTRACT_DIR / "full_table.csv"

_RANGE_RE = re.compile(
    r"(?P<lo>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*[-~]\s*(?P<hi>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def _parse_scalar(val) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if not s or s.upper() == "NA":
        return None
    if s.startswith("NA"):
        m = _RANGE_RE.search(s.replace("_range(", "").replace(")", ""))
        if m:
            lo, hi = float(m.group("lo")), float(m.group("hi"))
            return 0.5 * (lo + hi)
        return None
    if "_or_" in s:
        parts = [p for p in s.split("_or_") if p and p.upper() != "NA"]
        nums = [_parse_scalar(p) for p in parts]
        nums = [n for n in nums if n is not None]
        return float(np.mean(nums)) if nums else None
    if s.startswith("~"):
        s = s[1:]
    if s.startswith("inferred"):
        m = re.search(r"([\d.]+)\s*[-~]\s*([\d.]+e[+-]?\d+)", s, flags=re.I)
        if m:
            return 0.5 * (float(m.group(1)) + float(m.group(2)))
        m2 = re.search(r"([\d.]+e[+-]?\d+)", s, flags=re.I)
        if m2:
            return float(m2.group(1))
        m3 = re.search(r"([\d.]+)", s)
        return float(m3.group(1)) if m3 else None
    m = _RANGE_RE.search(s)
    if m:
        return 0.5 * (float(m.group("lo")) + float(m.group("hi")))
    try:
        return float(s)
    except ValueError:
        return None


def _parse_flow(val) -> float | None:
    """Extract first numeric flow (sccm) from strings like '3000_sccm(oxidation)'."""
    if val is None or str(val).strip().upper() in ("", "NA"):
        return None
    s = str(val)
    m = re.search(r"([\d.]+)\s*sccm", s, flags=re.I)
    if m:
        return float(m.group(1))
    m2 = re.search(r"([\d.]+)", s)
    return float(m2.group(1)) if m2 else None


def load_parsed_table(path: Path | None = None) -> pd.DataFrame:
    path = path or FULL_TABLE
    raw = pd.read_csv(path)
    rows: list[dict] = []
    for _, r in raw.iterrows():
        row = {
            "profile_id": r["profile_id"],
            "ref": r["ref"],
            "confidence": r.get("confidence", ""),
            "R_sheet_measured": float(r["R_sheet_measured_ohm_sq"]),
            "N_p_table": float(r["N_p_cm3"]),
            "z_p_table": float(r["z_p_um"]),
        }
        for col in ATHENA_FEATURES:
            val = r.get(col, r.get(col.replace("athena_", f"athena_{col}") if False else col))
            # columns in csv use names like athena_thick_um mapped below
            pass
        mapping = {
            "athena_thick": r.get("athena_thick_um"),
            "athena_c_boron": r.get("athena_c_boron_cm3"),
            "athena_temp1": r.get("athena_temp1_C"),
            "athena_time1": r.get("athena_time1_min"),
            "athena_temp2": r.get("athena_temp2_C"),
            "athena_time2": r.get("athena_time2_min"),
            "athena_F_N2": r.get("athena_F_N2"),
            "athena_F_O2": r.get("athena_F_O2"),
        }
        known: dict[str, float] = {}
        missing: list[str] = []
        for k, v in mapping.items():
            if k in ("athena_F_N2", "athena_F_O2"):
                num = _parse_flow(v)
            else:
                num = _parse_scalar(v)
            if num is not None and np.isfinite(num):
                known[k] = float(num)
            else:
                missing.append(k)
        row["known_params"] = known
        row["missing_params"] = missing
        row["n_known"] = len(known)
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_feature_bounds() -> dict[str, tuple[float, float]]:
    """5–95% quantile bounds from simulation dataset for free parameters."""
    from topcon_experiments.common.data import load_raw_dataframe
    from topcon_experiments.config import CURVE_DESCRIPTORS, DEFECT_DESCRIPTORS, MODEL2_FEATURES

    df = load_raw_dataframe()
    bounds: dict[str, tuple[float, float]] = {}
    for col in MODEL2_FEATURES + list(DEFECT_DESCRIPTORS):
        if col not in df.columns:
            continue
        s = df[col].astype(float)
        s = s[np.isfinite(s) & (s > 0)]
        if s.empty:
            continue
        bounds[col] = (float(s.quantile(0.05)), float(s.quantile(0.95)))
    return bounds
