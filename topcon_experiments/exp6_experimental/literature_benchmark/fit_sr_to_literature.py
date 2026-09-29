"""Fit SR formulas to literature curves / R_sheet with partial process constraints."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.mpl_style import apply_plot_style
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import (
    ATHENA_FEATURES,
    CURVE_TAIL_DEPTH_MIN,
    DEFECT_DESCRIPTORS,
    DEPTH_MAX,
    DOPING_DESCRIPTORS,
    LOG_FEATURES,
    MODEL2_FEATURES,
    OUTPUT_ROOT,
    SR_DOPING_CURVE_FORMULA_PATH,
)
from topcon_experiments.exp4_symbolic.sr_equation_utils import eval_sympy_expr, sort_formulas_by_accuracy
from topcon_experiments.exp6_experimental.literature_benchmark.curve_descriptors import (
    N_FLOOR_DEFAULT,
    extract_doping_descriptors,
)
from topcon_experiments.exp6_experimental.literature_benchmark.parse_literature_params import (
    dataset_feature_bounds,
    load_parsed_table,
)

LIT_CURVES_DIR = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "literature_double_gaussian"
    / "data"
    / "curves"
)
BASE_OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "literature_benchmark"
R_SHEET_SR = OUTPUT_ROOT / "exp4_symbolic" / "sr_tabular_athena_to_doping_R_sheet.csv"
CURVE_SR_LEGACY = SR_DOPING_CURVE_FORMULA_PATH  # tail_full: 18-dim, depth 0.25~2 um
ADAPTIVE_EXP_DIR = OUTPUT_ROOT / "exp4_symbolic" / "curve_adaptive_experiment"
CURVE_SR_ADAPTIVE = ADAPTIVE_EXP_DIR / "sr_formulas_doping_adaptive_full.csv"
CURVE_SR_ADAPTIVE_FLOOR = ADAPTIVE_EXP_DIR / "sr_formulas_doping_adaptive_floor1e18_full.csv"
CURVE_SR_KEEP_BSG = ADAPTIVE_EXP_DIR / "sr_formulas_doping_keep_bsg_full.csv"
CURVE_SR_KEEP_BSG_FLOOR = ADAPTIVE_EXP_DIR / "sr_formulas_doping_keep_bsg_floor1e18_full.csv"

# Each variant: (label, formula path, feature order, N_floor for fit window, depth offset)
# feature_order is the column order the SR formula was trained on.
# - legacy_tail: 18-dim (MODEL2 minus doping_R_sheet + depth); depth window [0.25, 2] um,
#   literature depth 0 == physical 0.25 um, so offset = 0.25.
# - adaptive / adaptive_floor1e18: 19-dim (full MODEL2 + depth); absolute physical depth,
#   literature depth is already physical, offset = 0.
@dataclass
class CurveVariant:
    label: str
    formula_path: Path
    feature_order: list[str]
    n_floor: float
    depth_offset_um: float
    fit_depth_max_um: float


def _legacy_feature_order() -> list[str]:
    return [c for c in MODEL2_FEATURES if c != "doping_R_sheet"] + ["depth_um"]


def _full_feature_order() -> list[str]:
    return list(MODEL2_FEATURES) + ["depth_um"]


VARIANTS: dict[str, CurveVariant] = {
    "adaptive": CurveVariant(
        label="adaptive",
        formula_path=CURVE_SR_ADAPTIVE,
        feature_order=_full_feature_order(),
        n_floor=N_FLOOR_DEFAULT,
        depth_offset_um=0.0,
        fit_depth_max_um=DEPTH_MAX,
    ),
    "adaptive_floor1e18": CurveVariant(
        label="adaptive_floor1e18",
        formula_path=CURVE_SR_ADAPTIVE_FLOOR,
        feature_order=_full_feature_order(),
        n_floor=1.0e18,
        depth_offset_um=0.0,
        fit_depth_max_um=DEPTH_MAX,
    ),
    "keep_bsg": CurveVariant(
        label="keep_bsg",
        formula_path=CURVE_SR_KEEP_BSG,
        feature_order=_full_feature_order(),
        n_floor=N_FLOOR_DEFAULT,
        depth_offset_um=0.0,
        fit_depth_max_um=DEPTH_MAX,
    ),
    "keep_bsg_floor1e18": CurveVariant(
        label="keep_bsg_floor1e18",
        formula_path=CURVE_SR_KEEP_BSG_FLOOR,
        feature_order=_full_feature_order(),
        n_floor=1.0e18,
        depth_offset_um=0.0,
        fit_depth_max_um=DEPTH_MAX,
    ),
    "legacy_tail": CurveVariant(
        label="legacy_tail",
        formula_path=CURVE_SR_LEGACY,
        feature_order=_legacy_feature_order(),
        n_floor=N_FLOOR_DEFAULT,
        depth_offset_um=CURVE_TAIL_DEPTH_MIN,
        fit_depth_max_um=DEPTH_MAX - CURVE_TAIL_DEPTH_MIN,
    ),
}
DEFAULT_VARIANT = "adaptive"

LOG_IN_MODEL = set(LOG_FEATURES)


def _best_expr(path: Path) -> str:
    df = sort_formulas_by_accuracy(pd.read_csv(path))
    row = df.iloc[0]
    return str(row.get("sympy_format") or row.get("equation", ""))


def _feature_vector(
    values: dict[str, float],
    depth_um: float | np.ndarray,
    *,
    feature_order: list[str] | None = None,
) -> np.ndarray:
    """Build model2+depth row(s) with log transform on training features."""
    order = feature_order if feature_order is not None else _full_feature_order()
    if np.isscalar(depth_um):
        depths = np.array([float(depth_um)])
        single = True
    else:
        depths = np.asarray(depth_um, dtype=float)
        single = False
    n = len(depths)
    mat = np.zeros((n, len(order)), dtype=float)
    for j, col in enumerate(order[:-1]):
        v = float(values[col])
        if col in LOG_IN_MODEL:
            v = np.log(max(v, 1e-30))
        mat[:, j] = v
    mat[:, -1] = depths
    return mat[0] if single else mat


def _predict_ln_curve(
    expr: str,
    values: dict[str, float],
    depth_um: np.ndarray,
    *,
    feature_order: list[str] | None = None,
) -> np.ndarray:
    X = _feature_vector(values, depth_um, feature_order=feature_order)
    return eval_sympy_expr(expr, X)


def _predict_r_sheet(expr: str, athena: dict[str, float]) -> float:
    X = np.zeros((1, len(ATHENA_FEATURES)))
    for j, col in enumerate(ATHENA_FEATURES):
        v = float(athena[col])
        if col in LOG_IN_MODEL:
            v = np.log(max(v, 1e-30))
        X[0, j] = v
    return float(eval_sympy_expr(expr, X)[0])


def _median_defaults() -> dict[str, float]:
    df = load_raw_dataframe()
    out: dict[str, float] = {}
    for col in MODEL2_FEATURES:
        s = df[col].astype(float)
        s = s[np.isfinite(s) & (s > 0)]
        out[col] = float(s.median()) if not s.empty else 1.0
    return out


def _load_reference_curve(profile_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = LIT_CURVES_DIR / f"{profile_id}_doping_curve.csv"
    df = pd.read_csv(path)
    return df["depth_um"].values.astype(float), df["N_cm3"].values.astype(float)


def extract_all_descriptors(
    N_floor: float = N_FLOOR_DEFAULT,
    *,
    depth_offset_um: float = 0.0,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    parsed = load_parsed_table()
    for _, meta in parsed.iterrows():
        pid = meta["profile_id"]
        z, N = _load_reference_curve(pid)
        z_phys = z + float(depth_offset_um)
        desc = extract_doping_descriptors(z_phys, N, N_floor=N_floor)
        desc["profile_id"] = pid
        desc["doping_R_sheet"] = meta["R_sheet_measured"]
        desc["N_p_table"] = meta["N_p_table"]
        desc["z_p_table"] = meta["z_p_table"]
        rows.append(desc)
    df = pd.DataFrame(rows)
    out = out_dir if out_dir is not None else BASE_OUT_DIR / "data"
    save_csv(df, out / "literature_curve_descriptors.csv")
    return df


def _fit_vector(
    free_names: list[str],
    x_free: np.ndarray,
    fixed: dict[str, float],
    bounds_map: dict[str, tuple[float, float]],
) -> dict[str, float]:
    out = dict(fixed)
    for name, val in zip(free_names, x_free):
        lo, hi = bounds_map.get(name, (val * 0.5, val * 1.5))
        out[name] = float(np.clip(val, lo, hi))
    return out


def fit_curve_shape(
    profile_id: str,
    variant: CurveVariant,
    *,
    use_curve_descriptors: bool = True,
    descriptor_mode: str = "anchors",
    r_sheet_athena: dict[str, float] | None = None,
) -> dict:
    """Optimize free features so the SR curve matches literature N(z).

    Literature depth is mapped to SR/physical depth via ``variant.depth_offset_um``:
    adaptive variants use 0 (literature depth already physical), legacy_tail uses
    0.25 um (literature depth 0 == physical 0.25 um).

    descriptor_mode:
      - ``anchors``: fix Table-1 N_p/z_p + known process; optimize missing process + other descriptors.
      - ``full``: fix all descriptors numerically extracted from the reference curve.
    """
    N_floor = variant.n_floor
    parsed = load_parsed_table().set_index("profile_id").loc[profile_id]
    z_ref, N_ref = _load_reference_curve(profile_id)
    defaults = _median_defaults()
    bounds_map = dataset_feature_bounds()
    known = dict(parsed["known_params"])

    fixed: dict[str, float] = dict(known)
    z_all, N_all = _load_reference_curve(profile_id)
    z_all_phys = z_all + variant.depth_offset_um
    desc = extract_doping_descriptors(z_all_phys, N_all, N_floor=N_floor)

    # Feature order used by this variant's SR formula (controls which descriptors
    # are inputs). doping_R_sheet is excluded for legacy_tail (18-dim training).
    feat_order = variant.feature_order
    feat_cols = feat_order[:-1]  # all but depth_um
    has_r_sheet = "doping_R_sheet" in feat_cols

    if use_curve_descriptors:
        if descriptor_mode == "full":
            for k in DOPING_DESCRIPTORS:
                if k == "doping_R_sheet" and not has_r_sheet:
                    continue
                if k in desc and np.isfinite(desc[k]):
                    fixed[k] = float(desc[k])
        else:
            # Anchor peak position/height to Table 1; let tail shape descriptors float.
            fixed["doping_N_peak"] = float(parsed["N_p_table"])
            fixed["doping_x_peak"] = float(parsed["z_p_table"])
            for k, v in desc.items():
                if k in defaults and np.isfinite(v):
                    defaults[k] = float(v)

    # Fit window: literature coords [0, fit_depth_max_um] (== physical depth window
    # after applying offset), and N >= floor.
    z_max_lit = variant.fit_depth_max_um
    mask = (z_ref >= 0.0) & (z_ref <= z_max_lit + 1e-9) & (N_ref >= N_floor)
    if not np.any(mask):
        return {
            "profile_id": profile_id,
            "variant": variant.label,
            "method": f"curve_shape_de_{descriptor_mode}",
            "error": "no points in SR fit window",
            "n_depth_fit": 0,
        }
    z_fit = z_ref[mask]
    ln_ref = np.log(np.maximum(N_ref[mask], 1e-30))
    z_fit_sr = z_fit + variant.depth_offset_um

    free_names = [c for c in feat_cols if c not in fixed]
    if not free_names:
        free_names = list(DEFECT_DESCRIPTORS)

    x0 = np.array([defaults.get(n, 1.0) for n in free_names], dtype=float)
    if r_sheet_athena:
        for i, n in enumerate(free_names):
            if n in r_sheet_athena:
                x0[i] = float(r_sheet_athena[n])
    bnds = []
    for i, n in enumerate(free_names):
        lo, hi = bounds_map.get(n, (max(x0[i] * 0.1, 1e-12), x0[i] * 10.0))
        if n in desc and np.isfinite(desc.get(n, float("nan"))):
            c = float(desc[n])
            lo = max(lo, c * 0.25)
            hi = min(hi, c * 4.0)
            if lo >= hi:
                lo, hi = bounds_map.get(n, (c * 0.25, c * 4.0))
        bnds.append((lo, hi))

    expr = _best_expr(variant.formula_path)

    def objective(x: np.ndarray) -> float:
        vals = _fit_vector(free_names, x, fixed, bounds_map)
        try:
            ln_pred = _predict_ln_curve(expr, vals, z_fit_sr, feature_order=feat_order)
        except Exception:
            return 1e12
        if not np.all(np.isfinite(ln_pred)):
            return 1e12
        return float(np.mean((ln_pred - ln_ref) ** 2))

    res = differential_evolution(
        objective,
        bounds=bnds,
        seed=42,
        maxiter=300,
        polish=True,
        tol=1e-6,
        workers=1,
    )
    best_vals = _fit_vector(free_names, res.x, fixed, bounds_map)
    ln_pred = _predict_ln_curve(expr, best_vals, z_fit_sr, feature_order=feat_order)
    if not np.all(np.isfinite(ln_pred)):
        # Optimizer landed on divergent params; report gracefully instead of inf/NaN.
        return {
            "profile_id": profile_id,
            "variant": variant.label,
            "method": f"curve_shape_de_{descriptor_mode}",
            "error": "SR prediction diverged (non-finite ln_pred)",
            "n_depth_fit": int(len(z_fit)),
            "curve_r2_log": float("nan"),
            "curve_rmse_cm3": float("nan"),
            "free_params": free_names,
            "optimized_params": {k: best_vals[k] for k in free_names},
            "fixed_params": {k: fixed[k] for k in fixed},
            "known_from_literature": known,
        }
    if len(ln_ref) < 2 or np.std(ln_ref) < 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum((ln_pred - ln_ref) ** 2) / np.sum((ln_ref - ln_ref.mean()) ** 2))
    rmse = float(np.sqrt(np.mean((np.exp(ln_pred) - N_ref[mask]) ** 2)))

    return {
        "profile_id": profile_id,
        "variant": variant.label,
        "method": f"curve_shape_de_{descriptor_mode}",
        "curve_rmse_cm3": rmse,
        "curve_r2_log": r2,
        "n_depth_fit": int(len(z_fit)),
        "depth_min_eff_um": 0.0,
        "depth_max_eff_um": variant.fit_depth_max_um,
        "depth_offset_um": variant.depth_offset_um,
        "n_floor": variant.n_floor,
        "free_params": free_names,
        "optimized_params": {k: best_vals[k] for k in free_names},
        "fixed_params": {k: fixed[k] for k in fixed},
        "known_from_literature": known,
    }


def fit_r_sheet(
    profile_id: str,
    target_r: float | None = None,
) -> dict:
    parsed = load_parsed_table().set_index("profile_id").loc[profile_id]
    target = float(target_r if target_r is not None else parsed["R_sheet_measured"])
    defaults = _median_defaults()
    bounds_map = dataset_feature_bounds()
    known = dict(parsed["known_params"])
    fixed = {**{k: defaults[k] for k in ATHENA_FEATURES}, **known}
    free_names = [c for c in ATHENA_FEATURES if c not in fixed]
    if not free_names:
        free_names = ["athena_c_boron", "athena_thick"]

    x0 = np.array([fixed.get(n, defaults[n]) for n in free_names], dtype=float)
    bnds = [bounds_map.get(n, (max(x0[i] * 0.1, 1e-6), x0[i] * 10.0)) for i, n in enumerate(free_names)]
    expr = _best_expr(R_SHEET_SR)

    def objective(x: np.ndarray) -> float:
        athena = {c: fixed[c] for c in ATHENA_FEATURES if c in fixed}
        for name, val in zip(free_names, x):
            athena[name] = float(val)
        try:
            pred = _predict_r_sheet(expr, athena)
        except Exception:
            return 1e12
        if not np.isfinite(pred):
            return 1e12
        return (pred - target) ** 2

    res = differential_evolution(objective, bounds=bnds, seed=42, maxiter=80, polish=True)
    athena = {c: fixed[c] for c in ATHENA_FEATURES if c in fixed}
    for name, val in zip(free_names, res.x):
        athena[name] = float(val)
    pred = _predict_r_sheet(expr, athena)

    return {
        "profile_id": profile_id,
        "method": "r_sheet_de",
        "R_sheet_target": target,
        "R_sheet_pred": pred,
        "R_sheet_rel_err_pct": 100.0 * abs(pred - target) / max(target, 1e-9),
        "free_params": free_names,
        "optimized_athena": {k: athena[k] for k in free_names},
        "known_from_literature": known,
    }


def plot_curve_comparison(profile_id: str, fit_result: dict, variant: CurveVariant, out_dir: Path) -> Path:
    if "error" in fit_result:
        return out_dir / "plots" / f"curve_fit_{profile_id}.png"
    z_ref, N_ref = _load_reference_curve(profile_id)
    vals = {**fit_result.get("fixed_params", {}), **fit_result.get("optimized_params", {})}
    expr = _best_expr(variant.formula_path)
    feat_order = variant.feature_order

    z_max = min(variant.fit_depth_max_um, float(z_ref.max()))
    z_plot = np.linspace(0.0, z_max, 300)
    z_plot_sr = z_plot + variant.depth_offset_um
    ln_pred = _predict_ln_curve(expr, vals, z_plot_sr, feature_order=feat_order)
    ln_pred = np.where(np.isfinite(ln_pred), ln_pred, np.nan)
    N_pred = np.exp(np.clip(ln_pred, -700, 700))

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    show = (z_ref >= 0.0) & (z_ref <= z_max + 1e-9) & (N_ref >= variant.n_floor)
    ax.semilogy(z_ref[show], N_ref[show], "o", ms=3, label="Literature (double Gaussian)", alpha=0.7)
    r2 = fit_result.get("curve_r2_log", float("nan"))
    ax.semilogy(z_plot, N_pred, "-", lw=2, label=f"SR fit (R²_log={r2:.3f})")
    ax.axhline(variant.n_floor, color="0.5", ls="--", lw=0.8)
    xlabel = "Depth (um, rel. to 0.25 um)" if variant.depth_offset_um > 0 else "Depth (um, physical)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Boron concentration (cm-3)")
    ax.set_title(f"{profile_id} [{variant.label}]: literature vs SR curve (0~{z_max:.2f} um)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = out_dir / "plots" / f"curve_fit_{profile_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def run_all(variant_label: str = DEFAULT_VARIANT) -> list[Path]:
    variant = VARIANTS[variant_label]
    out_dir = BASE_OUT_DIR / variant_label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    written: list[Path] = []
    desc_df = extract_all_descriptors(
        N_floor=variant.n_floor,
        depth_offset_um=variant.depth_offset_um,
        out_dir=out_dir,
    )
    written.append(out_dir / "data" / "literature_curve_descriptors.csv")

    curve_rows: list[dict] = []
    rs_rows: list[dict] = []
    parsed = load_parsed_table()

    for _, row in parsed.iterrows():
        pid = row["profile_id"]
        rs_athena: dict[str, float] | None = None
        try:
            rs = fit_r_sheet(pid)
            rs_rows.append(rs)
            known_athena = dict(parsed.set_index("profile_id").loc[pid]["known_params"])
            rs_athena = {**known_athena, **rs.get("optimized_athena", {})}
        except Exception as exc:
            rs_rows.append({"profile_id": pid, "error": str(exc)})

        try:
            cr = fit_curve_shape(pid, variant, r_sheet_athena=rs_athena)
            curve_rows.append(cr)
            if "error" not in cr:
                written.append(plot_curve_comparison(pid, cr, variant, out_dir))
        except Exception as exc:
            curve_rows.append({"profile_id": pid, "variant": variant_label, "error": str(exc)})

    curve_path = out_dir / "data" / "curve_shape_fit_summary.csv"
    rs_path = out_dir / "data" / "r_sheet_fit_summary.csv"
    save_csv(pd.DataFrame(curve_rows), curve_path)
    save_csv(pd.DataFrame(rs_rows), rs_path)
    written.extend([curve_path, rs_path])

    # merge feasibility table
    feas = parsed.merge(desc_df, on="profile_id", how="left")
    feas.to_csv(out_dir / "data" / "benchmark_feasibility_merged.csv", index=False)
    written.append(out_dir / "data" / "benchmark_feasibility_merged.csv")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit SR doping-curve formulas to literature profiles.",
    )
    parser.add_argument(
        "--curve-formula",
        choices=list(VARIANTS.keys()),
        default=DEFAULT_VARIANT,
        help="Which SR curve formula variant to benchmark.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three variants sequentially.",
    )
    args = parser.parse_args(argv)

    labels = list(VARIANTS.keys()) if args.all else [args.curve_formula]
    for label in labels:
        print(f"=== Variant: {label} ===")
        paths = run_all(label)
        for p in paths:
            print(p)


if __name__ == "__main__":
    main()
