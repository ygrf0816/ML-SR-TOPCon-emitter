"""Preprocess doping and defect curves for curve SR (subset used by PySR)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topcon_experiments.common.data import load_raw_dataframe
from topcon_experiments.common.log_utils import log, log_step, setup_runtime
from topcon_experiments.common.plot_utils import save_csv
from topcon_experiments.config import (
    DEFECT_CURVE_DIR,
    DOPING_CURVE_DIR,
    OUTPUT_ROOT,
    SR_MAX_FILES,
)
from topcon_experiments.exp4_symbolic.curve_utils import process_curve_file

EXP4_OUT = OUTPUT_ROOT / "exp4_symbolic"


def _build_curve_path_index(curve_dir: Path) -> dict[str, Path]:
    """Map lowercase file_base -> txt path (one scan of curve_dir)."""
    index: dict[str, Path] = {}
    for p in curve_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".txt":
            continue
        stem = p.stem.lower()
        index[stem] = p
        if stem.endswith("_trap_vacancies"):
            base = stem[: -len("_trap_vacancies")]
            index.setdefault(base, p)
    return index


def _resolve_curve_path(index: dict[str, Path], file_base: str) -> Path | None:
    key = file_base.lower()
    if key in index:
        return index[key]
    trap_key = f"{key}_trap_vacancies"
    return index.get(trap_key)


def process_all_curves(
    curve_type: str,
    *,
    max_files: int | None = SR_MAX_FILES,
) -> pd.DataFrame:
    """Process curves for SR; default only first ``max_files`` samples."""
    df = load_raw_dataframe()
    if curve_type == "doping":
        curve_dir = DOPING_CURVE_DIR
        if not curve_dir.exists():
            raise FileNotFoundError(
                f"Doping curve directory not found: {curve_dir}. "
                "Please ensure saomiao3csv is available under new_data/."
            )
    else:
        curve_dir = DEFECT_CURVE_DIR

    log(f"Building path index for {curve_type} from {curve_dir} ...")
    path_index = _build_curve_path_index(curve_dir)
    log(f"  indexed {len(path_index)} curve file keys")

    file_bases = df["file_base"].astype(str).tolist()
    if max_files is not None:
        file_bases = file_bases[: int(max_files)]
    total = len(file_bases)
    log(f"Processing {total} {curve_type} curves (max_files={max_files}) ...")

    rows: list[dict] = []
    missing = 0
    for i, file_base in enumerate(file_bases, start=1):
        path = _resolve_curve_path(path_index, file_base)
        if path is None:
            missing += 1
            if i <= 5 or i % 20 == 0 or i == total:
                log_step(i, total, f"skip missing {curve_type}: {file_base}")
            continue
        grid, raw_grid, fitted_grid = process_curve_file(path, curve_type=curve_type)
        for j in range(len(grid)):
            rows.append({
                "file_base": file_base,
                "curve_type": curve_type,
                "depth_um": float(grid[j]),
                "value_raw": float(raw_grid[j]),
                "value_fitted": float(fitted_grid[j]),
            })
        if i <= 3 or i % 10 == 0 or i == total:
            log_step(
                i, total,
                f"{curve_type} {file_base}: {len(grid)} pts, depth_max={grid[-1]:.4f} um",
            )

    if missing:
        log(f"Warning: {missing}/{total} samples missing {curve_type} curve files")
    out = pd.DataFrame(rows)
    log(
        f"{curve_type} done: {out['file_base'].nunique()} samples, "
        f"{len(out)} rows, depth_max={out['depth_um'].max():.4f} um"
    )
    return out


def process_and_save_curves(
    curve_type: str,
    *,
    max_files: int | None = SR_MAX_FILES,
) -> pd.DataFrame:
    EXP4_OUT.mkdir(parents=True, exist_ok=True)
    out_path = EXP4_OUT / f"curve_processed_{curve_type}.csv"
    df = process_all_curves(curve_type, max_files=max_files)
    save_csv(df, out_path)
    log(f"Saved {out_path}")
    return df


def main() -> None:
    setup_runtime()
    EXP4_OUT.mkdir(parents=True, exist_ok=True)
    log(f"Curve preprocess -> {EXP4_OUT} (SR_MAX_FILES={SR_MAX_FILES})")

    process_and_save_curves("doping")
    process_and_save_curves("defect")
    log("All curve preprocessing finished.")


if __name__ == "__main__":
    main()
