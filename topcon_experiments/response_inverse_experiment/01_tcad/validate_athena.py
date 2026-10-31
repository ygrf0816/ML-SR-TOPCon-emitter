"""Replay a known dataset sample to validate the reconstructed Athena pipeline.

If the reconstructed deck (5-decimal BSG thickness) is faithful, the regenerated
doping curve must coincide with the stored one, and the Atlas run must reproduce
the dataset label to a few hundredths of a percentage point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tcad_lib as T  # noqa: E402

# Paths are placeholders on purpose.
DATA = Path("<PATH>")          # the 10,159-row dataset CSV
CURVES = Path("<PATH>")        # directory of archived doping curves (*.txt)
BASE = "athena_boronscan_20260602_182056_46zvur"
OUT = HERE / "validate" / BASE

def main() -> None:
    df = pd.read_csv(DATA)
    row = df[df.file_base == BASE].iloc[0]
    params = {T.KEY[p]: float(row[p]) for p in T.PAR}
    print("params:", {k: f"{v:.6g}" for k, v in params.items()})
    OUT.mkdir(parents=True, exist_ok=True)

    str_path = OUT / f"{BASE}.str"
    if not str_path.exists():
        T.run_athena(params, str_path)
    print("str:", str_path.exists())

    x, n = T.curve_from_str(str_path)
    T.write_curve(x, n, OUT / "doping.txt")

    ref = np.loadtxt(CURVES / f"{BASE}.txt")
    print("regenerated n=%d  stored n=%d" % (len(x), len(ref)))
    grid = np.linspace(max(x.min(), ref[:, 0].min()), min(0.5, x.max(), ref[:, 0].max()), 400)
    a = np.interp(grid, x, n)
    b = np.interp(grid, ref[:, 0], ref[:, 1])
    rel = np.abs(a - b) / np.maximum(b, 1e-30)
    print("curve agreement over 0-0.5 um: median rel.dev %.3f%%  p95 %.3f%%  max %.3f%%"
          % (100 * np.median(rel), 100 * np.percentile(rel, 95), 100 * rel.max()))

    res = T.run_atlas(OUT, "doping.txt", fine=False)
    print("Atlas:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    print("dataset label: Voc %.4f  Jsc %.3f  FF %.3f  Eff %.3f"
          % (row.iv_Voc, row.iv_Jsc, row.iv_FF, row.iv_Eff))

if __name__ == "__main__":
    main()
