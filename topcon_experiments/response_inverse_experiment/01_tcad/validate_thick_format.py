"""Which `thick` precision reproduces the archived curves?

Two samples whose `thick` sits far from a 0.01 boundary are replayed with both
`:.2f` and `:.4f`.  The format that matches the archived .txt is the one the
generator used.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tcad_lib as T

# Paths are placeholders on purpose.
CSV = Path("<PATH>")           # the 10,159-row dataset CSV
ARCH = Path("<PATH>")          # directory of archived doping curves (*.txt)
SAMPLES = ["athena_boronscan_20260602_215531_1j8gan",
           "athena_boronscan_20260602_215541_p0an1a"]


def dev(a_x, a_n, b_x, b_n):
    lo, hi = max(a_x.min(), b_x.min()), min(a_x.max(), b_x.max())
    g = np.linspace(lo, hi, 900)
    a = np.interp(g, a_x, a_n)
    b = np.interp(g, b_x, b_n)
    rel = np.abs(a - b) / np.maximum(b, 1e-30)
    return 100 * np.median(rel), 100 * np.percentile(rel, 95), 100 * rel.max()


def main() -> None:
    df = pd.read_csv(CSV)
    lut = df.set_index(df.file_base.str.lower())
    out = T.WORK_ROOT / "fmt_test"
    rows = []
    for base in SAMPLES:
        r = lut.loc[base]
        params = {T.KEY[p]: float(r[p]) for p in T.PAR}
        ref = np.loadtxt(ARCH / f"{base}.txt")
        for fmt in (".2f", ".4f"):
            tag = base.split("_")[-1]
            work = out / f"{tag}_{fmt.strip('.')}"
            t0 = time.time()
            s = T.run_athena(work, params, thick_fmt=fmt)
            dt = time.time() - t0
            x, n = T.curve_from_str(s)
            med, p95, mx = dev(x, n, ref[:, 0], ref[:, 1])
            rows.append(dict(base=tag, thick=params["thick"], fmt=fmt, n=len(x),
                             med=med, p95=p95, max=mx, sec=dt))
            print(f"{tag}  thick={params['thick']:.5f} fmt={fmt}  n={len(x)}  "
                  f"median={med:.3f}%  p95={p95:.3f}%  max={mx:.3f}%  {dt:.0f}s")
    t = pd.DataFrame(rows)
    t.to_csv(T.HERE / "thick_format_test.csv", index=False, float_format="%.4f")
    print()
    print(t.groupby("fmt")[["med", "p95", "max"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
