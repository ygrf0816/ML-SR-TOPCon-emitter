"""End-to-end replay: Athena -> profile -> Atlas -> IV metrics vs the dataset label.

If the three samples below land on their stored efficiencies, the reconstruction is
faithful and the active-learning batch can use the same path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tcad_lib as T

# Paths are placeholders on purpose.
CSV = Path("<PATH>")           # the 10,159-row dataset CSV
ARCH = Path("<PATH>")          # directory of archived doping curves (*.txt)
SAMPLES = ["athena_boronscan_20260602_182056_46zvur",
           "athena_boronscan_20260602_215531_1j8gan",
           "athena_boronscan_20260602_215541_p0an1a"]


def main() -> None:
    df = pd.read_csv(CSV)
    lut = df.set_index(df.file_base.str.lower())
    rows = []
    for base in SAMPLES:
        r = lut.loc[base]
        params = {T.KEY[p]: float(r[p]) for p in T.PAR}
        tag = base.split("_")[-1]
        work = T.WORK_ROOT / "e2e" / tag
        s = T.run_athena(work, params, thick_fmt=".2f")
        x, n = T.curve_from_str(s)
        T.write_curve(x, n, work / "doping.txt")
        ref = np.loadtxt(ARCH / f"{base}.txt")
        g = np.linspace(0, min(x.max(), ref[-1, 0]), 900)
        a, b = np.interp(g, x, n), np.interp(g, ref[:, 0], ref[:, 1])
        cdev = 100 * np.median(np.abs(a - b) / np.maximum(b, 1e-30))
        m = T.run_atlas(work)
        rec = dict(base=tag, pce_label=float(r.iv_Eff), pce_tcad=m.get("Eff"),
                   d_pce=(m.get("Eff", np.nan) - float(r.iv_Eff)),
                   curve_dev_pct=cdev, n_iv=m.get("n_iv"), status=m.get("status"),
                   Voc=m.get("Voc"), Jsc=m.get("Jsc"), FF=m.get("FF"))
        rows.append(rec)
        print(f"{tag}: label {rec['pce_label']:.3f} | tcad {rec['pce_tcad']:.3f} "
              f"| dPCE {rec['d_pce']:+.3f} pp | curve dev {cdev:.3f}% | {rec['status']}")
    t = pd.DataFrame(rows)
    t.to_csv(T.HERE / "end2end_validation.csv", index=False, float_format="%.4f")
    print()
    print(t[["base", "pce_label", "pce_tcad", "d_pce", "curve_dev_pct"]].round(3).to_string(index=False))
    print("\nmax |dPCE| = %.3f pp" % t.d_pce.abs().max())


if __name__ == "__main__":
    main()
