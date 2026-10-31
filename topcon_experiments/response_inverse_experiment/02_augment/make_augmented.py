"""Build the augmentation rows in the exact schema of the 10,159-row dataset.

Rather than re-implementing the post-processing, this calls the generator's own
functions from `topcon_datagen_main.py` (path configured as `GEN` below)
(`_parse_str_1d`, `_build_doping_curve`, `_build_defect_curves`, `_doping_descriptors`,
`_curve_descriptors`, `_replace_one_with_nearby_max`).  That module - not the copy in
`se-TOPCon` - is the one that actually wrote `saomiao3csv`.

Verified on three archived samples: every descriptor reproduces the stored value to
4 significant figures and `n_points` matches exactly (see `descriptor_validation.csv`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TCAD = HERE.parent / "01_tcad"
# Paths are placeholders on purpose.
GEN = Path("<PATH>")           # directory that holds topcon_datagen_main.py + analyze_scans.py

sys.path.insert(0, str(TCAD))
sys.path.insert(0, str(GEN))
import tcad_lib as T                     # noqa: E402
from analyze_scans import compute_metrics, read_iv_from_csv   # noqa: E402

# tcad_lib puts the archived (withdrawn) generator on sys.path, which shadows the
# authoritative one, so load the real module by file location.
import importlib.util as _ilu            # noqa: E402
_spec = _ilu.spec_from_file_location("gen_main", GEN / "topcon_datagen_main.py")
M = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(M)

CSV = Path("<PATH>")           # the 10,159-row dataset CSV
RESULTS = TCAD / "results_50.csv"
OUT = HERE / "topcon_dataset_v1_aug.csv"

COLS = ["file_base", "athena_thick", "athena_c_boron", "athena_temp1", "athena_time1",
        "athena_temp2", "athena_time2", "athena_F_N2", "athena_F_O2",
        "doping_N_peak", "doping_x_peak", "doping_junction_depth", "doping_FWHM",
        "doping_gradient_max", "doping_dose",
        "defect_vac_N_peak", "defect_vac_gradient_max", "defect_vac_dose",
        "iv_Voc", "iv_Isc", "iv_Jsc", "iv_Pm", "iv_Vm", "iv_Im", "iv_FF", "iv_Eff",
        "doping_n_points", "defect_n_points"]

def descriptors_from_str(str_path: Path) -> dict:
    """Exactly the generator's own post-processing, on one .str."""
    data = M._parse_str_1d(str_path)

    acc = np.asarray(data["acceptor_72"], dtype=float)
    bor = np.asarray(data["boron_5"], dtype=float)
    conc = np.maximum(np.abs(acc), np.abs(bor)).tolist()      # BSG included
    x, n = M._build_doping_curve(data["y_um"], conc)
    dop = M._doping_descriptors(x, n)

    y = M._normalize_cutline_y(np.asarray(data["y_um"], dtype=float))
    vac = np.asarray(data["vacancies_0"], dtype=float)
    inter = np.asarray(data["interstitials_1"], dtype=float)
    traps = np.asarray(data["traps_12"], dtype=float)
    m = np.isfinite(y) & (y >= M.CUTLINE_Y_MIN) & (y <= M.CUTLINE_Y_MAX)
    y, vac, inter, traps = y[m], vac[m], inter[m], traps[m]
    order = np.argsort(y)
    y, vac, inter, traps = y[order], vac[order], inter[order], traps[order]
    vac = M._replace_one_with_nearby_max(vac)        # BSG placeholder "1"
    inter = M._replace_one_with_nearby_max(inter)
    traps = M._replace_one_with_nearby_max(traps)
    xd, vc, _ic, _tc = M._build_defect_curves(y.tolist(), vac.tolist(),
                                             inter.tolist(), traps.tolist())
    dfc = M._curve_descriptors(xd, vc)

    return {
        "doping_N_peak": dop["N_peak"], "doping_x_peak": dop["x_peak"],
        "doping_junction_depth": dop["junction_depth"], "doping_FWHM": dop["FWHM"],
        "doping_gradient_max": dop["gradient_max"], "doping_dose": dop["dose"],
        "doping_n_points": int(x.size),
        "defect_vac_N_peak": dfc["N_peak"],
        "defect_vac_gradient_max": dfc["gradient_max"],
        "defect_vac_dose": dfc["dose"],
        "defect_n_points": int(xd.size),
    }


def validate_descriptors() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    lut = df.set_index(df.file_base.str.lower())
    cases = {"46zvur": "athena_boronscan_20260602_182056_46zvur",
             "1j8gan": "athena_boronscan_20260602_215531_1j8gan",
             "p0an1a": "athena_boronscan_20260602_215541_p0an1a"}
    keys = ["doping_N_peak", "doping_x_peak", "doping_junction_depth", "doping_FWHM",
            "doping_gradient_max", "doping_dose", "defect_vac_N_peak",
            "defect_vac_gradient_max", "defect_vac_dose",
            "doping_n_points", "defect_n_points"]
    rows = []
    for tag, base in cases.items():
        got = descriptors_from_str(T.WORK_ROOT / "e2e" / tag / "out.str")
        rec = {"sample": tag}
        for k in keys:
            ref = float(lut.loc[base, k]); g = float(got[k])
            rec[k + "_csv"] = ref
            rec[k + "_recalc"] = g
            rec[k + "_rel%"] = 100 * (g - ref) / ref if ref else np.nan
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "descriptor_validation.csv", index=False, float_format="%.6g")
    return t


def row_from_run(rec: dict) -> dict:
    cid = rec["cand_id"]
    work = TCAD / "runs" / cid
    out = {"file_base": cid}
    for p in ("athena_thick", "athena_c_boron", "athena_temp1", "athena_time1",
              "athena_temp2", "athena_time2", "athena_F_N2", "athena_F_O2"):
        out[p] = rec[p]
    out.update(descriptors_from_str(work / "out.str"))

    v, i = read_iv_from_csv(work / "v1.csv")
    m = compute_metrics(np.asarray(v, float), np.asarray(i, float), scale_i=1.16)
    out.update({
        "iv_Voc": float(m["Voc"]), "iv_Isc": float(m["Isc"]), "iv_Jsc": float(m["Jsc"]),
        "iv_Pm": float(m["Pm"]), "iv_Vm": float(m["Vm"]), "iv_Im": float(m["Im"]),
        "iv_FF": float(m["FF"]), "iv_Eff": float(m["Eff"]),
    })
    return out


def main() -> None:
    v = validate_descriptors()
    rel = v[[c for c in v.columns if c.endswith("_rel%")]]
    print("descriptor reproduction, |relative error| % (max over 3 samples):")
    print(rel.abs().max().round(4).to_string())
    dev = v[[c for c in v.columns if c.endswith(("_n_points_csv", "_n_points_recalc"))]]
    print("\npoint counts:", dev.to_dict("records"))

    res = pd.read_csv(RESULTS)
    res = res[res.status == "ok"]
    rows = [row_from_run(r) for r in res.to_dict("records")]
    aug = pd.DataFrame(rows)[COLS]
    aug.to_csv(OUT, index=False, float_format="%.6g")
    print(f"\nwrote {OUT.name}: {len(aug)} rows x {len(COLS)} cols")
    print(aug[["file_base", "athena_thick", "athena_c_boron", "iv_Eff",
               "doping_x_peak", "doping_N_peak", "defect_vac_N_peak"]].head(6).round(4).to_string(index=False))
    print(f"\nPCE {aug.iv_Eff.min():.2f} - {aug.iv_Eff.max():.2f}, median {aug.iv_Eff.median():.2f}, "
          f">=26.5: {(aug.iv_Eff >= 26.5).sum()}/{len(aug)}")


if __name__ == "__main__":
    main()
