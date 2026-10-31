"""Shared TCAD helpers for the R1-4 active-learning batch.

Kept deliberately thin: Athena deck -> .str -> profile -> Atlas.

Two details that cost time to find, both verified against the archived generator
(`DATAGEN` below):

* The `models fermi pair.diff` line in the original deck is Atlas syntax.  In the
  original workflow the deck was fed to Athena line by line over a pipe, so the
  resulting SYNTAX ERROR was ignored and the run continued.  Fed as a batch file
  it aborts the run, so it is dropped here.  It has no effect on the profile.
* `parse_str_mesh` in `cutline_extract.py` keeps only region 3 (silicon).  The BSG
  layer lives in region 1 *above* the silicon and carries `c.boron`; dropping it
  removes the whole high-concentration plateau that the manuscript uses as the
  ohmic-contact approximation.  `read_str_profile` below keeps every region.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Paths are placeholders on purpose. Fill them in for your own installation.
DATAGEN = Path("<PATH>")       # directory that holds analyze_scans.py
DECK_SRC = Path("<PATH>")      # one archived Atlas run: V1.in plus the .nk / .lib files
TCAD_EXE = Path("<PATH>")      # Silvaco binaries (athena.exe, atlas.exe)
WORK_ROOT = Path("<PATH>")     # scratch directory for the batch runs

sys.path.insert(0, str(DATAGEN))
from analyze_scans import compute_metrics, read_iv_from_csv   # noqa: E402

PAR = ["athena_thick", "athena_c_boron", "athena_temp1", "athena_time1",
       "athena_temp2", "athena_time2", "athena_F_N2", "athena_F_O2"]
KEY = dict(zip(PAR, ["thick", "c_boron", "temp1", "time1", "temp2", "time2", "F_N2", "F_O2"]))

# substrate floor and depth window used when the dataset curve was written
DOPING_FLOOR = 2e15
CURVE_Y_MAX = 5.0

NKDIR_FILES = ("template.lib", "mgf2_nk.nk", "ito_nk.nk", "al2o3_nk.nk",
               "si_nk.nk", "ag_nk.nk", "polySi_nk.nk", "sinx_nk.nk")


# --------------------------------------------------------------------------- #
# Athena
# --------------------------------------------------------------------------- #
def build_athena_deck(params: dict, outstr: str, thick_fmt: str = ".4f") -> list[str]:
    return [
        "line x loc=0.0 spacing=0.1",
        "line x loc=0.5 spacing=0.1",
        "line y loc=0.0   spacing=0.001",
        "line y loc=0.2   spacing=0.001",
        "line y loc=0.8   spacing=0.005",
        "line y loc=2.0   spacing=0.01",
        "init silicon orientation=100 phosphor resistivity=3",
        "method full.cpl",
        f"deposit oxide thick={format(params['thick'], thick_fmt)} "
        f"c.boron={params['c_boron']:.6e}",
        f"diffuse time={params['time1']:.4f} temp={params['temp1']:.4f} press=1 nitrogen",
        f"diffuse time={params['time2']:.4f} temp={params['temp2']:.4f} "
        f"press=1 F.N2={params['F_N2']:.4f} F.O2={params['F_O2']:.4f}",
        "structure outfile=out.str",
        "quit",
    ]


def run_athena(work: Path, params: dict, thick_fmt: str = ".4f", timeout: int = 900) -> Path:
    """Run Athena in `work` (must be an ASCII path) and return the .str path."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "athena.in").write_text(
        "\n".join(build_athena_deck(params, "out.str", thick_fmt)) + "\n", encoding="utf-8")
    log = work / "athena_stdout.txt"
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run([str(TCAD_EXE / "athena.exe"), "athena.in"], cwd=work,
                            stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
    out = work / "out.str"
    if rc != 0 or not out.exists():
        raise RuntimeError(f"athena failed rc={rc} in {work}")
    return out


# --------------------------------------------------------------------------- #
# .str -> doping curve (all regions, BSG included)
# --------------------------------------------------------------------------- #
def read_str_profile(str_path: Path) -> list[dict]:
    lines = Path(str_path).read_text(encoding="utf-8", errors="replace").splitlines()

    coord: dict[int, tuple[float, float]] = {}
    for ln in lines:
        t = ln.split()
        if t and t[0] == "c" and len(t) >= 4:
            coord[int(t[1])] = (float(t[2]), float(t[3]))

    s_line = next((ln for ln in lines if ln.startswith("s ")), None)
    if s_line is None:
        raise ValueError(f"no 's' line in {str_path}")
    qids = [int(x) for x in s_line.split()[2:]]
    if qids and qids[0] == 513:
        qids = qids[1:]

    rows: list[dict] = []
    for ln in lines:
        t = ln.split()
        if not t or t[0] != "n" or len(t) < 6:
            continue
        node = int(t[1])
        if node not in coord:
            continue
        vals = [float(v) for v in t[4:]]
        if len(vals) != len(qids):
            continue
        d = dict(zip(qids, vals))
        x, y = coord[node]
        rows.append({
            "x": x, "y": y, "region": int(t[2]),
            "boron": d.get(5, 0.0) or 0.0,          # qid 5  = Boron
            "phos": d.get(3, 0.0) or 0.0,           # qid 3  = Phosphorus
            "acc": max(d.get(71, 0.0) or 0.0, d.get(72, 0.0) or 0.0),
            "vac": d.get(0, float("nan")),          # qid 0  = Vacancies
            "inter": d.get(1, float("nan")),        # qid 1  = Interstitials
        })
    return rows


def curve_from_str(str_path: Path, y_max: float = CURVE_Y_MAX,
                   floor: float = DOPING_FLOOR) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the dataset's doping curve: all regions, y shifted so min(y)=0,
    net = max(|P|,|acceptor|,|B|), floored at 2e15, sorted, de-duplicated."""
    rows = read_str_profile(str_path)
    if not rows:
        return np.array([]), np.array([])
    y = np.array([r["y"] for r in rows])
    n = np.array([max(abs(r["phos"]), abs(r["acc"]), abs(r["boron"])) for r in rows])
    xs = y - y.min()
    m = (xs >= 0) & (xs <= y_max) & (n > 0)
    xs, n = xs[m], np.maximum(n[m], floor)
    order = np.argsort(xs)
    xs, n = xs[order], n[order]

    ux, un = [], []
    i = 0
    while i < len(xs):
        j, v = i + 1, n[i]
        while j < len(xs) and xs[j] == xs[i]:
            v = max(v, n[j])
            j += 1
        ux.append(xs[i])
        un.append(v)
        i = j
    return np.asarray(ux), np.asarray(un)


def write_curve(x: np.ndarray, n: np.ndarray, out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8", newline="") as f:
        for xx, nn in zip(x, n):
            f.write(f"{float(xx):.15g}\t{float(nn):.8e}\n")


# --------------------------------------------------------------------------- #
# Atlas
# --------------------------------------------------------------------------- #
def make_atlas_deck(work: Path, curve_name: str = "doping.txt", fine: bool = False) -> Path:
    import re
    deck = (DECK_SRC / "V1.in").read_text(encoding="utf-8")
    deck = re.sub(r'infile="[^"]*"', f'infile="{curve_name}"', deck, count=1)
    if fine:
        deck = deck.replace("SOLVE name=anode vstep=0.100 vfinal=0.400",
                            "SOLVE name=anode vstep=0.010 vfinal=0.400")
        deck = deck.replace("SOLVE name=anode vstep=0.050 vfinal=0.800",
                            "SOLVE name=anode vstep=0.005 vfinal=0.800")
        deck = deck.replace("csvfile=v1.csv", "csvfile=v1_fine.csv")
        deck = deck.replace("outfile=v1.log", "outfile=v1_fine.log")
    p = work / ("run_fine.in" if fine else "run.in")
    p.write_text(deck, encoding="utf-8")
    return p


def run_atlas(work: Path, curve_name: str = "doping.txt", fine: bool = False) -> dict:
    for f in NKDIR_FILES:
        dst = work / f
        if not dst.exists():
            shutil.copy2(DECK_SRC / f, dst)
    deck = make_atlas_deck(work, curve_name, fine)
    log = work / ("atlas_fine.txt" if fine else "atlas_stdout.txt")
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run([str(TCAD_EXE / "atlas.exe"), deck.name], cwd=work,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    iv = work / ("v1_fine.csv" if fine else "v1.csv")
    if rc != 0 or not iv.exists():
        return {"status": f"atlas_failed rc={rc}"}
    v, i = read_iv_from_csv(iv)
    v = np.asarray(v, float)
    i = np.asarray(i, float)
    m = compute_metrics(v, i, scale_i=1.16)
    m0 = compute_metrics(v, i, scale_i=1.0)
    out = {"status": "ok", "n_iv": len(v), "Eff_raw": float(m0["Eff"])}
    out.update({k: float(m[k]) for k in ("Voc", "Jsc", "FF", "Eff")})
    return out
