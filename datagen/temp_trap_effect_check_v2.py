from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List

from analyze_scans import compute_metrics, read_iv_from_csv
from cd_txt_scan import run_one_with_cd


def parse_str_1d(path: Path) -> Dict[str, List[float]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    y_by_node: Dict[int, float] = {}
    for ln in lines:
        t = ln.split()
        if t and t[0] == "c" and len(t) >= 4:
            y_by_node[int(t[1])] = float(t[3])

    s_line = next((ln for ln in lines if ln.startswith("s ")), None)
    if s_line is None:
        raise ValueError(f"未找到 s 行: {path}")
    qids = list(map(int, s_line.split()[2:]))
    if qids and qids[0] == 513:
        qids = qids[1:]

    rows: List[Dict[int, float]] = []
    ys: List[float] = []
    for ln in lines:
        t = ln.split()
        if not t or t[0] != "n":
            continue
        node = int(t[1])
        y = y_by_node.get(node)
        if y is None or y < 0:
            continue
        vals = list(map(float, t[4:]))
        if len(vals) != len(qids):
            continue
        ys.append(y)
        rows.append({qid: v for qid, v in zip(qids, vals)})

    return {
        "y_um": ys,
        "acceptor_72": [r.get(72, float("nan")) for r in rows],
        "boron_5": [r.get(5, float("nan")) for r in rows],
        "vacancies_0": [r.get(0, float("nan")) for r in rows],
    }


def build_doping_curve_txt(data: Dict[str, List[float]], out_txt: Path) -> None:
    conc = data["acceptor_72"]
    if not any(math.isfinite(v) and abs(v) > 0 for v in conc):
        conc = data["boron_5"]
    pairs = sorted(
        (
            (float(y), abs(float(c)))
            for y, c in zip(data["y_um"], conc)
            if math.isfinite(y) and math.isfinite(c) and y >= 0 and c > 0
        ),
        key=lambda p: p[0],
    )
    with out_txt.open("w", encoding="utf-8", newline="") as f:
        for y, c in pairs:
            f.write(f"{y:.15g}\t{c:.8e}\n")


def build_trap_curve_txt(data: Dict[str, List[float]], out_txt: Path, scale: float) -> None:
    pairs = sorted(
        (
            (float(y), abs(float(v)) * scale)
            for y, v in zip(data["y_um"], data["vacancies_0"])
            if math.isfinite(y) and math.isfinite(v) and y >= 0 and abs(v) > 0
        ),
        key=lambda p: p[0],
    )
    with out_txt.open("w", encoding="utf-8", newline="") as f:
        for y, v in pairs:
            f.write(f"{y:.15g}\t{v:.8e}\n")


def read_metrics(csv_path: str) -> Dict[str, float]:
    v, i = read_iv_from_csv(csv_path)
    m = compute_metrics(v, i)
    return {k: float(m[k]) for k in ["Voc", "Jsc", "FF", "Eff"]}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    str_dir = root / "saomiao3"
    exe_dir = os.environ.get("TCAD_EXE_DIR", "D:/silvaco/exe")
    scales = [float(x) for x in os.environ.get("TRAP_SCALES", "1,1e6").split(",") if x.strip()]

    str_path_env = os.environ.get("ONE_STR_PATH", "").strip()
    if str_path_env:
        str_path = Path(str_path_env).resolve()
    else:
        cands = sorted(str_dir.glob("*.str"))
        if not cands:
            raise FileNotFoundError(f"未找到 .str 文件: {str_dir}")
        str_path = cands[0]

    out_dir = root / "quickcheck_trap_effect"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = str_path.stem

    data = parse_str_1d(str_path)
    dop_a = out_dir / f"{base}_A_doping.txt"
    dop_b = out_dir / f"{base}_B_doping.txt"
    build_doping_curve_txt(data, dop_a)
    build_doping_curve_txt(data, dop_b)

    rec_a = run_one_with_cd(str(dop_a), exe_dir, str(out_dir), defect_values=None)
    m_a = read_metrics(rec_a["outcsv"])
    print("=== Baseline ===")
    print(f"A CSV: {rec_a['outcsv']}")
    print(m_a)

    for scale in scales:
        trap_txt = out_dir / f"{base}_trap_vacancies_x{scale:g}.txt"
        build_trap_curve_txt(data, trap_txt, scale)
        rec_b = run_one_with_cd(
            str(dop_b),
            exe_dir,
            str(out_dir),
            defect_values={"defect_mode": "trap_curve", "trap_cdpath": str(trap_txt)},
        )
        m_b = read_metrics(rec_b["outcsv"])
        print(f"\n=== scale={scale:g} ===")
        print(f"B CSV: {rec_b['outcsv']}")
        for k in ["Voc", "Jsc", "FF", "Eff"]:
            print(f"{k}: A={m_a[k]:.8g}, B={m_b[k]:.8g}, Delta={m_b[k]-m_a[k]:.8g}")


if __name__ == "__main__":
    main()

