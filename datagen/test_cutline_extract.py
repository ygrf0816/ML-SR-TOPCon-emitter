"""
简单测试：对已有 .str 用 cutline 方法提取轻/重掺剖面、复合曲线和工艺指标。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from cutline_extract import extract_composite_from_str

SCRIPT_DIR = Path(__file__).resolve().parent
STR_DIR = SCRIPT_DIR / "saomiao3"
OUT_DIR = SCRIPT_DIR / "defect_exports" / "cutlines"


def main() -> None:
    str_files = sorted(STR_DIR.glob("SE_Scan_*.str"))
    if not str_files:
        str_files = sorted(STR_DIR.glob("*.str"))
    if not str_files:
        print(f"[ERROR] 未找到 .str: {STR_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_file = str_files[0]
    print(f"[INFO] 测试文件: {test_file.name}")

    x_light = float(os.environ.get("CUTLINE_X_LIGHT", "50.0"))
    x_heavy = float(os.environ.get("CUTLINE_X_HEAVY", "7.5"))
    y_max = float(os.environ.get("CUTLINE_Y_MAX", "5.0"))

    result = extract_composite_from_str(test_file, x_light=x_light, x_heavy=x_heavy, y_max=y_max)
    base = test_file.stem

    for tag, profile in (("light", result["light"]), ("heavy", result["heavy"]), ("composite", result["composite"])):
        out_txt = OUT_DIR / f"{base}_{tag}_cutline.txt"
        with out_txt.open("w", encoding="utf-8", newline="") as f:
            for y, n in zip(profile["y_um"], profile["net_doping"]):
                f.write(f"{float(y):.15g}\t{float(n):.8e}\n")
        print(f"[OK] {tag} cutline -> {out_txt} ({len(profile['y_um'])} 点)")

    metrics = result["metrics"]
    print("[OK] 工艺指标 (cutline 计算):")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    metrics_path = OUT_DIR / f"{base}_process_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 指标 JSON -> {metrics_path}")


if __name__ == "__main__":
    main()
