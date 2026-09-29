"""
清理历史仿真数据，便于在扩展参数后重新收数。

默认删除（保留目录本身）：
- saomiao3/*.str, se_scan.jsonl, debug_*
- saomiao3csv/*.txt
- saomiao3jv/*.csv, *.log, iv_run_records.jsonl
- defect_exports 下 csv/txt/json（保留目录）
- dataset/topcon_dataset.csv
- 根目录调试残留 topcon_0.str, pipeline_run*.log

用法：
  python cleanup_old_data.py          # 预览将删除的文件
  python cleanup_old_data.py --yes    # 确认删除
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PATTERNS = [
    ROOT / "saomiao3" / "*.str",
    ROOT / "saomiao3" / "se_scan.jsonl",
    ROOT / "saomiao3" / "debug_*",
    ROOT / "saomiao3csv" / "*.txt",
    ROOT / "saomiao3jv" / "*.csv",
    ROOT / "saomiao3jv" / "*.log",
    ROOT / "saomiao3jv" / "iv_run_records.jsonl",
    ROOT / "defect_exports" / "*.csv",
    ROOT / "defect_exports" / "cutlines" / "*",
    ROOT / "defect_exports" / "trap_curves" / "*",
    ROOT / "dataset" / "topcon_dataset.csv",
    ROOT / "topcon_0.str",
    ROOT / "pipeline_run.log",
    ROOT / "pipeline_run2.log",
]


def collect_targets() -> list[Path]:
    found: list[Path] = []
    for pat in PATTERNS:
        parent = pat.parent
        if "*" in pat.name:
            if parent.is_dir():
                found.extend(sorted(parent.glob(pat.name)))
        elif pat.exists():
            found.append(pat)
    # 去重
    uniq = []
    seen = set()
    for p in found:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def main() -> None:
    parser = argparse.ArgumentParser(description="清理旧仿真数据")
    parser.add_argument("--yes", action="store_true", help="确认执行删除")
    args = parser.parse_args()

    targets = collect_targets()
    if not targets:
        print("[INFO] 没有需要清理的文件。")
        return

    print(f"[INFO] 将删除 {len(targets)} 个文件/条目：")
    for p in targets:
        print(f"  - {p.relative_to(ROOT)}")

    if not args.yes:
        print("\n预览模式。确认删除请运行: python cleanup_old_data.py --yes")
        return

    removed = 0
    for p in targets:
        try:
            if p.is_file():
                p.unlink()
                removed += 1
        except Exception as e:
            print(f"[WARN] 删除失败: {p} -> {e}")

    # 确保空目录仍在
    for d in ["saomiao3", "saomiao3csv", "saomiao3jv", "defect_exports/cutlines", "defect_exports/trap_curves", "dataset"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    print(f"[OK] 已删除 {removed} 个文件。可以重新开始收数。")


if __name__ == "__main__":
    main()
