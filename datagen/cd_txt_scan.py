"""
cd_txt_scan.py

用途:
- 基于 random_param_scan.py 的流程, 扫描指定目录(默认项目根下的 `saomiao3csv`)中的所有掺杂曲线 `.txt` 文件,
- 使用自定义曲线掺杂模型 `topcon_n_cd` 进行仿真(通过设置 `cdpath`),
- 将产生的 `.log/.str/.csv` 结果保存至项目根下的 `saomiao3jv` 目录。

新增行为:
- 若输出目录中已存在同名 `.csv`(与 `.txt` 文件名相同的基名), 则跳过该曲线, 避免重复计算。

注意:
- 本脚本为可直接在 IDE 中运行的形式, 不依赖命令行参数。
- 若找不到 `saomiao3csv`, 将依次尝试 `saomiao3.csv` 与 `saomiao3` 目录。
"""

import os
import sys
import time
import csv
import threading
from typing import Dict, List, Optional, Any

from python_tcad import pythontcad
import tcadmodel
from tqdm import tqdm

# 单次 IV 仿真超时（秒）；超时后终止 ATLAS 进程并跳过该样本，避免卡住整批
IV_TIMEOUT_SECONDS = int(os.environ.get("IV_TIMEOUT_SECONDS", "600"))

# 可选：缺陷映射表（CSV）。
# 为空表示使用 tcadmodel 默认缺陷参数；
# 非空时按 file_base 将缺陷参数注入每个样本的 IV 仿真。
DEFECT_MAP_CSV = os.environ.get("DEFECT_MAP_CSV", "").strip()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_txt_files(search_dir: str) -> List[str]:
    files: List[str] = []
    for root, _, fnames in os.walk(search_dir):
        for f in fnames:
            if f.lower().endswith(".txt"):
                files.append(os.path.join(root, f))
    files.sort()
    return files


def already_done(txt_path: str, out_dir: str, existing_csv_basenames: "set[str]") -> bool:
    """判断该 txt 是否已计算过: 规则为 out_dir 下存在同名 .csv。大小写不敏感。"""
    base_name = os.path.splitext(os.path.basename(txt_path))[0].lower()
    return base_name in existing_csv_basenames


def load_defect_map(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    读取缺陷映射表，键为 file_base(小写)。
    允许列（任意子集）：
    Nt_top_N, Nt_polySi_top, Nt_polySi_rear, Nt_Si_SiOx, Nt_SiOx_Poly,
    taun_Si, taun_polySi_rear_P, resist_rear,
    defect_mode, trap_cdpath
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    if not csv_path:
        return mapping
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"缺陷映射 CSV 不存在: {csv_path}")

    allowed = {
        "nt_top_n": ("Nt_top_N", "float"),
        "nt_polysi_top": ("Nt_polySi_top", "float"),
        "nt_polysi_rear": ("Nt_polySi_rear", "float"),
        "nt_si_siox": ("Nt_Si_SiOx", "float"),
        "nt_siox_poly": ("Nt_SiOx_Poly", "float"),
        "taun_si": ("taun_Si", "float"),
        "taun_polysi_rear_p": ("taun_polySi_rear_P", "float"),
        "resist_rear": ("resist_rear", "float"),
        "defect_mode": ("defect_mode", "str"),
        "trap_cdpath": ("trap_cdpath", "str"),
    }

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return mapping
        lower_headers = {h.lower(): h for h in reader.fieldnames}
        if "file_base" not in lower_headers:
            raise ValueError("缺陷映射 CSV 必须包含列: file_base")

        fb_col = lower_headers["file_base"]
        for row in reader:
            fb = str(row.get(fb_col, "")).strip().lower()
            if not fb:
                continue
            rec: Dict[str, Any] = {}
            for lk, (target_key, value_type) in allowed.items():
                if lk in lower_headers:
                    raw = str(row.get(lower_headers[lk], "")).strip()
                    if raw == "":
                        continue
                    if value_type == "float":
                        rec[target_key] = float(raw)
                    else:
                        rec[target_key] = raw
            if rec:
                mapping[fb] = rec
    return mapping


def terminate_tcad_processes(tcad: pythontcad) -> None:
    """终止 atlas/athena 子进程，避免超时后僵尸进程占用 license。"""
    for proc_name in ("atlas_process", "athena_process"):
        proc = getattr(tcad, proc_name, None)
        if proc is None or proc.poll() is not None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass


def run_atlas_simulation(tcad: pythontcad, timeout_seconds: int = IV_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """
    在指定超时内执行 ATLAS 命令列表。
    返回 {"success": bool, "error": str|None, "timed_out": bool}
    """
    result: Dict[str, Any] = {"success": False, "error": None, "timed_out": False}

    def _worker() -> None:
        try:
            tcad.tcad.updatecommand()
            tcad.updateprocess()
            tcad.run_atlas_start(output_state=False)
            for cmd in tcad.tcad.get_allcommandlist():
                tcad.run_atlas_command(cmd, output_state=False)
            result["success"] = True
        except Exception as exc:
            result["error"] = str(exc)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds)

    if worker.is_alive():
        result["timed_out"] = True
        result["error"] = f"IV 仿真超时（>{timeout_seconds}s）"
        terminate_tcad_processes(tcad)
        worker.join(timeout=3)
        return result

    if not result["success"] and result["error"] is None:
        result["error"] = "IV 仿真未完成"
    return result


def apply_defect_params(tcad_obj: Any, defect_values: Optional[Dict[str, Any]]) -> None:
    """将缺陷参数注入 topcon_n_cd 实例。"""
    if not defect_values:
        return
    for k, v in defect_values.items():
        if not hasattr(tcad_obj, k):
            continue
        # 支持数值参数与字符串参数（如 defect_mode/trap_cdpath）
        if isinstance(v, (int, float)):
            setattr(tcad_obj, k, float(v))
        else:
            setattr(tcad_obj, k, str(v))


def run_one_with_cd(
    txt_path: str,
    exe_dir: str,
    out_dir: str,
    defect_values: Optional[Dict[str, float]] = None,
    timeout_seconds: int = IV_TIMEOUT_SECONDS,
) -> Dict:
    """使用 topcon_n_cd 对单个曲线文件运行一次仿真, 返回记录字典。"""
    base_name = os.path.splitext(os.path.basename(txt_path))[0]
    file_tag = f"{base_name}"

    outlog = os.path.join(out_dir, f"{file_tag}.log")
    outstr = os.path.join(out_dir, f"{file_tag}.str")
    outcsv = os.path.join(out_dir, f"{file_tag}.csv")

    # 初始化仿真器
    a = pythontcad()
    if exe_dir:
        a.set_tcad(exe_dir)

    # 使用自定义曲线掺杂模型
    a.tcad = tcadmodel.topcon_n_cd()

    # nk 与模板路径: 设为当前脚本目录
    nk_tpl_dir = os.path.dirname(__file__)
    a.tcad.nkpath = nk_tpl_dir + os.sep
    a.tcad.template_lib = os.path.join(nk_tpl_dir, "template.lib")

    # 设置掺杂曲线文件
    a.tcad.cdpath = txt_path
    # 注入随工艺变化的缺陷参数（若提供）
    apply_defect_params(a.tcad, defect_values)

    # 输出文件
    a.tcad.outlog = outlog
    a.tcad.outstr = outstr
    a.tcad.outcsv = outcsv

    sim_result = run_atlas_simulation(a, timeout_seconds=timeout_seconds)
    if sim_result["timed_out"]:
        status = f"timeout: {sim_result['error']}"
    elif not sim_result["success"]:
        status = f"error: {sim_result['error']}"
    elif not os.path.isfile(outcsv):
        status = "error: csv not found"
    else:
        status = "ok"

    rec = {
        "cd_txt": txt_path,
        "outlog": outlog,
        "outstr": outstr,
        "outcsv": outcsv,
        "model": "topcon_n_cd",
        "ts": int(time.time()),
        "status": status,
        "timeout_seconds": timeout_seconds,
    }
    if defect_values:
        rec["defect_values"] = defect_values
    return rec


def resolve_input_dir(project_root: str) -> str:
    """按优先级返回存在的输入目录。"""
    candidates = [
        os.path.join(project_root, "saomiao3csv"),
        os.path.join(project_root, "saomiao3.csv"),
        os.path.join(project_root, "saomiao3"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        f"未找到输入目录: {candidates} 中的任意一个。请确认 saomiao3csv/saomiao3.csv/saomiao3 是否存在"
    )


def main():
    # 项目根 = se-TOPCon 文件夹
    project_root = os.path.abspath(os.path.dirname(__file__))

    # 输入/输出目录（默认指向项目主数据）
    in_dir = resolve_input_dir(project_root)
    out_dir = os.path.join(project_root, "saomiao3jv")
    ensure_dir(out_dir)

    # ATLAS 可执行目录(如需修改, 可直接改此变量)
    exe_dir = os.environ.get("TCAD_EXE_DIR", r"C:\sedatools\exe")

    txt_files = list_txt_files(in_dir)
    if not txt_files:
        print(f"在目录中未找到 .txt 掺杂文件: {in_dir}")
        return

    print(f"发现 {len(txt_files)} 个掺杂曲线文件, 开始仿真……")

    defect_map = {}
    if DEFECT_MAP_CSV:
        defect_map = load_defect_map(DEFECT_MAP_CSV)
        print(f"已加载缺陷映射: {DEFECT_MAP_CSV} (样本数={len(defect_map)})")
    else:
        print("未指定 DEFECT_MAP_CSV，使用固定默认缺陷参数进行 IV 计算。")

    ok_count = 0
    err_count = 0
    timeout_count = 0
    skip_count = 0

    # 预先收集已存在的输出 .csv 基名(小写)
    existing_csv_basenames = set()
    try:
        for f in os.listdir(out_dir):
            if f.lower().endswith('.csv'):
                existing_csv_basenames.add(os.path.splitext(f)[0].lower())
    except FileNotFoundError:
        pass

    for idx, txt_path in enumerate(txt_files, 1):
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        if already_done(txt_path, out_dir, existing_csv_basenames):
            skip_count += 1
            print(f"[{idx}/{len(txt_files)}] 跳过(已存在): {base_name}")
            continue

        print(f"[{idx}/{len(txt_files)}] 运行: {txt_path} (timeout={IV_TIMEOUT_SECONDS}s)")
        try:
            defects = defect_map.get(base_name.lower())
            rec = run_one_with_cd(txt_path, exe_dir, out_dir, defect_values=defects)
            status = str(rec.get("status", "error"))
            if status == "ok":
                ok_count += 1
                existing_csv_basenames.add(base_name.lower())
            elif status.startswith("timeout"):
                timeout_count += 1
                print(f"  超时跳过: {status}")
            else:
                err_count += 1
                print(f"  失败: {status}")
        except Exception as e:
            err_count += 1
            print(f"  失败: {e}")

    print(
        f"完成。成功 {ok_count} 个, 跳过 {skip_count} 个, "
        f"超时 {timeout_count} 个, 失败 {err_count} 个。结果输出目录: {out_dir}"
    )


if __name__ == "__main__":
    main()


