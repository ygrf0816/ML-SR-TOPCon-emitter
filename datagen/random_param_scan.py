import argparse
import json
import os
import random
import string
import sys
import time
from typing import Dict

from python_tcad import pythontcad
import tcadmodel


def loguniform(low: float, high: float) -> float:
    import math

    log_low = math.log(low)
    log_high = math.log(high)
    return math.exp(random.uniform(log_low, log_high))


def rand_name(prefix: str = "Topcon_n_rand_") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    suf = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}{ts}_{suf}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def one_run(exe_dir: str, project_root: str) -> Dict:
    """执行一次随机参数仿真，返回记录字典。"""
    # 路径
    saomiao_dir = os.path.join(project_root, "saomiao")
    ensure_dir(saomiao_dir)

    # 随机文件名
    base = rand_name()
    outlog = os.path.join(saomiao_dir, f"{base}.log")
    outstr = os.path.join(saomiao_dir, f"{base}.str")
    outcsv = os.path.join(saomiao_dir, f"{base}.csv")

    # 采样参数（与示例扫描范围一致，使用对数均匀）
    params = {
        "resist_rear": loguniform(5e-4, 5e-2),     # Ω·cm^2
        "taun_Si": loguniform(1e-3, 1e-1),        # s
        "Nt_top_N": loguniform(1e10, 5e11),       # cm^-2 或 cm^-3 视定义
        "Nt_polySi_top": loguniform(1e13, 5e15),  # cm^-3
    }

    # 初始化仿真器
    a = pythontcad()
    if exe_dir:
        a.set_tcad(exe_dir)

    # 选用 topcon_n 模型
    a.tcad = tcadmodel.topcon_n()

    # nk 与模板指向当前脚本所在目录，避免绝对路径依赖
    nk_tpl_dir = os.path.dirname(__file__)
    a.tcad.nkpath = nk_tpl_dir + os.sep
    a.tcad.template_lib = os.path.join(nk_tpl_dir, "template.lib")

    # 应用参数
    a.tcad.resist_rear = params["resist_rear"]
    a.tcad.taun_Si = params["taun_Si"]
    a.tcad.Nt_top_N = params["Nt_top_N"]
    a.tcad.Nt_polySi_top = params["Nt_polySi_top"]

    # 输出文件
    a.tcad.outlog = outlog
    a.tcad.outstr = outstr
    a.tcad.outcsv = outcsv

    # 生成命令并执行
    a.tcad.updatecommand()
    a.updateprocess()
    for cmd in a.tcad.get_allcommandlist():
        a.run_atlas_command(cmd)

    rec = {
        "file_base": base,
        "outlog": outlog,
        "outstr": outstr,
        "outcsv": outcsv,
        "params": params,
        "ts": int(time.time()),
        "status": "ok",
    }
    return rec


def main():
    # parser = argparse.ArgumentParser(description="随机生成参数并运行一次/多次 TOPCon_n 仿真，记录到 saomiao.jsonl")
    # parser.add_argument("--exe", default=os.environ.get("TCAD_EXE_DIR", "D:/silvaco/exe"), help="ATLAS 可执行目录（默认环境变量 TCAD_EXE_DIR 或 D:/silvaco/exe)")
    # parser.add_argument("--runs", type=int, default=1, help="运行次数，默认 1")
    # parser.add_argument("--seed", type=int, default=None, help="随机种子")
    # args = parser.parse_args()


    # 项目根目录：TOPCon/pycode/../../
    project_root = "D:\\code\\silvaco\\tcadproject\\saomiao2"
    jsonl_path = os.path.join(project_root, "saomiao.jsonl")

    for _ in range(max(1, int(10000))):
        try:
            rec = one_run("D:/silvaco/exe", project_root)
        except Exception as e:  # 失败也写入一条，便于追踪
            base = rand_name()
            rec = {
                "file_base": base,
                "params": {},
                "ts": int(time.time()),
                "status": f"error: {e}",
            }
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"完成，记录已追加到: {jsonl_path}")


if __name__ == "__main__":
    main()


