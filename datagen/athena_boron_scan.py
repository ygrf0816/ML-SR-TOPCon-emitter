import json
import math
import os
import random
import string
import time
from typing import Dict, List
import concurrent.futures as futures
from itertools import repeat

from python_tcad import pythontcad

# 采样模式:
# - "discrete": 从离散网格中抽样（旧模式）
# - "continuous": 在边界内连续随机生成（推荐）
SAMPLING_MODE = "continuous"

# 可选：固定随机种子以复现数据集；None 表示每次不同
RANDOM_SEED = None
if RANDOM_SEED is not None:
	random.seed(RANDOM_SEED)


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def rand_name(prefix: str = "Athena_BoronScan_") -> str:
	ts = time.strftime("%Y%m%d_%H%M%S")
	suf = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
	return f"{prefix}{ts}_{suf}"


def frange_inclusive(start: float, stop: float, step: float) -> List[float]:
	# 生成包含端点的浮点序列，避免累计误差
	n = int(round((stop - start) / step))
	vals = [start + i * step for i in range(n + 1)]
	# 统一四舍五入到小数点后三位，避免0.30000000004
	return [round(v, 6) for v in vals]


def logspace_inclusive(start: float, stop: float, num: int) -> List[float]:
	log_low = math.log10(start)
	log_high = math.log10(stop)
	if num == 1:
		return [start]
	vals = [10 ** (log_low + i * (log_high - log_low) / (num - 1)) for i in range(num)]
	return vals


def loguniform(start: float, stop: float) -> float:
	"""连续对数均匀采样。"""
	log_low = math.log10(start)
	log_high = math.log10(stop)
	return 10 ** random.uniform(log_low, log_high)


def sample_params() -> Dict:
	if SAMPLING_MODE == "continuous":
		# 连续随机组合：避免离散网格导致参数取值分布不均匀
		return {
			"thick": round(random.uniform(0.01, 0.2), 5),
			"c_boron": loguniform(1e19, 5e21),
			# "c_boron": loguniform(9e21, 9e21),
			"temp1": round(random.uniform(900, 1100), 4),
			"time1": round(random.uniform(30, 300), 4),
			"temp2": round(random.uniform(900, 1100), 4),
			"time2": round(random.uniform(15, 90), 4),
			"F_N2": round(random.uniform(1, 10), 4),
			"F_O2": round(random.uniform(1, 10), 4),
		}

	# 1) deposit: thick 0.01-0.2 step 0.01
	thick_vals = frange_inclusive(0.01, 0.2, 0.01)

	# 2) deposit: c.boron 5e20-5e21 对数间隔 10点
	cb_vals = logspace_inclusive(1e19, 5e21, 10)

	# 3) diffuse1: temp 900-1100 step 10
	temp1_vals = list(range(900, 1100 + 1, 10))

	# 4) diffuse1: time 30-300 step 15
	time1_vals = list(range(30, 300 + 1, 15))

	# 5) diffuse2: temp 900-1100 step 10
	temp2_vals = list(range(900, 1100 + 1, 10))

	# 6) diffuse2: time 15-90 step 15
	time2_vals = list(range(15, 90 + 1, 15))

	# 7) F.N2: 1-10 step 1
	fn2_vals = list(range(1, 10 + 1))

	# 8) F.O2: 1-10 step 1
	fo2_vals = list(range(1, 10 + 1))

	return {
		"thick": random.choice(thick_vals),
		"c_boron": random.choice(cb_vals),
		"temp1": random.choice(temp1_vals),
		"time1": random.choice(time1_vals),
		"temp2": random.choice(temp2_vals),
		"time2": random.choice(time2_vals),
		"F_N2": random.choice(fn2_vals),
		"F_O2": random.choice(fo2_vals),
	}


def build_athena_deck(params: Dict, outstr: str) -> List[str]:
	lines = [
		"line x loc=0.0 spacing=0.1",
		"line x loc=0.5 spacing=0.1",
		"line y loc=0.0   spacing=0.001",
		"line y loc=0.2   spacing=0.001",
		"line y loc=0.8   spacing=0.005",
		"line y loc=2.0   spacing=0.01",
		"init silicon orientation=100 phosphor resistivity=3",
		"method full.cpl",
		"models fermi pair.diff",
		f"deposit oxide thick={params['thick']:.2f} c.boron={params['c_boron']:.6e}",
		f"diffuse time={params['time1']} temp={params['temp1']} press=1 nitrogen",
		f"diffuse time={params['time2']} temp={params['temp2']} press=1 F.N2={params['F_N2']} F.O2={params['F_O2']}",
		f"structure outfile={outstr}",
		"quit",
	]
	return lines


def run_one(exe_dir: str, out_dir: str) -> Dict:
	ensure_dir(out_dir)
	# 随机文件名（不含扩展名）
	base = rand_name()
	outstr = os.path.join(out_dir, f"{base}.str")

	# 采样参数
	params = sample_params()

	# 初始化并启动 Athena 进程
	a = pythontcad()
	if exe_dir:
		a.set_tcad(exe_dir)
	a.updateprocess()

	# 生成并执行 deck
	deck = build_athena_deck(params, outstr)
	for cmd in deck:
		# 逐条发送到 athena
		a.run_athena_command(cmd, output_state=False)

	rec = {
		"file_base": base,
		"outstr": outstr,
		"params": params,
		"ts": int(time.time()),
		"status": "ok",
	}
	return rec


def _worker_entry(exe_dir: str, out_dir: str) -> Dict:
	# 子进程入口：执行一次并捕获异常，返回记录给主进程统一写入
	try:
		return run_one(exe_dir, out_dir)
	except Exception as e:
		base = rand_name()
		return {
			"file_base": base,
			"outstr": os.path.join(out_dir, f"{base}.str"),
			"params": {},
			"ts": int(time.time()),
			"status": f"error: {e}",
		}


def run_scan(
	runs: int = 10000,
	out_dir: str = "saomiao3",
	exe_dir: str = "D:/silvaco/exe",
	jsonl: str = None,
	parallel: bool = False,
	workers: int = max(1, (os.cpu_count() or 1)),
) -> None:
	# 为 IDE 场景准备：直接调用该函数即可运行，无需命令行参数
	jsonl_path = jsonl or os.path.join(out_dir, "athena_boron_scan.jsonl")
	ensure_dir(out_dir)
	dirname = os.path.dirname(jsonl_path)
	if dirname:
		ensure_dir(dirname)

	if parallel and workers > 1:
		with futures.ProcessPoolExecutor(max_workers=workers) as ex, open(jsonl_path, "a", encoding="utf-8") as f:
			for rec in ex.map(
				_worker_entry,
				repeat(exe_dir, runs),
				repeat(out_dir, runs),
				chunksize=1,
			):
				f.write(json.dumps(rec, ensure_ascii=False) + "\n")
	else:
		for _ in range(max(1, int(runs))):
			try:
				rec = run_one(exe_dir, out_dir)
			except Exception as e:
				base = rand_name()
				rec = {
					"file_base": base,
					"outstr": os.path.join(out_dir, f"{base}.str"),
					"params": {},
					"ts": int(time.time()),
					"status": f"error: {e}",
				}
			with open(jsonl_path, "a", encoding="utf-8") as f:
				f.write(json.dumps(rec, ensure_ascii=False) + "\n")

	print(f"完成，记录已追加到: {jsonl_path}")


def main():
	# IDE 直接运行本文件时，使用默认配置。
	run_scan(parallel=True,workers=15)

if __name__ == "__main__":
	run_scan()


