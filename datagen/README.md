# se-TOPCon 文件说明

这个文件夹用于整理“SE 工艺仿真结果导入 TOPCon 器件仿真”的相关脚本和参考文件。当前只是从现有工程中复制出相关文件，方便后续在这个目录里继续改造成完整自动化流程。

## 推荐理解顺序

1. `se.in`：SE 工艺 Athena 脚本，是前端工艺仿真的输入来源。
2. `topcon_datagen_main.py`：TOPCon 数据生成主入口，负责把 Athena 结构结果提取成掺杂/缺陷输入，再批量运行 TOPCon IV，并汇总数据集。
3. `cd_txt_scan.py`：读取掺杂曲线 `.txt`，调用 TOPCon 自定义掺杂模型运行 ATLAS 器件仿真。
4. `analyze_scans.py`：从 TOPCon 输出的 IV CSV 中计算 `Voc`、`Jsc`、`FF`、`Eff` 等电池指标。
5. `tcadmodel.py`、`python_tcad.py`、`tcaddata.py`、`tcadutils.py`、`template.lib` 和 `.nk` 文件：TOPCon 器件模型运行依赖。

## 自动化主流程相关文件

### `se.in`

Silvaco Athena 的选择性发射极工艺仿真脚本。脚本前部集中定义了 SE 工艺参数，例如轻掺/重掺注入剂量、注入能量、驱入时间、退火温度和退火时间；后部负责生成结构、提取方阻、结深和表面浓度等工艺指标。后续自动化时，它可以作为 SE 工艺 deck 模板或参数扫描基础。

### `topcon_datagen_main.py`

TOPCon 自动化数据生成主入口。它已有完整的后处理链路：从 Athena `.str` 文件解析掺杂和缺陷信息，导出 TOPCon 可读取的掺杂曲线 `.txt`，生成缺陷映射，再调用 TOPCon IV 仿真，最后汇总成数据集。这个文件最适合作为“SE 结果接入 TOPCon”的主控脚本参考。

### `cd_txt_scan.py`

基于外部掺杂曲线 `.txt` 运行 TOPCon 器件仿真的脚本。它使用 `tcadmodel.topcon_n_cd()`，通过 `cdpath` 把工艺提取出的掺杂曲线导入 TOPCon 模型，并支持通过缺陷映射表注入缺陷参数。

### `analyze_scans.py`

负责读取 ATLAS 输出的 IV 曲线 CSV，并计算电池性能指标，包括 `Voc`、`Jsc`、`FF`、`Eff`、`Pm` 等。自动化流程最终得到“电池结果”时，需要用它把仿真 CSV 转成可分析的指标表。

## TOPCon 模型与运行依赖

### `python_tcad.py`

TCAD 调用封装脚本，负责启动 Athena/ATLAS 进程、读取 `.in` deck、替换变量，并逐条发送命令执行。`topcon_datagen_main.py`、`cd_txt_scan.py`、`athena_boron_scan.py` 等脚本都会依赖它来实际调用 Silvaco。

### `tcadmodel.py`

TOPCon 器件模型定义文件。里面包含 `topcon_n()`、`topcon_n_cd()` 等模型类，定义器件结构、材料、接触、电学模型、缺陷参数、掺杂曲线导入方式和输出文件设置。SE 结果导入 TOPCon 时，核心就是让工艺提取出的曲线进入这里的 `topcon_n_cd` 模型。

### `tcaddata.py`

Silvaco 命令对象和参数定义库。`tcadmodel.py` 通过它生成 ATLAS/Athena 命令。文件较大，主要是底层命令拼装依赖，一般不直接修改。

### `tcadutils.py`

TCAD 命令生成的辅助函数文件。它为 `tcaddata.py`、`tcadmodel.py` 提供较底层的工具函数。

### `template.lib`

ATLAS 用户函数模板库。TOPCon 模型中会设置 `template_lib` 指向这个文件，用于支持模型里的自定义材料/迁移率/复合等函数调用。

### `.nk` 文件

这些文件是光学常数数据，供 TOPCon 模型中的材料光学参数使用：

- `ag_nk.nk`：银材料光学常数。
- `al2o3_nk.nk`：氧化铝材料光学常数。
- `ito_nk.nk`：ITO 材料光学常数。
- `mgf2_nk.nk`：氟化镁材料光学常数。
- `polySi_nk.nk`：多晶硅材料光学常数。
- `si_nk.nk`：硅材料光学常数。
- `sinx_nk.nk`：氮化硅材料光学常数。

## Athena/SE 相关参考脚本

### `athena_boron_scan.py`

原来用于 Athena 硼扩散工艺随机扫描的脚本。它展示了如何用 Python 生成 Athena deck、随机采样工艺参数、批量运行 Athena 并把参数记录到 JSONL。虽然它不是当前 `se.in` 的直接版本，但对把 `se.in` 改造成可批量参数扫描的 Python 控制脚本很有参考价值。

### `extract_athena_defect_features.py`

从 Athena 导出的结构 CSV 中提取缺陷曲线和统计特征，并生成用于 TOPCon IV 的缺陷参数映射表。当前 `topcon_datagen_main.py` 已经包含了更完整的 `.str` 提取逻辑，但这个文件对理解“工艺缺陷如何映射到器件缺陷参数”有参考价值。

### `temp_trap_effect_check_v2.py`

用于快速检查从 Athena 结构中提取的陷阱/缺陷曲线对 TOPCon IV 结果的影响。它会从一个 `.str` 文件构造掺杂曲线和 trap 曲线，然后分别运行基准模型和带 trap 曲线的模型，比较电池指标变化。

### `test_athena_io.py`

Athena 进程交互测试脚本。用于验证 Python 是否能正确启动 `athena.exe` 并向其发送命令。后续如果自动化运行 SE deck 时遇到进程通信问题，可以用它排查。

## 扫描、验证和分析参考脚本

### `random_param_scan.py`

早期 TOPCon 随机参数扫描脚本。它直接随机采样 TOPCon 器件参数并运行 `topcon_n()`，主要用于参考“如何批量生成参数、运行仿真并记录 JSONL”。

### `run_eff2030_quickcheck.py`

读取已有掺杂曲线数据，导出 `.txt` 曲线并用 `topcon_n_cd()` 快速跑 TOPCon IV 的检查脚本。它对“外部曲线导入 TOPCon 模型”很有参考价值，也包含超时和跳过已完成样本的处理逻辑。

### `summarize_boron_scan.py`

用于汇总 Athena 硼扩散扫描结果的分析脚本。它可以作为后续汇总 SE 工艺扫描结果、提取统计指标和建立数据表的参考。

## 后续改造方向

当前文件已经覆盖自动化链路所需的主要部件，但还没有专门为 `se.in` 写新的控制脚本。后续如果继续开发，可以按下面方向改造：

1. 以 `athena_boron_scan.py` 的方式，把 `se.in` 参数化并批量运行 SE Athena。
2. 复用 `topcon_datagen_main.py` 的 `.str` 解析逻辑，从 SE 结构结果中提取 TOPCon 所需的掺杂曲线和缺陷曲线。
3. 通过 `cd_txt_scan.py` 调用 `topcon_n_cd()`，把 SE 掺杂曲线导入 TOPCon 器件模型。
4. 通过 `analyze_scans.py` 从 TOPCon IV CSV 中计算最终电池指标。
