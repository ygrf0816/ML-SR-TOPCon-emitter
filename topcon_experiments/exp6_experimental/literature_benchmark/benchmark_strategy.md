# 文献实验对标策略（参数不全时的处理方案）

> 对应目录：`exp6_experimental/literature_benchmark/`  
> 文献提取结果：`extract_literature_info/full_table.csv`、`summary.md`  
> 文献参考曲线：`outputs/literature_double_gaussian/data/curves/D*_doping_curve.csv`

## 问题诊断

10 篇文献中 **0 篇** 能直接提供 SR 曲线公式所需的全部 8 个工艺参数；**仅 D8** 可较完整映射方阻公式。  
Table 1 的 `R_sh / N_p / z_p / z_f1 / z_f2` 是**双高斯拟合输出**，应作为对标目标，而非公式输入。

## 三层对标策略

### 层 1：曲线描述符从曲线本身提取（优先，已实现）

**思路**：文献已给出双高斯系数 → 我们已生成 0~2 μm、截断至 4×10¹⁸ cm⁻³ 的参考曲线。  
从该曲线数值积分/微分，提取与仿真数据集一致的 7 个掺杂描述符：

| 描述符 | 含义 |
|--------|------|
| `doping_N_peak` | 峰值浓度 |
| `doping_x_peak` | 峰值深度 |
| `doping_junction_depth` | N ≥ 4×10¹⁸ 的最深点 |
| `doping_FWHM` | 半高全宽 |
| `doping_gradient_max` | max\|dN/dz\| |
| `doping_dose` | ∫N dz（至结深） |
| `doping_R_sheet` | 直接取 Table 1 的 R_sh |

**脚本**：`literature_benchmark/curve_descriptors.py` → 输出 `outputs/literature_benchmark/data/literature_curve_descriptors.csv`

这样 **不再依赖文献正文补全描述符**。

### 层 2：方阻公式 — 固定已知工艺、优化未知参数

**公式**（`athena_to_doping_R_sheet`，实际使用 4 个变量）：

- 输入：`athena_thick`, ln(`athena_c_boron`), `athena_temp2`, `athena_time2`
- 输出：`doping_R_sheet` (Ω/□)

**做法**：

1. 从 `full_table.csv` 解析已知工艺（范围取中值，如 860–900℃ → 880℃）
2. 固定已知参数
3. 对缺失参数在仿真数据集 5%~95% 分位范围内做差分进化优化
4. 目标：最小化 `(R_sheet_pred - R_sheet_table)²`

**注意**：这是**反演/校准**，得到的是“与 SR 公式自洽的一组虚拟工艺”，不是文献真实配方。  
论文中应表述为 *inverse calibration under partial constraints*。

### 层 3：掺杂曲线公式 — 形状拟合（截断区内）

**公式**（`doping_curve_tail_full`，depth ∈ [0.25, 2] μm）：

- 训练时输入：**18 维**（8 工艺 + 6 掺杂描述符 + 3 缺陷 + depth；**不含** `doping_R_sheet`）
- 输出：ln(N)
- 关键：`x17` = `depth_um`，`x10` = `doping_junction_depth`（勿与当前 19 维 `MODEL2_FEATURES` 混用）

**做法**（`fit_sr_to_literature.py`，默认 `descriptor_mode=anchors`）：

1. **固定** Table 1 的 `N_p`/`z_p` + 文献已知工艺
2. **优化**缺失工艺 + 其余掺杂/缺陷描述符（以曲线提取值为初值，0.25×~4× 边界）
3. 目标：文献深度 0~1.75 μm（≡ 物理深度 0.25~2 μm）且 `N ≥ 4×10¹⁸` 区间最小化 MSE(ln N)
4. 绘图/拟合均用文献坐标（0 起点）；喂入 SR 公式时深度 **+0.25 μm**
5. 截断区外不参与拟合

**解读**：缺陷描述符与缺失工艺在这里充当**形状调节自由度**，仅保证尾部窗口内曲线形态一致，不声称物理真实。

## 对标类型与推荐动作

| 类型 | 条件 | 方阻对标 | 曲线对标 |
|------|------|----------|----------|
| A 直接 | ≥4 个核心工艺已知 | 固定已知 + 优化其余 | 层1描述符 + 优化缺陷/缺失工艺 |
| B 部分 | 2–3 个工艺已知 | 反演校准 R_sh | 形状拟合（主要自由度在缺陷描述符） |
| C 仅曲线 | 工艺几乎全无 | 不做（或仅报告 Table R_sh） | **仅比较双高斯曲线 vs 仿真曲线形状**（不经过 SR） |

当前 10 条文献大致归属：A≈1(D8)，B≈5，C≈4(D1,D3,D4,D9)。

## 当前对标结果（anchors 模式，2025-06）

| ID | 曲线 R²_log | 方阻反演 | 已知工艺数 | 备注 |
|----|------------|---------|-----------|------|
| D1 | 0.986 | ✓ | 2 | 缺 BSG/硼源等，形状拟合良好 |
| D2 | 0.239 | ✓ | 4 | 尾部形状偏差较大 |
| D3 | 0.851 | ✓ | 1 | 拟合窗口仅 3 点（浅峰） |
| D4 | -4.1 | ✓ | 1 | 浅结，有效深度 <0.25 μm |
| D5 | -66 | ✓ | 7 | 工艺几乎全已知，自由度不足 |
| D6 | 0.925 | ✓ | 2 | |
| D7 | 差 | ✓ | 7 | 同 D5，仅缺 F_N2 |
| D8 | **0.982** | ✓ | 6 | 最佳文献配方，曲线/方阻均可对标 |
| D9 | **0.999** | ✓ | 0 | 全靠优化，形状极好 |
| D10 | **0.999** | ✓ | 4 | |

方阻层（层 2）10/10 条均可将 Table `R_sh` 反演至相对误差 <10⁻⁶%。

输出目录：`outputs/literature_benchmark/data/` + `plots/curve_fit_D*.png`

```bash
# 1) 生成文献双高斯曲线（若尚未运行）
python -m topcon_experiments.exp6_experimental.literature_double_gaussian.double_gaussian

# 2) 提取描述符 + 形状/方阻拟合 + 出图
python -m topcon_experiments.exp6_experimental.literature_benchmark.fit_sr_to_literature
```

## 输出文件

| 文件 | 内容 |
|------|------|
| `literature_curve_descriptors.csv` | 从曲线提取的 7 描述符 + R_sh |
| `curve_shape_fit_summary.csv` | 每条文献曲线形状拟合 RMSE / R² |
| `r_sheet_fit_summary.csv` | 方阻反演结果 |
| `plots/curve_fit_D*.png` | 文献曲线 vs SR 拟合曲线叠加图 |
| `benchmark_feasibility_merged.csv` | 提取表 + 描述符合并 |

## 双高斯先验注入 SR（2025-06-27；2026-07-21 复核与修正）

> 背景：上表 anchors 模式及 5 种纯 SR 窗口方案（legacy_tail / adaptive / adaptive_floor1e18 /
> keep_bsg / keep_bsg_floor1e18）在仿真集与文献集上精度均差（中位 R²_log 多为负值），
> 即"SR 公式与实测双高斯曲线对不上"的根本症结。

**方案**（`exp4_symbolic/run_curve_prior_sr_experiment.py`）：先对每条曲线拟合 4 参数双高斯
(N_p, z_p, z_f1, z_f2) 作为先验，SR 只学残差：

- `residual`：目标 = ln N − ln N_DG，最终 N(z) = N_DG(z) · exp(SR(特征, z))，DG 4 参数作为附加特征
- `feature_aug`：目标仍为 ln N，DG 参数 + ln N_DG 作为附加特征（失败，见下）
- `struct_bias`：自定义 gauss_kernel 算子（**未训练完成**，无公式产物）

**训练窗口（以代码为准，不是过时注释）**：`adaptive` BSG 裁剪（绝对物理深度，不归零）→
DG 在 `N≥1e18` 上拟合；历史 SR 训练仍用窗内全部 `N>0` 点。

**结果**（复核于 2026-07-21）：

| 模型 | 仿真集中位 R²_log | 文献集中位 R²_log (D1–D10) | 文献 n(R²>0.95) |
|------|------------------|---------------------------|-----------------|
| double_gaussian（上限参照） | ~0.99 | 1.000 | 10/10 |
| **prior_residual_full** | ~0.98–0.99 | **0.998** | **10/10** |
| prior_residual_athena | ~0.79–0.83 | 0.875 | 2/10 |
| prior_feature_aug_* | 负值 | 大负值 | 0/10 |

**诚实结论（重要）**：

1. **与文献双高斯曲线“对得上”主要靠双高斯骨架本身**；`prior_residual_full` 的 R² 与纯 DG 几乎同级，
   相对 DG 的 ΔR² 中位接近 0 或略负——**不能写成“SR 学到了可泛化的强修正”**。
2. 文献 D1–D10 参考曲线本身就是 Table-1 双高斯合成剖面，再 fit DG → R²=1 再乘残差，
   **存在循环验证成分**；外推金标准应换成数字化 ECV/SIMS。
3. 形态是 "DG 骨架 × SR 修正"：需要对曲线先做 DG 拟合；**不是**纯“工艺参数 → 曲线”。
   纯工艺版 `prior_residual_athena` 明显更弱。
4. `feature_aug` 路线彻底失败，说明必须以乘性残差方式强制利用先验。

评估脚本：`eval_prior_sr_on_sim.py`（仿真集）、`fit_prior_sr_to_literature.py`（文献集）
输出：`outputs/prior_sr_eval/`、`outputs/literature_benchmark/prior_/`

## 模型文件位置（不是单独的 .pth/.pkl）

| 内容 | 路径 |
|------|------|
| 残差 SR 公式库（含最优式） | `outputs/exp4_symbolic/curve_prior_experiment/sr_formulas_doping_residual_full.csv` |
| 每样本双高斯 4 参数 | `outputs/exp4_symbolic/curve_prior_experiment/data/dg_params.csv` |
| 自适应去 BSG 后的仿真曲线 | `outputs/exp4_symbolic/curve_processed_doping_adaptive.csv` |
| 仿真二次检验（按 train/test） | `exp6_experimental/outputs/prior_sr_eval/data/prior_sr_summary_by_split.csv` |
| **相对纯 DG 的 ΔR²** | `exp6_experimental/outputs/prior_sr_eval/data/prior_sr_delta_vs_dg_by_split.csv` |
| 带 BSG 截断可视化的逐样本图 | `exp6_experimental/outputs/prior_sr_eval/plots/sim_curve_checks/` |

最优 residual 公式实际主要只用 `x18=depth, x20=dg_z_p, x22=dg_z_f2`。
CSV 最优行即 SR 部分；完整模型 = 双高斯函数 + `dg_params` + CSV 公式 + \(N=N_{DG}e^{SR}\)。

## 仿真二次检验（修正 BSG 截断后，2026-07-21）

- 已修：`detect_bsg_cliff_end` **强制 cut ≥ athena_thick + 0.03 μm**，重建 adaptive CSV + `dg_params`。
- 评估：`N≥1e18` 段；按训练同一 group split 分开 train/test；逐样本图标出排除的 BSG 段。
- `prior_residual_full`：test 中位 R²≈**0.989**（9/9 >0.9）；但请同时看 **ΔR² vs DG**。
- 注意：磁盘上 SR 公式仍可能是旧窗口训练产物；完全自洽需 `--force` 重训 residual。

## 后续可增强

1. **ECV 数字化**：对 D5/D6/D8 论文图逐条数字化，替换双高斯参考曲线
2. **Benick 博士论文**：补 D3 扩散配方，可能提升 D1/D3 至 B 类
3. **多目标**：同时约束 R_sh + 曲线形状（权重 λ₁·MSE_R + λ₂·MSE_curve）
4. **不确定性**：对工艺范围（非单值）做蒙特卡洛，给出对标置信带

## 论文表述建议

- 工艺参数不全时，**不强行声称“用文献工艺代入 SR 公式”**
- 应写：**“在部分工艺约束下，通过反演缺失参数使 SR 预测与文献发表的方阻/掺杂剖面在有效深度窗口内一致”**
- Table 1 双高斯曲线作为 **external benchmark profile**，与 TCAD+SR 链条解耦
- **不要**把 prior_residual 的高 R² 写成“SR 单独解决了对标”；应同时报告纯双高斯基线与 ΔR²
- “工艺 → 曲线”主张只留给 athena-only 路径；full/描述符版标为 DG 精修（需测线或仿真曲线先验）
