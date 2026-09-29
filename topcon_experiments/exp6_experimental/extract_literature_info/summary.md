# 工艺参数提取总表与对标可行性评估

> 生成时间：2026-06-24  
> 主论文：2026Xu-LECO (Progress in Photovoltaics, 2026)  
> 任务：从 Table 1 引用文献 [37]–[46] 中提取工艺参数，用于符号回归（SR）公式对标

---

## 文献与PDF文件映射

| ID  | 文献编号 | 论文标题 | 期刊/年份 | PDF文件名 |
|-----|----------|----------|-----------|-----------|
| D1  | [37] | Impact of boron doping on electrical performance and efficiency of n-TOPCon solar cell | Solar Energy 227 (2021) | 1-s2.0-S0038092X21007398-main.pdf |
| D2  | [38] | Superb improvement of boron doping in selective emitter for TOPCon solar cells via boron-doped silicon paste | Solar Energy 247 (2022) | 1-s2.0-S0038092X22007642-main.pdf |
| D3  | [39] | Boron Emitter Passivation With Al₂O₃ and Al₂O₃/SiNx Stacks Using ALD Al₂O₃ | IEEE JPV 3(1) (2013) | Boron_Emitter_Passivation_With...pdf |
| D4  | [40] | Optimization of efficiency enhancement of TOPCon cells with boron selective emitter | Sol. En. Mat. Sol. Cells 263 (2023) | 1-s2.0-S0927024823004063-main.pdf |
| D5  | [41] | Study of boron diffusion for p⁺ emitter of large area n-type TOPCon silicon solar cells | Applied Physics A 126 (2020) | s00339-020-03851-5.pdf |
| D6  | [42] | Boron tube diffusion process parameters for high-efficiency n-TOPCon solar cells with selective boron emitters | Sol. En. Mat. Sol. Cells 253 (2023) | 1-s2.0-S0927024823000521-main.pdf |
| D7  | [43] | Study on selective emitter fabrication through an innovative pre-diffusion process for enhanced efficiency in TOPCon solar cells | Prog. PV 32(3) (2024) | Progress in Photovoltaics - 2023 - Chen - ...pdf |
| D8  | [44] | Optimization of boron depletion for boron-doped emitter of n-type TOPCon solar cells | Mat. Sci. Semicon. Proc. 178 (2024) | 1-s2.0-S1369800124003202-main.pdf |
| D9  | [45] | High-Efficiency TOPCon Solar Cells With Laser-Assisted Localized Boron Doping via Silicon Paste | IEEE JPV 14(4) (2024) | High-Efficiency_TOPCon_Solar_Cells_With...pdf |
| D10 | [46] | High-efficiency TOPCon solar cell with superior p⁺ and p⁺⁺ layer via one-step processing | Solar Energy 271 (2024) | 1-s2.0-S0038092X24001427-main.pdf |

---

## 表 1：工艺参数提取总表

> 说明：只记录文献中明确给出的数值；需推算的标注 `inferred`；未找到的填 `NA`。单位已统一换算。R_sh、N_p、z_p 等为测量输出（OUTPUT），非公式输入（INPUT）。

### CSV格式（可粘贴）

```
profile_id,ref,paper_title,sample_label,athena_thick_um,athena_c_boron_cm3,athena_temp1_C,athena_time1_min,athena_temp2_C,athena_time2_min,athena_F_N2,athena_F_O2,R_sheet_measured_ohm_sq,N_p_cm3,z_p_um,z_f1,z_f2,source_page_table,process_mapping_notes,missing_params,confidence
D1,37,Wang2021_SolEnergy227,BCl3_tube_D1-D3,NA,NA,NA,NA,950,NA_range(0.5-20),NA,NA,117,1.94e19,0.17,0.1959,0.3148,Table2_P5,BCl3 tube diffusion;4-step process(deposition+oxidation+drive-in+cooling);BCl3=10-20 ml/min;氧化温度950C;R_sh=117 matches D3 emitter in paper not D1;扩散时间范围给出,athena_thick|athena_c_boron|athena_temp1|athena_time1|athena_F_N2|athena_F_O2|athena_time2(exact),partial
D2,38,Hong2022_SolEnergy247,B_doped_Si_paste_SE,NA,NA,800_or_850,30,850,30,NA,NA,39,6.36e19,0.12,0.1732,0.4151,P2-P7,Two-step: Step1(B-doped Si paste screen printing+850C diffusion 30min N2); Step2(BBr3 tube diffusion); R_sh=39 matches BS20-60(850C 60min) variant not base condition; c_boron only via ICP wt% data not cm-3,athena_thick|athena_c_boron_cm3|athena_F_N2|athena_F_O2,partial
D3,39,Richter2013_IEEE_JPV,BBr3_Al2O3_passivation,NA,NA,890_or_940,NA,NA,NA,NA,NA,95,9.71e19,0.03,0.0320,0.1311,TableI_P2,Passivation paper NOT diffusion paper; BBr3 tube diffusion details from cited Benick PhD thesis[38]; Emitter E1:890C initial; E1*:940C; E2:890C+Ar drive-in; no diffusion time given; D3 R_sh=95 closest to E2(88 Ohm/sq),athena_thick|athena_c_boron|athena_time1|athena_temp2|athena_time2|athena_F_N2|athena_F_O2,no
D4,40,Li2023_SEMSC263,mask_etchback_SE,NA,NA,NA,NA,~1000,NA,NA,NA,324,4.12e18,0.10,0.1088,0.4274,P1-P4,Low-pressure BCl3 diffusion; 4-step(pre-oxidation+deposition+drive-in+post-oxidation); mask and etch-back for SE; only oxidation temp mentioned ~1000C; no detailed recipe given,athena_thick|athena_c_boron|athena_temp1|athena_time1|athena_time2|athena_F_N2|athena_F_O2,no
D5,41,Zhou2020_AppPhysA126,BBr3_tube_range,~0.10,NA_inferred(~1-5e19 from ECV),860-900,50,960-1000,60,NA,50-300_sccm(pre-dep)_0-3000_sccm(drive-in),112,2.12e19,0.16,0.2226,0.3131,P2_P8_Fig3-6,BBr3 tube diffusion; BSG thickness ~100nm from Page8; temps/times given as ranges not single values; O2 flow experimental matrix variable; need specific recipe from ECV figure digitization for D5 match,athena_c_boron_exact|athena_F_N2|exact_single_recipe,partial
D6,42,Wang2023_SEMSC253,BCl3_tube_SE,NA,NA,NA,NA,960,NA,NA,3000_sccm(oxidation),109,2.35e19,0.15,0.1556,0.3181,Table1_P2,BCl3 tube diffusion; detailed Table1 with 5 process parameters(GBCl3/Tdrive-in/Toxidation/GO2/toxidation); base: GBCl3=90sccm Tdrive-in=960C Toxidation=990C GO2=3000sccm t=60min; base R_sh=125 not 109; D6 likely from a specific variant condition,athena_thick|athena_c_boron|athena_temp1|athena_time1|athena_time2_exact|athena_F_N2,partial
D7,43,Chen2024_ProgPV,pre_diffusion_SE,~0.032,NA(~1e20 from ECV peak),810,9.67(580s),925,4.5(270s),NA,3.4:1_O2:BCl3_ratio(Step1)_3.1:1(Step2),269,4.70e18,0.17,0.2232,0.4040,Fig1_P3,BBr3 tube diffusion 2-step(from Figure 1 flow chart); Step1:810C deposition; Step2:925C oxidation; BSG~32nm; O2:BCl3 ratios given not absolute flows; R_sh=269 NOT found in paper measurements(74-203 Ohm/sq range); values likely from different batch or inferred from simulation,athena_c_boron_exact|athena_F_N2|athena_F_O2_absolute,partial
D8,44,Peng2024_MSSP178,BCl3_tube_boron_depletion,NA,NA,840_基线or600_优化,5(300s),1050_基线,75(4500s),3000_sccm(pre-ox)_0(oxidation),1000_sccm(pre-ox)_15000_sccm(oxidation),69,1.88e19,0.10,0.1752,0.7535,Table1_P3,BCl3 tube diffusion; MOST COMPLETE process recipe(Table1): pre-ox(840C 300s)->deposition(845C 540s BCl3=135sccm)->drive-in(945C 300s)->oxidation(1050C 4500s)->cooling; base R_sh=129-137 NOT 69; D8 N_peak=1.84e19 close to 1.88e19; z_p mismatch(0.10 vs 0.8-1.0 um),athena_thick|athena_c_boron_cm3,Best_recipe_available
D9,45,Hong2024_IEEE_JPV,laser_B_Si_paste,NA,NA,NA,NA,NA,NA,NA,NA,152,6.79e18,0.20,0.2248,0.5433,P4-P8,Laser-assisted B-doped Si paste doping NOT standard tube diffusion; NO standard tube parameters for p++; p+ baseline BCl3 tube R_sh~119 Ohm/sq; laser: 532nm 1.3J/cm2 20m/s; D9 values(R_sh=152) NOT found in this paper; may come from ref[19] Hong2022_SolEnergy,athena_thick|athena_c_boron|athena_temp1|athena_time1|athena_temp2|athena_time2|athena_F_N2|athena_F_O2,no
D10,46,Liu2024_SolEnergy271,one_step_BCl3_paste,NA,NA,950,20,950,20,NA,NA,322,7.91e18,0.048,0.0723,0.2966,P7,One-step BCl3+B-doped Si paste co-diffusion 950C 20min; NO pre-deposition+drive-in sub-structure; BSG thick NA; p+ R_sh=108.69 p++ R_sh=83.12; D10 R_sh=322 NOT in paper; N_peak=8.68e18 close to 7.91e18; mismatch on R_sh suggests D10 may use different variant,athena_thick|athena_c_boron|athena_F_N2|athena_F_O2,partial
```

### Markdown格式详细表

| profile_id | ref | athena_thick (μm) | athena_c_boron (cm⁻³) | athena_temp1 (℃) | athena_time1 (min) | athena_temp2 (℃) | athena_time2 (min) | athena_F_N2 | athena_F_O2 | R_sh (Ω/□) | N_p (cm⁻³) | z_p (μm) | z_f1 | z_f2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | [37] | NA | NA | NA | NA | 950 | NA(range 0.5-20) | NA | NA | 117 | 1.94×10¹⁹ | 0.17 | 0.1959 | 0.3148 |
| D2 | [38] | NA | NA | 800/850 | 30 | 850 | 30 | NA | NA | 39 | 6.36×10¹⁹ | 0.12 | 0.1732 | 0.4151 |
| D3 | [39] | NA | NA | 890/940 | NA | NA | NA | NA | NA | 95 | 9.71×10¹⁹ | 0.03 | 0.0320 | 0.1311 |
| D4 | [40] | NA | NA | NA | NA | ~1000 | NA | NA | NA | 324 | 4.12×10¹⁸ | 0.10 | 0.1088 | 0.4274 |
| D5 | [41] | ~0.10 | inferred(~1-5×10¹⁹) | 860-900 | 50 | 960-1000 | 60 | NA | 50-300 sccm / 0-3000 sccm | 112 | 2.12×10¹⁹ | 0.16 | 0.2226 | 0.3131 |
| D6 | [42] | NA | NA | NA | NA | 960 | NA | NA | 3000 sccm | 109 | 2.35×10¹⁹ | 0.15 | 0.1556 | 0.3181 |
| D7 | [43] | ~0.032 | inferred(~1×10²⁰) | 810 | 9.67 | 925 | 4.5 | NA | O₂:BCl₃=3.4:1 / 3.1:1 | 269 | 4.70×10¹⁸ | 0.17 | 0.2232 | 0.4040 |
| D8 | [44] | NA | NA | 840/600 | 5 | 1050 | 75 | 3000/0 sccm | 1000/15000 sccm | 69 | 1.88×10¹⁹ | 0.10 | 0.1752 | 0.7535 |
| D9 | [45] | NA | NA | NA | NA | NA | NA | NA | NA | 152 | 6.79×10¹⁸ | 0.20 | 0.2248 | 0.5433 |
| D10 | [46] | NA | NA | 950 | 20 | 950 | 20 | NA | NA | 322 | 7.91×10¹⁸ | 0.048 | 0.0723 | 0.2966 |

---

## 表 2：每篇文献摘要

### D1 — [37] Wang et al. 2021, Solar Energy 227

- **电池结构**：N型 TOPCon 太阳能电池
- **硼扩散路线**：BCl₃ 管式炉扩散（四步法：沉积→氧化→推进→冷却）
- **Table 1 关系**：本文 Table 2 给出 D1–D3 三组发射极工艺参数。D3 发射极（BCl₃=20 ml/min, 沉积5 min, 推进5 min, 氧化55 min at 990°C）的 N_max=1.94×10¹⁹ cm⁻³ 与主论文 Table 1 中 D1 的已知输出匹配，但本文 D1 发射极的结深为 0.5 μm（远大于 z_p=0.17 μm）。**R_sh=117 Ω/□ 未在本文中出现。**
- **无法映射的原因**：BSG厚度未给出；硼源浓度（cm⁻³）未给出，仅给出 BCl₃ 流量(ml/min)；一次扩散温度和时间缺失；气体流量缺失。
- **补充材料 (mmc1.docx) 发现**：
  - 包含 J₀ₑ,ₘₑₜₐₗ 计算方法说明（公式(1): J₀,tot = (J₀,cont - J₀,pass)f + 2J₀,pass）
  - **Fig. S1**: J₀,total vs. 金属化比例 f 的线性拟合图 → 可用于提取接触区 J₀ 值
  - **Fig. S2**: 发射极暗饱和电流密度 J₀ 对比图 (D1/D2/D3/N1/N2)
  - **Fig. S3** (image7-11): 固定结深(0.5/0.63/0.8μm)下，Eff/Voc/Jsc/FF/Rser vs. 表面浓度的散点图。**x轴表面浓度包含关键值 1.94×10¹⁹ atoms/cm³**（与D1的 N_p 完全匹配！），以及 2.88e19, 4e19, 1.49e19 等
  - **Fig. S4** (image12-16): 固定表面浓度(~3e19 和 ~2.6e19 atoms/cm³)下，Eff/Voc/Jsc/FF/Rser vs. 结深的散点图
  - ⚠️ **补充材料未提供任何新的工艺输入参数**（温度/时间/流量/厚度均未出现）。其价值在于验证了实验条件范围和 J₀ 数据，可用于后续曲线形状对比

### D2 — [38] Hong et al. 2022, Solar Energy 247

- **电池结构**：N型 TOPCon 太阳能电池（选择性发射极 p⁺/p⁺⁺）
- **硼扩散路线**：两步法 — Step1: B-doped Si浆料丝网印刷+850℃热扩散30min(N₂)；Step2: 传统BBr₃管式扩散(轻掺p⁺层)
- **Table 1 关系**：本文 BS20-60 条件（850℃, 60 min）的 R_sh=38.8 Ω/□ 与已知输出 R_sh=39 Ω/□ 匹配。但浆料中硼含量仅以 ICP wt% 给出（BS20: B=0.8994 mg/L），无法直接转换为 cm⁻³。
- **无法映射的原因**：BSG厚度未给出；硼源浓度（cm⁻³）无法从 ICP wt% 推算；N₂/O₂流量未量化。

### D3 — [39] Richter et al. 2013, IEEE JPV 3(1)

- **电池结构**：p⁺np⁺ 对称寿命测试结构（非完整电池）
- **硼扩散路线**：BBr₃ 管式炉扩散（预沉积 890℃/940℃ → Ar推进 → 热氧化），但**详细配方来自引用的 Benick 博士论文[38]**，本文未给出
- **Table 1 关系**：D3 R_sh=95 Ω/□ 最接近本文发射极 E2（88±1 Ω/□）。但本文是**钝化论文**而非扩散论文，几乎所有扩散工艺参数缺失，需查阅 Benick 2010 博士论文。
- **无法映射的原因**：本文为钝化研究，扩散配方仅引用外部文献。BSG厚度、扩散时间、气体流量、推进温度/时间均未给出。**8个工艺参数中仅 temp1 有部分值**。

### D4 — [40] Li et al. 2023, Sol. En. Mat. Sol. Cells 263

- **电池结构**：N型 TOPCon 太阳能电池（选择性发射极）
- **硼扩散路线**：低压 BCl₃ 管式扩散，四步法（预氧化→沉积→推进→后氧化），mask and etch-back 方法实现 SE
- **Table 1 关系**：R_sh=324 Ω/□ 未在本文出现（本文测量值范围 63-274 Ω/□）。本文聚焦于结深与 J₀ₘₑₜₐₗ 关系而非扩散配方。
- **无法映射的原因**：仅提及氧化温度>1000℃，其他扩散参数均缺失。未给出 BCl₃ 流量、沉积温度/时间、推进温度/时间。

### D5 — [41] Zhou et al. 2020, Applied Physics A 126

- **电池结构**：N型 TOPCon 太阳能电池（大面积 158.75 mm）
- **硼扩散路线**：BBr₃ 管式炉扩散（预沉积+推进两步法）
- **Table 1 关系**：BSG厚度~100 nm (Page 8); 预沉积860-900℃/50min; 推进960-1000℃/60min; O₂流量作为实验矩阵变量(50-300 sccm pre-dep; 0-3000 sccm drive-in)。R_sh=112 Ω/□ 在本文典型范围内（70-90 Ω/□ 为最优条件）。
- **部分可映射**：BSG厚度有值(~0.10 μm)，但温度/时间为范围而非单值。**需要对 Fig. 3/5/6 的 ECV 曲线逐条数字化才能精确匹配 D5 条件。**

### D6 — [42] Wang et al. 2023, Sol. En. Mat. Sol. Cells 253

- **电池结构**：N型 TOPCon 太阳能电池（选择性发射极 p⁺⁺/p⁺）
- **硼扩散路线**：BCl₃ 管式炉扩散，5个工艺参数矩阵（GBCl₃/Tdrive-in/Toxidation/GO₂/toxidation）
- **Table 1 关系**：基准条件(GBCl₃=90sccm, Tdrive-in=960℃, Toxidation=990℃, GO₂=3000sccm, t=60min)的 R_sh=125 Ω/□ ≠ 109 Ω/□。D6 的 R_sh=109 Ω/□ 可能来自某个特定变体条件。Table 1 中推进温度940℃的变体给出 R_sh=173 Ω/□，960℃给出142 Ω/□，985℃给出125 Ω/□。
- **部分可映射**：Table 1 给出完整5参数矩阵，但无BSG厚度、无硼源浓度(cm⁻³)、无沉积温度/时间、无N₂流量。

### D7 — [43] Chen et al. 2024, Prog. PV 32(3)

- **电池结构**：N型 TOPCon 太阳能电池（创新预扩散选择性发射极）
- **硼扩散路线**：BBr₃ 管式炉扩散，两步法（从 Figure 1 流程图提取）
- **Table 1 关系**：Figure 1 流程图给出：Step1 沉积 810℃/580s(9.67min)，Step2 氧化 925℃/270s(4.5min)；BSG~32nm；O₂:BCl₃ 比值 3.4:1(Step1)和 3.1:1(Step2)。**但 R_sh=269 Ω/□ 未在本文测量中出现**（本文范围 74-203 Ω/□）。D7 可能来自不同批次或仿真拟合结果。
- **部分可映射**：BSG厚度(~0.032 μm)、temp1(810℃)、time1(9.67min)、temp2(925℃)、time2(4.5min) 均有值。缺 c_boron 和绝对气体流量。

### D8 — [44] Peng et al. 2024, Mat. Sci. Semicon. Proc. 178

- **电池结构**：N型 TOPCon 太阳能电池（182mm, 130±10 μm）
- **硼扩散路线**：BCl₃ 管式炉扩散，**最完整的工艺配方**（Table 1: 8步全流程）
- **Table 1 关系**：基线工艺完整：进舟→预氧化(840℃,300s,N₂=3000,O₂=1000sccm)→沉积(845℃,540s,BCl₃=135sccm)→推进(945℃,300s)→氧化(1050℃,4500s,O₂=15000sccm)→冷却→出舟。基线 R_sh=129-137 Ω/□ ≠ 69 Ω/□。N_peak=1.84×10¹⁹ 接近 D8 的 1.88×10¹⁹。**z_p=0.10 μm 与本文结深0.8-1.0 μm严重不符。**
- **最接近可映射**：8步完整配方给出几乎全部参数。缺 BSG厚度（但氧化后形成~95nm SiO₂）和硼源浓度（仅 BCl₃ 流量135 sccm）。

### D9 — [45] Hong et al. 2024, IEEE JPV 14(4)

- **电池结构**：N型 TOPCon 太阳能电池（激光辅助选择性发射极）
- **硼扩散路线**：p⁺ 基线层=BCl₃ 管式扩散(~119 Ω/□)；p⁺⁺ 层=**B-doped Si浆料+532nm激光辐照(1.3 J/cm²)**，非标准管式扩散
- **Table 1 关系**：R_sh=152 Ω/□ **未在本文出现**。本文最佳 p⁺⁺ 层 R_sh=81.9 Ω/□，p⁺ 层~119 Ω/□。D9 可能来自参考文献 [19]（Hong 2022 Sol. Energy，B-doped Si paste + 热扩散而非激光）。
- **无法映射的原因**：激光掺杂工艺与管式扩散物理机制完全不同，无法映射 temp2/time2/F_N2/F_O2。p⁺ 基线扩散参数也未给出。

### D10 — [46] Liu et al. 2024, Solar Energy 271

- **电池结构**：N型 TOPCon 太阳能电池（一步法 p⁺/p⁺⁺ 共扩散）
- **硼扩散路线**：一步法：BCl₃(全域p⁺) + 丝网印刷B-doped Si浆料(局部p⁺⁺) **同时** 950℃/20min 热扩散，无预沉积+推进子结构
- **Table 1 关系**：一步扩散950℃/20min；p⁺ R_sh=108.69 Ω/□，p⁺⁺ R_sh=83.12 Ω/□。**R_sh=322 Ω/□ 未在本文出现**（3倍差异）。N_peak=8.68×10¹⁸ 接近 D10 的 7.91×10¹⁸ (+10%)。
- **部分可映射**：temp1=temp2=950℃, time1=time2=20min。但一步法无两步子结构，BSG厚度和硼源浓度缺失。
- **补充材料 (mmc1.pdf) 发现**：
  - 仅 1 页，仅包含 **Fig. S1** 的标题："Mean azimuth shift of boron atoms on each axis at different temperatures: (a)750℃, (b)850℃, (c)950℃, (d)1050℃"
  - 这是**分子动力学模拟（MD）图**，展示不同温度下硼原子在各轴上的平均方位偏移
  - ⚠️ **完全不含工艺参数或实验数据**，对参数提取无帮助

---

## 表 3：对标可行性评估

| ID | R_sheet_formula_ready (4核心参数) | curve_formula_ready (8工艺参数) | recommended_action | 详细说明 |
|---|---|---|---|---|
| D1 | **no** | **no** | 仅能做曲线形状对比 | 仅 temp2=950℃ 有值；athena_thick、c_boron、time2 均缺失。BCl₃流量10-20 ml/min 可部分替代 c_boron 但无法直接代入公式 |
| D2 | **partial** | **partial** | 需假设补全 | temp1/2 和 time1/2 有值(800/850℃×30min)；缺 thick、c_boron、F_N2、F_O2。浆料路线与管式扩散有本质差异，映射假设需谨慎 |
| D3 | **no** | **no** | 仅能做曲线形状对比 | 仅 temp1=890/940℃ 有部分值；其余6个参数全部缺失。需查阅 Benick PhD thesis 获取完整配方 |
| D4 | **no** | **no** | 仅能做曲线形状对比 | 仅 temp2~1000℃ 有近似值；7个参数缺失。论文聚焦器件性能而非工艺配方 |
| D5 | **partial** | **partial** | 需假设补全+ECV数字化 | BSG thick~0.10μm 有值；temp1/2 和 time1/2 有范围值；O₂ 流量有矩阵值。但温度/时间为范围而非单值，需数字化 Fig.3/5/6 才能精确匹配 D5 |
| D6 | **partial** | **partial** | 需假设补全 | Table 1 有5参数矩阵(temp2=960℃, O₂=3000sccm)；缺 thick、c_boron、temp1、time1。基准 R_sh=125≠109，需确定 D6 对应的变体条件 |
| D7 | **partial** | **partial** | 需假设补全 | Fig.1 给出 temp1=810℃, time1=9.67min, temp2=925℃, time2=4.5min, thick~0.032μm；缺 c_boron 和绝对气体流量。R_sh=269不在论文范围内 |
| D8 | **yes** (最接近) | **partial** | 可直接代入部分参数 + 需假设补全 c_boron | **最完整配方**：temp1=840℃, time1=5min, temp2=1050℃, time2=75min, F_N₂=3000sccm, F_O₂=1000/15000sccm；缺 thick(~95nm SiO₂可推断)和 c_boron(BCl₃=135sccm可替代) |
| D9 | **no** | **no** | 仅能做曲线形状对比 | 激光掺杂路线完全不同于管式扩散，无法映射标准参数。R_sh=152不在论文中，可能来自其他文献 |
| D10 | **partial** | **partial** | 需假设补全 | 一步法 temp=950℃, time=20min 有值；缺 thick、c_boron、F_N₂、F_O₂。一步法映射为 temp1=temp2 需验证 |

---

## 汇总统计

| 指标 | 数量 | ID列表 |
|------|------|--------|
| **可直接跑方阻公式** (R_sheet_formula_ready=yes) | **1** | D8 |
| **可部分跑方阻公式** (partial) | **5** | D2, D5, D6, D7, D10 |
| **仅能做曲线形状对比** (no) | **4** | D1, D3, D4, D9 |
| **可跑完整曲线流程** (curve_formula_ready=yes) | **0** | — |
| **可部分跑完整曲线流程** (partial) | **5** | D2, D5, D6, D7, D10 |

---

## 关键发现与建议

### 1. 文献与 Table 1 数据匹配性存疑

多个 ID 的已知输出值（R_sh、N_p、z_p）与对应论文中的实测值**不一致**：
- D1: R_sh=117 Ω/□ 在论文中未出现
- D4: R_sh=324 Ω/□ 在论文中未出现
- D7: R_sh=269 Ω/□ 在论文中未出现（论文范围 74-203 Ω/□）
- D8: R_sh=69 Ω/□ vs 论文 129-137 Ω/□；z_p=0.10 μm vs 论文 0.8-1.0 μm
- D9: R_sh=152 Ω/□ 在论文中未出现
- D10: R_sh=322 Ω/□ vs 论文 108.69 Ω/□

**可能原因**：主论文 Table 1 的双高斯拟合参数可能来自**仿真拟合**而非直接从文献实测值复制，或来自文献中特定子条件/不同批次。

### 2. 工艺参数缺失模式

- **athena_thick (BSG厚度)**：仅 D5(~0.10 μm) 和 D7(~0.032 μm) 有值；其余8篇均 NA
- **athena_c_boron (硼源浓度 cm⁻³)**：全部10篇均为 NA 或 inferred；文献通常给出 BCl₃/BBr₃ 流量而非源浓度
- **athena_F_N₂ / athena_F_O₂**：仅 D8 和 D6 有完整值；D5 和 D7 有部分值

### 3. 建议后续步骤

1. **最高优先级**：获取 Benick 2010 博士论文（D3 [39] 的硼扩散配方来源），可同时补全 D1 和 D3 的参数
2. **数字化 ECV 曲线**：对 D5 (Fig.3/5/6)、D6 (Fig.3)、D8 (Fig.2/4/5) 的 ECV 曲线进行数字化，提取精确掺杂剖面数据用于曲线形状验证
3. **确认 D-ID 与文献的精确映射**：主论文 Table 1 中 R_sh=117/324/269/69/152/322 Ω/□ 的确切来源需要与作者确认
4. **D8 为最佳对标起点**：拥有最完整8步工艺配方，可先以此验证 SR 方阻公式和曲线公式
5. **D9 需单独处理**：激光掺杂路线需建立独立映射框架，不适用标准管式扩散 SR 公式

---

---

## 补充材料 (Supplementary Material) 审查报告

> 检查时间：2026-06-25 | 新增文件：2份

### 补充材料清单

| ID | 文件名 | 格式 | 页数/内容 |
|---|--------|------|-----------|
| D1 [37] | `1-s2.0-S0038092X21007398-mmc1.docx` | DOCX (Word) | 10图 + J₀计算方法说明 |
| D10 [46] | `1-s2.0-S0038092X24001427-mmc1.pdf` | PDF | 1页（MD模拟图） |

### D1 补充材料详细审查

**文件**: `1-s2.0-S0038092X21007398-mmc1.docx` (Wang2021 Solar Energy 227 的 Supporting Information)

**包含内容**:
| 章节 | 内容 | 是否含工艺参数 |
|------|------|:---:|
| Section 1 | J₀ₑ,ₘₑₜₐₗ 计算公式及提取方法 | ❌ 器件表征方法 |
| Fig. S1 | J₀,total vs. f (金属化比例) 线性拟合 | ❌ J₀ 数据 |
| Fig. S2 | D1/D2/D3/N1/N2 发射极 J₀ 对比柱状图 | ❌ J₀ 数据 |
| Fig. S3(a-e) | Eff/Voc/Jsc/FF/Rser vs. 表面浓度 (固定结深0.5/0.63/0.8μm) | ⚠️ 含表面浓度范围数据 |
| Fig. S4(a-e) | Eff/Voc/Jsc/FF/Rser vs. 结深 (固定浓度~3e19/2.6e19) | ⚠️ 含结深范围数据 |

**从 Fig. S3/S4 可提取的辅助数据**:
- **表面浓度实验值** (x轴): ~1.15×10¹⁹, ~1.22×10¹⁹, ~1.49×10¹⁹, **~1.94×10¹⁹**, ~2.25×10¹⁹, ~2.63×10¹⁹, ~2.88×10¹⁹, ~3.14×10¹⁹, ~4×10¹⁹ atoms/cm³
  - ✅ **确认 1.94×10¹⁹ cm⁻³ 是本文实际使用的表面浓度之一**
- **结深实验值**: 0.5 μm, 0.63 μm, 0.8 μm（三组）
- **电池性能对应关系** (从散点图读取):
  - N_s=1.94e19 @ xj=0.8μm → Eff≈22.98%, Voc≈0.702V, FF≈81.07%
  - N_s=4e19 @ xj=0.63μm → Eff≈23.30% (最优)

**能否补全缺失参数**: ❌ **不能。** 补充材料聚焦于器件性能分析，不含扩散工艺配方。

### D10 补充材料详细审查

**文件**: `1-s2.0-S0038092X24001427-mmc1.pdf` (Liu2024 Solar Energy 271 的 Supporting Information)

**包含内容**:
| 章节 | 内容 | 是否含工艺参数 |
|------|------|:---:|
| Fig. S1 caption | 硼原子在750/850/950/1050℃下的平均方位偏移 MD 模拟图 | ❌ 分子动力学理论 |

**能否补全缺失参数**: ❌ **完全不能。** 仅1页MD模拟图，无任何实验或工艺信息。

### 结论

| 评估项 | 结果 |
|--------|------|
| D1 缺失参数是否可补全 | ❌ 否 — 补充材料仅含器件性能和J₀数据 |
| D10 缺失参数是否可补全 | ❌ 否 — 补充材料仅为MD模拟图 |
| 补充材料是否有其他价值 | D1: Fig.S3/S4的性能-掺杂关系数据可用于后续曲线形状验证；Fig.S1/S2可用于J₀模型校准 |
| 建议 | 两份补充材料对**工艺参数提取任务无实质帮助**，仍需通过其他途径获取缺失参数 |

---

## 每篇文献参数出处摘录

### D1 [37] — 关键原文摘录

- **BCl₃ 流量**："BCl₃ flow rates of 10, 15, and 20 ml/min" — Table 2, Page 5
- **氧化温度**："oxidation temperature of 950°C" — Table 2, Page 5
- **氧化时间**："oxidation times of 20, 55, and 70 min" — Table 2, Page 5
- **推进时间**："drive-in times of 0.5, 5, and 20 min" — Table 2, Page 5

### D2 [38] — 关键原文摘录

- **浆料扩散**："850°C for 30 min in N₂ atmosphere" — Page 2, L65; Page 6, L392-393
- **Si浆料组成**："Si powder (1 μm) + B powder (500 nm, 5 wt%)" — Page 2, L65
- **浆料厚度**："screen-printed on Si wafer, 200°C dried 30 s" — Page 2

### D3 [39] — 关键原文摘录

- **扩散温度**："890°C (E1, E2, E3, E4), 940°C (E1*)" — Table I, Page 2
- **钝化层**："Al₂O₃ by ALD, 1.53 Å/cycle at 130°C" — Page 6-7
- **配方来源**："BBr₃ diffusion recipe described in detail in Ref. [38]" — Page 2 (Benick PhD thesis)

### D4 [40] — 关键原文摘录

- **扩散步骤**："pre-oxidation → deposition of boron sources → drive-in → post-oxidation" — Page 2, L82-85
- **氧化温度**：">1000°C, equivalent to step ④" — Page 1, L84
- **硼源**："BCl₃ gas" (低压管式扩散)

### D5 [41] — 关键原文摘录

- **预沉积**："860-900°C, 50 min" — Page 2
- **推进**："960-1000°C, 60 min" — Page 2
- **BSG厚度**："~100 nm" — Page 8 (优化条件描述)
- **O₂流量**："50/80/100/150/200/300 sccm (pre-dep); 0/1000/2000/3000 sccm (drive-in)" — Table 1, Page 2

### D6 [42] — 关键原文摘录

- **基准工艺**：Table 1, Page 2: GBCl₃=90 sccm, Tdrive-in=960°C, Toxidation=990°C, GO₂=3000 sccm, toxidation=60 min
- **BCl₃流量范围**："60, 90, 120 sccm" — Table 1
- **推进温度范围**："940, 960, 985°C" — Table 1

### D7 [43] — 关键原文摘录

- **流程图 Figure 1**：Step1: 810℃, 沉积580s+380s=9.67min; Step2: 925℃, 氧化270s=4.5min
- **O₂:BCl₃ 比值**："3.4:1 (Step1), 3.1:1 (Step2)" — Figure 1 标注
- **BSG**："~32 nm" — Page 4 推断

### D8 [44] — 关键原文摘录

- **完整8步配方 Table 1**：
  - 进舟: 750℃, 500s, N₂=5000 sccm
  - 预氧化: 840℃, 300s, N₂=3000+O₂=1000 sccm
  - 沉积: 845℃, 540s, N₂=1300+O₂=800+BCl₃=135 sccm
  - 推进: 945℃, 300s, N₂=5000 sccm
  - 氧化: 1050℃, 4500s, O₂=15000 sccm
  - 冷却: 840℃, 1500s, N₂=5000 sccm
  - 出舟: 700℃, 600s, N₂=5000 sccm

### D9 [45] — 关键原文摘录

- **激光参数**："Nd:YVO₄, 532 nm, 80 μm spot, 20 m/s, 30-50 ns pulse, 1.3 J/cm²" — Page 4
- **浆料**："B-doped Si paste, screen-printed, 200°C 30s dried, 2-3 μm film" — Page 7
- **p⁺ 基线**："BCl₃ tube diffusion, R_sh≈119 Ω/□" — Page 7

### D10 [46] — 关键原文摘录

- **一步扩散**："950°C, 20 min" — Page 7, 最优条件
- **共扩散机制**："BCl₃ gas (全域p⁺) + screen-printed B-doped Si paste (局部p⁺⁺)" — Section 2
- **p⁺ 测量值**："R_sh=108.69 Ω/□, N_peak=8.68×10¹⁸ cm⁻³, junction depth=0.53 μm" — Page 7-8
