# LEiDA 论文分析代码

本目录仅包含分析代码，不包含绘图、真实数据、受试者编号、原始文件名或本机路径。所有输入与输出位置均须通过命令行显式指定，仓库中不要放置任何研究数据或结果表。

## 分析模块

| 脚本 | 内容 |
|---|---|
| `00_attach_demographics.py` | 将年龄和性别合并到 K=8 动态指标表 |
| `01_prepare_leida_inputs.py` | 将 DPABI ROI 时间序列转换为匿名化 pyleida 输入 |
| `02_run_leida_analysis.py` | 合并样本的 LEiDA 分解、K=2–20 聚类与动态指标计算 |
| `03_export_k8_state_vectors.py` | 导出 K=8 状态向量 |
| `04_longitudinal_lmm.py` | 组别 × 时间线性混合效应模型 |
| `05_prespecified_cross_sectional_ancova.py` | 四个预设横断面 ANCOVA 比较 |
| `06_ahba_pls_shared.py` | AHBA、PLS1、空间零模型和基因富集共用函数 |
| `07_ahba_surface_spin.py` | 预先指定 State 7 的正式 surface-spin PLS1 分析 |
| `08_cell_type_enrichment.py` | State 7 的 Seidlitz 细胞类型富集 |
| `09_cortical_layer_enrichment.py` | State 7 的 L1–L6 皮层层级富集 |
| `10_normality_tests.py` | DSRS28 回归变量的正态性描述 |
| `11_longterm_dsrs_prediction.py` | 长期 DSRS 负担的探索性预测分析 |

## 隐私要求

- 不要把原始数据、人口学表、ID映射表、逐人预测或分析结果提交到 Git。
- 脚本 01 要求私有映射表包含 `source_group,source_id,anonymous_id,intervention`。该表只在本地读取，不会被复制到输出目录。
- `anonymous_id` 只能使用无含义的字母、数字或连字符，不要使用姓名、住院号、扫描号或可逆编码。
- PSD 记录的 `intervention` 只能为 `Acupuncture` 或 `Sham`；其他组可留空。
- 脚本 11 不读取任何真实ID字段，也没有逐人结果导出选项。
- `.gitignore` 默认排除 CSV、Excel、MAT、NIfTI、pickle 和常见数据/结果目录。若将代码复制到另一个仓库，必须同时复制并复核 `.gitignore`。

脚本 01 的私有映射表示意如下；示意值不对应任何真实受试者：

```text
source_group,source_id,anonymous_id,intervention
PSD,source_a,P-A001,Acupuncture
PSD,source_b,P-A002,Sham
HC,source_c,H-A001,
```

## State 7 AHBA 分析

State 7 在 AHBA 分析前已经预先指定。正式 surface-spin 分析只读取 `State_7`：

- AHBA 表达矩阵 `X`：105 个 Brainnetome 左侧皮层 ROI × 基因数；
- State 7 空间向量 `y`：105 × 1；
- ROI PLS1 score：105 × 1；
- 基因权重：每个 AHBA 背景基因对应一个权重及其 bootstrap 统计量。

因此，State 7 的整体空间经验 P 值不做跨状态校正。后续细胞类型富集仍在 2 个方向 × 7 类细胞的 14 项检验内做 BH-FDR；皮层层级富集仍在 2 个方向 × 6 个层级的 12 项检验内做 BH-FDR。

## 运行方式

安装固定版本依赖：

```bash
python -m pip install -r requirements.txt
```

所有路径均用占位符表示，运行时替换成本地私有位置。例如：

```bash
python 01_prepare_leida_inputs.py \
  --psd-pre <private_psd_pre> \
  --psd-post <private_psd_post> \
  --no-psd <private_no_psd> \
  --healthy-controls <private_hc> \
  --id-map <private_id_map.csv> \
  --output <private_pyleida_input>
```

```bash
python 07_ahba_surface_spin.py \
  --eigenvectors <k8_state_vectors.csv> \
  --lut <brainnetome_lut.txt> \
  --atlas <brainnetome_atlas.nii.gz> \
  --abagen-data <private_abagen_data> \
  --surface-annot <lh_annotation_file> \
  --surface-sphere <lh_sphere_file> \
  --seidlitz-file <cell_type_reference.xlsx> \
  --output-directory <private_state7_results>
```

`06_ahba_pls_shared.py` 是脚本 07 和 08 的共用计算模块；正式主分析从脚本 07 启动。脚本 08、09 和 11 同样要求显式传入输入与输出目录，可运行 `python <脚本名> --help` 查看参数。

## 发布前检查

发布 GitHub/Zenodo 前至少执行：

```bash
python -m compileall .
git status --short
```

还应人工确认仓库中没有 CSV、Excel、MAT、NIfTI、pickle、日志、缓存、结果目录以及任何可识别受试者的信息。
