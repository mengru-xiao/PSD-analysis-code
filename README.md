# LEiDA 论文分析代码

本目录仅包含分析代码，不包含绘图、真实数据、受试者编号、原始文件名或本机路径。所有输入与输出位置均须通过命令行显式指定。

## 分析模块

| 脚本 | 内容 |
|---|---|
| `00_attach_demographics.py` | 将年龄和性别合并到 K=8 动态指标表 |
| `01_prepare_leida_inputs.py` | 将时间序列转换为匿名化 pyleida 输入 |
| `02_run_leida_analysis.py` | 合并样本的 LEiDA 分解、K=2–20 聚类与动态指标计算 |
| `03_export_k8_state_vectors.py` | 导出 K=8 状态向量 |
| `04_longitudinal_lmm.py` | 组别 × 时间线性混合效应模型 |
| `05_prespecified_cross_sectional_ancova.py` | 预设横断面 ANCOVA 比较 |
| `06_ahba_pls_shared.py` | AHBA、PLS1、空间零模型和基因富集共用函数 |
| `07_ahba_surface_spin.py` | 显著脑状态 surface-spin PLS1 分析 |
| `08_cell_type_enrichment.py` | Seidlitz 细胞类型富集 |
| `09_cortical_layer_enrichment.py` | 皮层层级富集 |
| `10_normality_tests.py` | 回归变量的正态性描述 |
| `11_longterm_dsrs_prediction.py` | 长期 DSRS 负担的探索性预测分析 |

## 隐私要求

脚本 01 的私有映射表示意如下；示意值不对应任何真实受试者：

```text
source_group,source_id,anonymous_id,intervention
PSD,source_a,P-A001,Acupuncture
PSD,source_b,P-A002,Sham
HC,source_c,H-A001,
```


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


