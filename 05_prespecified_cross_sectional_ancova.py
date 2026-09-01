"""针刺组两个时间点与 HC/noPSD 的四个预设组间比较分析。

研究问题
--------
本脚本用于比较针刺组在干预前后分别与两个参照组（HC 和 noPSD）的动态脑
状态指标差异。分析严格限定为以下四个预设比较：

    1. Acupuncture_Post vs HC
    2. Acupuncture_Post vs noPSD
    3. Acupuncture_Pre  vs HC
    4. Acupuncture_Pre  vs noPSD

注意：这些比较回答的是“针刺组某一时间点与参照人群是否存在差异”，而不是
“针刺治疗效果是否优于假针刺”。后一个纵向干预问题应使用 group x time
线性混合效应模型（LMM）回答。

分析指标与转换方式
------------------
使用 K=8 的三类 LEiDA 动态指标：

    1. Fractional occupancy（8 个状态）：
       在原始比例尺度上分析，不做 log1p 转换。其模型系数可直接转换为
       调整后的百分点差（beta * 100）。

    2. Dwell time（8 个状态）：
       使用 log1p(dwell time) 作为因变量，以降低右偏值的影响并容纳可能
       出现的 0 值。模型 beta 位于 log(dwell time + 1) 尺度，同时输出
       100 * (exp(beta) - 1)，表示 (dwell time + 1) 的相对差异百分比。

    3. Transition probability（8 x 8 = 64 条有向转换）：
       在原始概率尺度上分析，不做 log1p 转换，不基于观察到的零值比例删除
       路径。模型系数可解释为调整后的转换概率百分点差。由于转换概率可能
       含较多 0 值，此部分应作为探索性结果谨慎解释；输出中提供每条路径的
       零值比例供审查。

统计模型
--------
对每一个指标/状态（或转换路径）及每一个预设比较，单独拟合协方差分析模型：

    outcome ~ comparison_group + Age_centered + C(Sex)

其中 comparison_group 编码为：当前比较中第一个组 = 1，参照组 = 0。
因此 Adjusted_Beta > 0 表示第一个组的调整后指标高于参照组，反之亦然。
Age 以当前比较样本的均值中心化；Sex 作为分类协变量处理。

为减轻组间方差不齐及有界结局造成的标准误偏差，p 值和 95% 置信区间使用
HC3 异方差稳健标准误计算，并采用 t 分布推断。

多重比较校正
------------
本分析的科学问题是：针刺组在某一个明确时间点与某一个明确参照人群相比，
哪些具体脑状态存在差异。因此，四个预设比较分别视为四个事先定义的比较问题，
Benjamini-Hochberg FDR 校正在每个预设比较内部执行：

    - Occupancy:              每个比较校正 8 个状态（每个校正家族 n = 8）
    - Dwell time:             每个比较校正 8 个状态（每个校正家族 n = 8）
    - Transition probability: 每个比较校正 64 条路径（每个校正家族 n = 64）

该策略不能被解释为在四个比较合并后的总体 32 项检验范围内控制 FDR，而是
在每一个预设临床比较问题内控制状态层面的 FDR。转换概率由于检验数量较多
且数据稀疏，仅作为探索性结果解释。

输入与输出
----------
输入目录：
    LEiDA_results/dynamics_metrics/k_8/

输出目录（新建，不覆盖旧版探索性两两比较结果）：
    LEiDA_results/prespecified_cross_sectional_ancova/

输出文件：
    occupancies_prespecified_4_comparisons_ancova_fdr.csv
    dwell_times_prespecified_4_comparisons_ancova_fdr.csv
    transitions_probabilities_prespecified_4_comparisons_ancova_fdr.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


COMPARISONS = [
    ("Acupuncture_Post", "HC"),
    ("Acupuncture_Post", "noPSD"),
    ("Acupuncture_Pre", "HC"),
    ("Acupuncture_Pre", "noPSD"),
]

STATE_COLUMNS = [f"PL_state_{state}" for state in range(1, 9)]

METRIC_CONFIG = {
    "occupancies": {
        "input": "occupancies_age_sex.csv",
        "output": "occupancies_prespecified_4_comparisons_ancova_fdr.csv",
        "columns": STATE_COLUMNS,
        "transformation": "raw_probability",
    },
    "dwell_times": {
        "input": "dwell_times_age_sex.csv",
        "output": "dwell_times_prespecified_4_comparisons_ancova_fdr.csv",
        "columns": STATE_COLUMNS,
        "transformation": "log1p",
    },
    "transitions_probabilities": {
        "input": "transitions_probabilities_age_sex.csv",
        "output": "transitions_probabilities_prespecified_4_comparisons_ancova_fdr.csv",
        "columns": None,
        "transformation": "raw_probability",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run four prespecified cross-sectional ANCOVAs.")
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def load_data(input_path):
    """读取并核对输入表中的基本变量与四个分析组。"""
    data = pd.read_csv(input_path)
    required = {"subject_id", "condition", "Age", "Sex"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{input_path.name} 缺少必要列：{missing}")

    required_groups = sorted({group for comparison in COMPARISONS for group in comparison})
    missing_groups = sorted(set(required_groups).difference(data["condition"].dropna().unique()))
    if missing_groups:
        raise ValueError(f"{input_path.name} 缺少比较所需组别：{missing_groups}")

    if data[["Age", "Sex"]].isna().any().any():
        raise ValueError(f"{input_path.name} 的 Age 或 Sex 含缺失值，请先处理后再运行。")
    return data


def outcome_columns(data, configured_columns, metric_name):
    """获取需要分析的状态或转换路径列，并确保 K=8 结构完整。"""
    if configured_columns is not None:
        missing = sorted(set(configured_columns).difference(data.columns))
        if missing:
            raise ValueError(f"{metric_name} 缺少状态列：{missing}")
        return configured_columns

    columns = [column for column in data.columns if column.startswith("From_")]
    if len(columns) != 64:
        raise ValueError(f"{metric_name} 应包含 64 条转换路径，实际找到 {len(columns)} 条。")
    return columns


def transform_outcome(values, transformation):
    """按预先规定的尺度转换结局变量。"""
    if transformation == "log1p":
        if (values < 0).any():
            raise ValueError("dwell time 出现负值，无法进行 log1p 转换。")
        return np.log1p(values)
    if transformation == "raw_probability":
        if ((values < 0) | (values > 1)).any():
            raise ValueError("概率指标应位于 0 到 1 之间，请检查输入数据。")
        return values.astype(float)
    raise ValueError(f"无法识别的转换方式：{transformation}")


def fit_one_comparison(data, outcome, metric_name, transformation, group_1, group_2):
    """对单个结局和单个预设组间比较拟合调整后的 ANCOVA。"""
    subset = data[data["condition"].isin([group_1, group_2])].copy()
    group_sizes = subset["condition"].value_counts()
    n_group_1 = int(group_sizes.get(group_1, 0))
    n_group_2 = int(group_sizes.get(group_2, 0))
    if n_group_1 == 0 or n_group_2 == 0:
        raise ValueError(f"{outcome}: {group_1} vs {group_2} 存在空组。")

    duplicated = subset.duplicated(subset=["subject_id", "condition"])
    if duplicated.any():
        raise ValueError(f"{outcome}: {group_1} vs {group_2} 存在重复 subject_id 记录。")

    subset["comparison_group"] = (subset["condition"] == group_1).astype(int)
    subset["Age_centered"] = subset["Age"] - subset["Age"].mean()
    subset["Sex"] = pd.Categorical(subset["Sex"])
    subset["outcome_model"] = transform_outcome(subset[outcome], transformation)

    model = smf.ols(
        "outcome_model ~ comparison_group + Age_centered + C(Sex)",
        data=subset,
    ).fit(cov_type="HC3", use_t=True)

    parameter = "comparison_group"
    beta = float(model.params[parameter])
    ci_lower, ci_upper = model.conf_int().loc[parameter].astype(float)
    t_stat = float(model.tvalues[parameter])
    p_raw = float(model.pvalues[parameter])
    df_resid = float(model.df_resid)
    partial_eta_squared = (t_stat**2) / (t_stat**2 + df_resid)

    row = {
        "Metric": metric_name,
        "Outcome": outcome,
        "Comparison": f"{group_1}_vs_{group_2}",
        "Group_1": group_1,
        "Group_2": group_2,
        "N_Group_1": n_group_1,
        "N_Group_2": n_group_2,
        "Outcome_Scale": transformation,
        "Group_1_Raw_Mean": subset.loc[subset["condition"] == group_1, outcome].mean(),
        "Group_2_Raw_Mean": subset.loc[subset["condition"] == group_2, outcome].mean(),
        "Raw_Mean_Difference": (
            subset.loc[subset["condition"] == group_1, outcome].mean()
            - subset.loc[subset["condition"] == group_2, outcome].mean()
        ),
        "Adjusted_Beta": beta,
        "CI95_Lower": ci_lower,
        "CI95_Upper": ci_upper,
        "Robust_T": t_stat,
        "DF_Residual": df_resid,
        "P_raw": p_raw,
        "Partial_Eta_Squared": partial_eta_squared,
        "HC3_Robust_SE": float(model.bse[parameter]),
    }

    if transformation == "raw_probability":
        row["Adjusted_Difference_Percentage_Points"] = beta * 100
    else:
        row["DwellPlus1_Relative_Difference_Percent"] = (np.exp(beta) - 1) * 100

    if metric_name == "transitions_probabilities":
        row["Zero_Ratio_All_Subjects"] = float((data[outcome] == 0).mean())
        row["Sparse_Over_90Pct_Zero"] = row["Zero_Ratio_All_Subjects"] > 0.90

    return row


def analyze_metric(metric_name, config):
    """完成一个指标的预设比较，并在每个比较内部执行状态/路径层面 FDR 校正。"""
    data = load_data(config["input"])
    columns = outcome_columns(data, config["columns"], metric_name)
    rows = []
    for outcome in columns:
        for group_1, group_2 in COMPARISONS:
            rows.append(
                fit_one_comparison(
                    data,
                    outcome,
                    metric_name,
                    config["transformation"],
                    group_1,
                    group_2,
                )
            )

    results = pd.DataFrame(rows)
    results["FDR_Family"] = results["Comparison"]
    results["P_fdr"] = np.nan
    results["FDR_Family_Size"] = 0
    for comparison, index in results.groupby("Comparison").groups.items():
        results.loc[index, "P_fdr"] = multipletests(
            results.loc[index, "P_raw"], method="fdr_bh"
        )[1]
        results.loc[index, "FDR_Family_Size"] = len(index)

    results["Significant_Raw"] = results["P_raw"] < 0.05
    results["Significant_FDR"] = results["P_fdr"] < 0.05
    results["Significance"] = np.where(results["Significant_FDR"], "*", "ns")
    return results.sort_values(["Comparison", "P_fdr", "P_raw", "Outcome"])


def main():
    """运行三类动态指标的四个预设 ANCOVA 比较并保存新结果。"""
    args = parse_args()
    input_dir = args.input_directory.resolve()
    output_dir = args.output_directory.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("开始进行针刺组 Pre/Post 与 HC/noPSD 的四个预设组间比较...")
    print("模型：outcome ~ comparison_group + Age_centered + C(Sex)，HC3 稳健标准误")

    for metric_name, config in METRIC_CONFIG.items():
        run_config = dict(config)
        run_config["input"] = input_dir / config["input"]
        if not run_config["input"].exists():
            raise FileNotFoundError(f"缺少指标文件：{config['input']}")
        results = analyze_metric(metric_name, run_config)
        output_path = output_dir / config["output"]
        results.to_csv(output_path, index=False, encoding="utf-8-sig")

        family_sizes = sorted(results["FDR_Family_Size"].unique())
        print(f"\n{metric_name}: {len(results)} 项结果；FDR 在每个预设比较内分别校正，家族大小 = {family_sizes}")
        print(f"FDR q < 0.05 的结果数：{int(results['Significant_FDR'].sum())}")
        preview_columns = ["Outcome", "Comparison", "Adjusted_Beta", "P_raw", "P_fdr", "Significance"]
        print(results[preview_columns].head(10).to_string(index=False))
        print(f"{metric_name} 结果已保存。")

    print("\n分析完成。请将本脚本生成的新目录作为修正后的正式 pairwise 结果来源。")


if __name__ == "__main__":
    main()
