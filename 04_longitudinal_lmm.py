"""K=8 动态脑状态指标的组别 x 时间线性混合效应模型分析。

分析目的
--------
比较针刺组（Acupuncture）与假针刺组（Sham）从干预前（Pre）到干预后
（Post）的动态脑网络指标变化是否不同。本脚本分析 LEiDA 输出的三类指标：
1. fractional occupancy：每个 LEiDA 状态的占用率；
2. dwell time：每个 LEiDA 状态的平均驻留时间。
3. transition probability：状态之间的转换概率。

三类指标应先进行统计检验，再依据校正后的结果决定哪些内容进入正文图、
补充材料或不重点呈现；统计分析范围不应由预期作图内容预先决定。

样本与模型
----------
仅分析 condition 为 Acupuncture_Pre/Acupuncture_Post 或 Sham_Pre/Sham_Post
的受试者。脚本要求每名受试者恰好具有一次 Pre 和一次 Post 测量，防止由于
被试命名错误将纵向配对拆开。

对每一种指标的每个状态分别拟合随机截距线性混合效应模型：

    z(状态指标) ~ group * time + z(Age) + C(Sex) + (1 | subject)

其中 Sham 和 Pre 为参考水平；Sex 作为分类协变量处理。模型使用 REML
估计随机截距。目标效应为：

    group[T.Acupuncture]:time[T.Post]

该交互项 beta 表示针刺组相对于假针刺组的 Pre-to-Post 改变量差异
（difference-in-differences），因结局已经标准化，beta 为标准化效应量。
模型首先使用 BFGS 优化器拟合；若该优化器因数值问题失败，则使用 Powell
优化器重试。若随机截距方差接近 0，输出中标记为边界拟合，表示该结局几乎
没有可由随机截距解释的个体间方差，但仍保留交互效应估计供审查和报告。

多重比较
--------
对三个指标族分别进行 Benjamini-Hochberg FDR 校正：
occupancy 在 8 个状态间校正，dwell time 在 8 个状态间校正，
transition probability 在全部 8 x 8 = 64 个方向性转换项间校正。
FDR q < 0.05 标记为显著，0.05 <= q < 0.10 标记为趋势。

输入与输出
----------
输入：
    LEiDA_results/dynamics_metrics/k_8/occupancies_age_sex.csv
    LEiDA_results/dynamics_metrics/k_8/dwell_times_age_sex.csv
    LEiDA_results/dynamics_metrics/k_8/transitions_probabilities_age_sex.csv
输出：
    LEiDA_results/LMM_Results_new/LMM_occupancies_Results.csv
    LEiDA_results/LMM_Results_new/LMM_dwell_times_Results.csv
    LEiDA_results/LMM_Results_new/LMM_transitions_probabilities_Results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.multitest import multipletests


STATE_COLUMNS = [f"PL_state_{state}" for state in range(1, 9)]

METRICS = {
    "occupancies": {
        "filename": "occupancies_age_sex.csv",
        "outcomes": STATE_COLUMNS,
    },
    "dwell_times": {
        "filename": "dwell_times_age_sex.csv",
        "outcomes": STATE_COLUMNS,
    },
    "transitions_probabilities": {
        "filename": "transitions_probabilities_age_sex.csv",
        "outcomes": None,
    },
}
TARGET_EFFECT = "group[T.Acupuncture]:time[T.Post]"
RANDOM_VARIANCE_BOUNDARY = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run K=8 longitudinal mixed-effects models.")
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def load_longitudinal_data(input_path, outcome_columns):
    """读取并验证用于纵向 LMM 的针刺组和假针刺组数据。"""
    df = pd.read_csv(input_path)
    required = {"subject_id", "condition", "Age", "Sex", *outcome_columns}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{input_path.name} 缺少必要列：{missing}")

    data = df[df["condition"].str.contains("Acupuncture|Sham", na=False)].copy()
    data["sub_id"] = data["subject_id"].astype(str).str.rsplit("_", n=1).str[0]
    data["time"] = data["condition"].map(lambda value: "Post" if "Post" in value else "Pre")
    data["group"] = data["condition"].map(
        lambda value: "Acupuncture" if "Acupuncture" in value else "Sham"
    )

    data["time"] = pd.Categorical(data["time"], categories=["Pre", "Post"])
    data["group"] = pd.Categorical(data["group"], categories=["Sham", "Acupuncture"])
    data["Sex"] = pd.Categorical(data["Sex"])

    visit_counts = pd.crosstab(data["sub_id"], data["time"]).reindex(
        columns=["Pre", "Post"], fill_value=0
    )
    invalid = visit_counts[(visit_counts["Pre"] != 1) | (visit_counts["Post"] != 1)]
    if not invalid.empty:
        raise ValueError(f"有 {len(invalid)} 名受试者无法构成唯一的 Pre/Post 配对。")

    group_counts = data.groupby("sub_id", observed=True)["group"].nunique()
    if (group_counts != 1).any():
        raise ValueError("发现同一受试者被分入多个组别，请检查 condition 或 subject_id。")

    if data["Age"].std(ddof=1) == 0:
        raise ValueError("Age 无方差，无法作为标准化协变量进入模型。")

    return data


def group_mean(data, state, group, time):
    """返回指定组别和时间点的原始指标均值。"""
    selected = data[(data["group"] == group) & (data["time"] == time)][state]
    return selected.mean()


def fit_random_intercept_model(work):
    """拟合随机截距模型，并在首选优化器失败时进行数值稳定的重试。"""
    model = smf.mixedlm(
        "outcome_z ~ group * time + Age_z + C(Sex)",
        data=work,
        groups=work["sub_id"],
        re_formula="1",
    )
    errors = []
    for optimizer in ("bfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                warnings.simplefilter("always", UserWarning)
                fit = model.fit(reml=True, method=optimizer, maxiter=2000, disp=False)
            fit_warnings = sorted({str(item.message) for item in caught})
            if bool(fit.converged):
                return fit, optimizer, "；".join(fit_warnings)
            errors.append(f"{optimizer}: 未收敛")
        except (np.linalg.LinAlgError, ValueError) as error:
            errors.append(f"{optimizer}: {type(error).__name__} ({error})")

    raise RuntimeError("随机截距模型拟合失败；" + "；".join(errors))


def fit_metric(metric_name, input_path, outcome_columns):
    """对一个动态指标族的结局项拟合随机截距 LMM 并执行 FDR 校正。"""
    if outcome_columns is None:
        columns = pd.read_csv(input_path, nrows=0).columns
        outcome_columns = [column for column in columns if column.startswith("From_")]
        if len(outcome_columns) != 64:
            raise ValueError(
                f"{input_path.name} 应包含 64 个 From_* 转换概率列，实际找到 "
                f"{len(outcome_columns)} 个。"
            )

    data = load_longitudinal_data(input_path, outcome_columns)
    results = []

    print(
        f"\n正在分析 {metric_name}：{len(data)} 次观测，"
        f"{data['sub_id'].nunique()} 名受试者，{len(outcome_columns)} 项检验"
    )

    for state in outcome_columns:
        work = data.copy()
        outcome_sd = work[state].std(ddof=1)
        if np.isclose(outcome_sd, 0):
            raise ValueError(f"{metric_name} 的 {state} 无方差，无法拟合模型。")

        work["outcome_z"] = (work[state] - work[state].mean()) / outcome_sd
        work["Age_z"] = (work["Age"] - work["Age"].mean()) / work["Age"].std(ddof=1)

        fit, optimizer, fit_warnings = fit_random_intercept_model(work)

        if TARGET_EFFECT not in fit.params:
            raise KeyError(f"模型未生成目标交互项：{TARGET_EFFECT}")

        random_intercept_variance = float(fit.cov_re.iloc[0, 0])
        boundary_random_effect = random_intercept_variance < RANDOM_VARIANCE_BOUNDARY
        acupuncture_pre = group_mean(data, state, "Acupuncture", "Pre")
        acupuncture_post = group_mean(data, state, "Acupuncture", "Post")
        sham_pre = group_mean(data, state, "Sham", "Pre")
        sham_post = group_mean(data, state, "Sham", "Post")
        raw_did = (acupuncture_post - acupuncture_pre) - (sham_post - sham_pre)

        results.append(
            {
                "Metric": metric_name,
                "Brain_State": state,
                "Effect_Size_Beta": fit.params[TARGET_EFFECT],
                "Z_statistic": fit.tvalues[TARGET_EFFECT],
                "p_raw": fit.pvalues[TARGET_EFFECT],
                "Acupuncture_Pre_Mean": acupuncture_pre,
                "Acupuncture_Post_Mean": acupuncture_post,
                "Sham_Pre_Mean": sham_pre,
                "Sham_Post_Mean": sham_post,
                "Raw_Difference_in_Differences": raw_did,
                "N_observations": len(work),
                "N_subjects": work["sub_id"].nunique(),
                "Converged": fit.converged,
                "Optimizer": optimizer,
                "Random_Intercept_Variance": random_intercept_variance,
                "Boundary_Random_Effect": boundary_random_effect,
                "Fit_Warnings": fit_warnings,
            }
        )

    result_df = pd.DataFrame(results)
    result_df["p_FDR"] = multipletests(result_df["p_raw"], method="fdr_bh")[1]
    result_df["Sig_Status"] = result_df["p_FDR"].map(
        lambda value: "*" if value < 0.05 else ("趋势" if value < 0.10 else "ns")
    )
    return result_df


def main():
    """运行三类动态指标的 LMM 分析并保存结果。"""
    args = parse_args()
    input_dir = args.input_directory.resolve()
    output_dir = args.output_directory.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("开始运行 K=8 动态脑状态指标的组别 x 时间 LMM 分析...")
    print("模型：z(指标) ~ group * time + z(Age) + C(Sex) + (1 | subject)")

    for metric_name, config in METRICS.items():
        input_path = input_dir / config["filename"]
        if not input_path.exists():
            raise FileNotFoundError(f"缺少指标文件：{config['filename']}")
        result_df = fit_metric(metric_name, input_path, config["outcomes"])
        output_path = output_dir / f"LMM_{metric_name}_Results.csv"
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        display_columns = [
            "Brain_State",
            "Effect_Size_Beta",
            "Z_statistic",
            "p_raw",
            "p_FDR",
            "Sig_Status",
        ]
        print(result_df[display_columns].sort_values("p_raw").to_string(index=False))
        boundary_count = int(result_df["Boundary_Random_Effect"].sum())
        if boundary_count:
            print(f"注意：{metric_name} 有 {boundary_count} 项随机截距方差接近 0，已在结果表中标记。")
        print(f"{metric_name} 结果已保存。")

    print("\n分析完成。FDR 校正分别在三个指标族内部进行，转换概率包含 64 项检验。")


if __name__ == "__main__":
    main()
