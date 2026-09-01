"""EA 组基线动态脑网络指标预测长期 DSRS 恢复轨迹的探索性分析。

这是供公开代码仓库使用的纯分析版本，不包含绘图代码、真实数据、受试者
编号或本机绝对路径，只保存模型层面的汇总结果。脚本不读取真实ID，也不
提供逐人终点或折外预测导出功能。

研究问题
--------
本脚本用于探索：在电针（EA）治疗开始前，基线临床信息联合基线动态
脑网络停留时间（dwell time, DT）或占用率（fractional occupancy,
FO），能否预测治疗后较长期的 DSRS 症状负担轨迹。

本分析所用候选模型来源
----------------------
前期 LMM 分析提示 State 2、State 6 和 State 7 的 DT 与 FO 值得进一步
考察；随后的模型筛查中，针对长期治疗后轨迹，加入这三个状态基线 DT
的模型表现相对更好。但 DT 的优势来自同一数据内的筛查结果，并不构成
排除 FO 的充分依据。因此，本脚本将 DT 与 FO 作为两个平行的探索性
脑特征集合进行同流程比较，不再扫描额外脑状态、更多终点或更多模型。

重要解释边界
------------
该模型是在既有数据上经过探索后选定，属于探索性分析，不能表述为已经
获得独立验证的预测模型。报告结果时需要同时呈现临床参照模型和加入
脑指标后的增强模型，避免只展示筛选后表现更好的结果。

临床终点
--------
真实访视时间为第 0、7、14、28、42、98、182 天。终点定义为：

    ``Post_AUC_DSRS_Day7_182``
        第 7、14、28、42、98 和 182 天 DSRS 曲线下面积除以 175 天，
        表示治疗后第 7 至 182 天的平均 DSRS 症状负担。数值越低表示
        总体恢复轨迹越好。

该终点不包含基线 DSRS，因此在预测变量中加入基线 DSRS 不会产生
“目标变化量本身含有基线值”的数学耦合问题。由于部分受试者缺失长期
访视，分析使用上述六个随访点均完整的个体。

预定义预测变量
--------------
1. 临床参照模型 ``Clinical``：
       ``DSRS_T1 + Age + Sex``；
2. DT 脑指标描述模型 ``Brain_DT``：
       ``State2_DT + State6_DT + State7_DT``；
3. DT 增强模型 ``Clinical_plus_DT``：
       ``DSRS_T1 + Age + Sex + State2_DT + State6_DT + State7_DT``。
4. FO 脑指标描述模型 ``Brain_FO``：
       ``State2_FO + State6_FO + State7_FO``；
5. FO 增强模型 ``Clinical_plus_FO``：
       ``DSRS_T1 + Age + Sex + State2_FO + State6_FO + State7_FO``。

``DSRS_T1`` 为治疗前基线 DSRS；``State*_DT`` 和 ``State*_FO`` 分别为
同一基线期的动态脑状态停留时间和占用率。性别编码沿用输入数据原始
编码，仅作为协变量进入模型。

预测模型与验证方式
------------------
模型固定为线性支持向量回归（linear SVR，``C=1``，``epsilon=0.1``），
这是前期模型筛查中长期终点的候选方案。样本量较小，因此采用留一交叉
验证（LOOCV）获得每位受试者的折外预测值。每一折的标准化参数仅根据
训练样本拟合，然后应用到被留出的测试个体，避免信息泄漏。

性能指标包括：

    - MAE：平均绝对误差；
    - RMSE：均方根误差；
    - CV R2：基于全部折外预测值的决定系数；
    - Pearson r：观察终点与折外预测值的线性相关，仅作预测一致性描述。

置换检验与增量评价
------------------
1. 单模型置换检验：
       随机打乱终点值与基线特征的对应关系，重新完成 LOOCV；检验实际
       RMSE 是否小于随机对应关系下的 RMSE，以及实际相关系数是否更大。
2. 脑指标增量置换检验：
       对 DT 和 FO 分别执行。保留临床变量与终点的真实关系，仅随机
       打乱待加入脑特征的受试者对应关系；比较增强模型相比 ``Clinical``
       的 RMSE 改善是否大于随机加入相应脑特征时可获得的改善。
3. 配对 Bootstrap 区间：
       对真实 LOOCV 预测产生的个体误差进行配对重采样，给出增强模型
       相对临床模型的 MAE/RMSE 改善的 95% 区间，用于描述不确定性。

输出结果
--------
输出保存在 ``--output-directory`` 指定的本地目录。默认输出包括：

    - ``model_performance_with_permutation.csv``：模型预测性能和置换 P 值；
    - ``incremental_brain_value.csv``：加入 DT 或 FO 相对临床模型的增量结果；
    - ``permutation_incremental_brain_distribution.csv``：DT/FO 增量置换分布；
    - ``full_sample_linear_svr_weights.csv``：全样本拟合后的标准化权重，
      仅用于探索性解释，不替代折外预测表现；
    - ``analysis_summary.txt``：便于论文撰写与后续绘图的结果摘要。

运行环境
--------
运行示例：

    python 11_longterm_dsrs_prediction.py --input /private/path/prediction_table.csv

默认执行 5000 次置换与 5000 次 Bootstrap。首次检查流程时可使用
``--permutations 100 --bootstraps 500`` 快速运行；用于最终探索性结果
时应保留默认次数或明确报告所采用的次数。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


RANDOM_SEED = 20260526
N_PERMUTATIONS = 5000
N_BOOTSTRAPS = 5000

POST_VISIT_DAYS = np.asarray([7, 14, 28, 42, 98, 182], dtype=float)
POST_DSRS_COLUMNS = ["DSRS_T2", "DSRS_T3", "DSRS_T4", "DSRS_T5", "DSRS_T6", "DSRS_T7"]
ENDPOINT = "Post_AUC_DSRS_Day7_182"

CLINICAL_FEATURES = ["DSRS_T1", "Age", "Sex"]
DT_FEATURES = ["State2_DT", "State6_DT", "State7_DT"]
FO_FEATURES = ["State2_FO", "State6_FO", "State7_FO"]
FEATURE_SETS = {
    "Clinical": CLINICAL_FEATURES,
    "Brain_DT": DT_FEATURES,
    "Clinical_plus_DT": CLINICAL_FEATURES + DT_FEATURES,
    "Brain_FO": FO_FEATURES,
    "Clinical_plus_FO": CLINICAL_FEATURES + FO_FEATURES,
}


def load_and_prepare_data(input_file: Path) -> pd.DataFrame:
    """读取预测表并构造第 7 至 182 天的平均 DSRS 负担终点。"""
    if not input_file.exists():
        raise FileNotFoundError("找不到长期预测输入表。")
    data = pd.read_csv(input_file, encoding="utf-8-sig")
    required = (
        set(POST_DSRS_COLUMNS)
        | set(CLINICAL_FEATURES)
        | set(DT_FEATURES)
        | set(FO_FEATURES)
    )
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"输入数据缺少分析字段：{missing}")
    analysis = data.dropna(
        subset=POST_DSRS_COLUMNS + CLINICAL_FEATURES + DT_FEATURES + FO_FEATURES
    ).copy()
    scores = analysis[POST_DSRS_COLUMNS].to_numpy(dtype=float)
    analysis[ENDPOINT] = (
        np.trapz(scores, x=POST_VISIT_DAYS, axis=1)
        / (POST_VISIT_DAYS[-1] - POST_VISIT_DAYS[0])
    )
    columns = (
        [ENDPOINT]
        + CLINICAL_FEATURES
        + DT_FEATURES
        + FO_FEATURES
        + POST_DSRS_COLUMNS
    )
    return analysis[columns].reset_index(drop=True)


def create_model() -> Pipeline:
    """建立固定的线性 SVR 管线；标准化只会在训练折内拟合。"""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="linear", C=1.0, epsilon=0.1)),
        ]
    )


def loocv_predict(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """对固定模型生成无信息泄漏的 LOOCV 折外预测值。"""
    predictions = np.full(len(y), np.nan, dtype=float)
    for train_index, test_index in LeaveOneOut().split(x):
        model = create_model()
        model.fit(x[train_index], y[train_index])
        predictions[test_index] = model.predict(x[test_index])[0]
    return predictions


def calculate_performance(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """根据折外预测值计算性能指标。"""
    prediction_r, prediction_p = stats.pearsonr(y, predicted)
    return {
        "MAE": float(mean_absolute_error(y, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(y, predicted))),
        "CV_R2": float(r2_score(y, predicted)),
        "Prediction_r": float(prediction_r),
        "Prediction_r_descriptive_P": float(prediction_p),
    }


def model_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    observed_metrics: dict[str, float],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """打乱终点标签，评价单个模型的预测性能是否优于随机关系。"""
    permuted_rmse = np.empty(N_PERMUTATIONS, dtype=float)
    permuted_r = np.empty(N_PERMUTATIONS, dtype=float)
    for index in range(N_PERMUTATIONS):
        permuted_y = rng.permutation(y)
        predicted = loocv_predict(x, permuted_y)
        current = calculate_performance(permuted_y, predicted)
        permuted_rmse[index] = current["RMSE"]
        permuted_r[index] = current["Prediction_r"]
    p_rmse = (1.0 + np.sum(permuted_rmse <= observed_metrics["RMSE"])) / (N_PERMUTATIONS + 1.0)
    p_r = (1.0 + np.sum(permuted_r >= observed_metrics["Prediction_r"])) / (N_PERMUTATIONS + 1.0)
    return float(p_rmse), float(p_r)


def bootstrap_increment(
    y: np.ndarray,
    clinical_predicted: np.ndarray,
    enhanced_predicted: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """配对重采样折外预测误差，描述加入 DT 后误差改善的不确定性。"""
    n_subjects = len(y)
    delta_rmse = np.empty(N_BOOTSTRAPS, dtype=float)
    delta_mae = np.empty(N_BOOTSTRAPS, dtype=float)
    for index in range(N_BOOTSTRAPS):
        sample = rng.integers(0, n_subjects, size=n_subjects)
        y_sample = y[sample]
        clinical_sample = clinical_predicted[sample]
        enhanced_sample = enhanced_predicted[sample]
        delta_rmse[index] = (
            np.sqrt(mean_squared_error(y_sample, clinical_sample))
            - np.sqrt(mean_squared_error(y_sample, enhanced_sample))
        )
        delta_mae[index] = (
            mean_absolute_error(y_sample, clinical_sample)
            - mean_absolute_error(y_sample, enhanced_sample)
        )
    return {
        "Delta_RMSE_CI_Lower": float(np.percentile(delta_rmse, 2.5)),
        "Delta_RMSE_CI_Upper": float(np.percentile(delta_rmse, 97.5)),
        "Delta_MAE_CI_Lower": float(np.percentile(delta_mae, 2.5)),
        "Delta_MAE_CI_Upper": float(np.percentile(delta_mae, 97.5)),
    }


def incremental_brain_permutation_test(
    data: pd.DataFrame,
    y: np.ndarray,
    clinical_predicted: np.ndarray,
    observed_delta_rmse: float,
    added_features: list[str],
    added_label: str,
    rng: np.random.Generator,
) -> tuple[float, pd.DataFrame]:
    """在保留临床关系的前提下，检验指定脑特征加入后的 RMSE 改善。"""
    clinical = data[CLINICAL_FEATURES].to_numpy(dtype=float)
    brain_features = data[added_features].to_numpy(dtype=float)
    clinical_rmse = float(np.sqrt(mean_squared_error(y, clinical_predicted)))
    rows = []
    for index in range(N_PERMUTATIONS):
        permuted_brain = brain_features[rng.permutation(len(brain_features))]
        enhanced_x = np.column_stack([clinical, permuted_brain])
        predicted = loocv_predict(enhanced_x, y)
        enhanced_rmse = float(np.sqrt(mean_squared_error(y, predicted)))
        rows.append(
            {
                "Brain_Feature_Set": added_label,
                "Permutation": index + 1,
                "Permuted_Enhanced_RMSE": enhanced_rmse,
                "Delta_RMSE_Clinical_Minus_Permuted_Enhanced": clinical_rmse - enhanced_rmse,
            }
        )
    distribution = pd.DataFrame(rows)
    p_value = (
        1.0
        + np.sum(
            distribution["Delta_RMSE_Clinical_Minus_Permuted_Enhanced"].to_numpy()
            >= observed_delta_rmse
        )
    ) / (N_PERMUTATIONS + 1.0)
    return float(p_value), distribution


def full_sample_weights(data: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """输出全样本标准化空间中的线性 SVR 权重，仅供探索性解释。"""
    rows = []
    for model_name, features in FEATURE_SETS.items():
        x = data[features].to_numpy(dtype=float)
        model = create_model()
        model.fit(x, y)
        coefficients = model.named_steps["svr"].coef_.reshape(-1)
        for feature, coefficient in zip(features, coefficients):
            rows.append(
                {
                    "Model": model_name,
                    "Feature": feature,
                    "Standardized_SVR_Weight": float(coefficient),
                    "Interpretation_Note": "仅供探索性描述；预测能力应依据 LOOCV 结果",
                }
            )
    return pd.DataFrame(rows)


def write_summary(
    data: pd.DataFrame,
    performance: pd.DataFrame,
    increment: pd.DataFrame,
    output_dir: Path,
) -> None:
    """将适合查看和后续绘图决策的结论写入文本摘要。"""
    text = [
        "EA 组基线临床变量联合 State 2/6/7 DT 或 FO 预测长期 DSRS 轨迹：探索性分析摘要",
        "=" * 78,
        "",
        "方法：",
        "终点为第7至182天 DSRS 曲线下面积/175天，即治疗后平均 DSRS 症状负担；",
        "终点不包含基线 DSRS。模型固定为 linear SVR (C=1, epsilon=0.1)，",
        "采用 LOOCV，并将标准化限定在每个训练折内。",
        "",
        f"纳入长期随访完整样本数：N = {len(data)}",
        "",
        "五种预定义模型的 LOOCV 表现：",
        performance[
            [
                "Model",
                "Features",
                "N",
                "MAE",
                "RMSE",
                "CV_R2",
                "Prediction_r",
                "Permutation_P_RMSE",
            ]
        ].to_string(index=False),
        "",
        "核心增量比较（DT 与 FO 增强模型分别相对于 Clinical）：",
        increment.to_string(index=False),
        "",
        "解释提醒：",
        "DT 与 FO 均作为平行探索特征集合；比较二者时需结合增强模型误差、",
        "相对于 Clinical 的增量置换 P 值以及 Bootstrap 区间判断。",
        "该分析为前期筛查后的探索性候选模型比较；任何正向结果仍需独立样本验证，",
        "不应将同一数据内的性能直接表述为可推广的临床预测准确度。",
    ]
    (output_dir / "analysis_summary.txt").write_text("\n".join(text), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    """读取私有输入、输出位置及随机重复次数。"""
    parser = argparse.ArgumentParser(description="长期 DSRS 轨迹的基线脑指标探索性预测分析")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="私有预测 CSV；不得加入公开仓库。",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="公开代码目录之外的汇总结果目录。",
    )
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS, help="置换检验次数")
    parser.add_argument("--bootstraps", type=int, default=N_BOOTSTRAPS, help="Bootstrap 重采样次数")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="所有随机过程的基础种子")
    return parser.parse_args()


def main() -> None:
    """运行固定候选模型的完整探索性分析并保存全部结果。"""
    global N_PERMUTATIONS, N_BOOTSTRAPS
    args = parse_args()
    N_PERMUTATIONS = args.permutations
    N_BOOTSTRAPS = args.bootstraps
    if N_PERMUTATIONS < 1 or N_BOOTSTRAPS < 1:
        raise ValueError("置换检验次数和 Bootstrap 次数必须为正整数。")
    input_file = args.input.resolve()
    output_dir = args.output_directory.resolve()
    print("开始进行长期 DSRS 轨迹的基线临床变量 + State 2/6/7 DT/FO 探索性预测分析...")
    print(f"本次设置：置换检验 {N_PERMUTATIONS} 次，Bootstrap {N_BOOTSTRAPS} 次。")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_prepare_data(input_file)
    y = data[ENDPOINT].to_numpy(dtype=float)

    performance_rows = []
    model_predictions: dict[str, np.ndarray] = {}
    for model_offset, (model_name, features) in enumerate(FEATURE_SETS.items()):
        print(f"正在评价模型：{model_name}")
        x = data[features].to_numpy(dtype=float)
        predicted = loocv_predict(x, y)
        model_predictions[model_name] = predicted
        observed_metrics = calculate_performance(y, predicted)
        permutation_rng = np.random.default_rng(args.seed + model_offset)
        p_rmse, p_r = model_permutation_test(x, y, observed_metrics, permutation_rng)
        performance_rows.append(
            {
                "Endpoint": ENDPOINT,
                "Model": model_name,
                "Features": " + ".join(features),
                "N": len(data),
                **observed_metrics,
                "N_Permutations": N_PERMUTATIONS,
                "Permutation_P_RMSE": p_rmse,
                "Permutation_P_Prediction_r": p_r,
            }
        )
    performance = pd.DataFrame(performance_rows).sort_values("RMSE")
    clinical_metrics = performance.loc[performance["Model"] == "Clinical"].iloc[0]
    increment_rows = []
    increment_distributions = []
    for index, (brain_label, brain_features) in enumerate([("DT", DT_FEATURES), ("FO", FO_FEATURES)]):
        enhanced_model = f"Clinical_plus_{brain_label}"
        enhanced_metrics = performance.loc[performance["Model"] == enhanced_model].iloc[0]
        delta_rmse = float(clinical_metrics["RMSE"] - enhanced_metrics["RMSE"])
        delta_mae = float(clinical_metrics["MAE"] - enhanced_metrics["MAE"])
        print(f"正在检验 {brain_label} 相对于临床参照模型的增量价值...")
        increment_rng = np.random.default_rng(args.seed + 100 + index)
        increment_p, increment_distribution = incremental_brain_permutation_test(
            data,
            y,
            model_predictions["Clinical"],
            delta_rmse,
            brain_features,
            brain_label,
            increment_rng,
        )
        increment_distributions.append(increment_distribution)
        bootstrap_rng = np.random.default_rng(args.seed + 200 + index)
        bootstrap_ci = bootstrap_increment(
            y,
            model_predictions["Clinical"],
            model_predictions[enhanced_model],
            bootstrap_rng,
        )
        increment_rows.append(
            {
                "Endpoint": ENDPOINT,
                "Reference_Model": "Clinical",
                "Enhanced_Model": enhanced_model,
                "Added_Brain_Features": " + ".join(brain_features),
                "N": len(data),
                "Reference_RMSE": clinical_metrics["RMSE"],
                "Enhanced_RMSE": enhanced_metrics["RMSE"],
                "Delta_RMSE_Reference_Minus_Enhanced": delta_rmse,
                "Reference_MAE": clinical_metrics["MAE"],
                "Enhanced_MAE": enhanced_metrics["MAE"],
                "Delta_MAE_Reference_Minus_Enhanced": delta_mae,
                **bootstrap_ci,
                "N_Permutations": N_PERMUTATIONS,
                "Permutation_P_Delta_RMSE": increment_p,
            }
        )
    increment = pd.DataFrame(increment_rows)
    increment_distribution = pd.concat(increment_distributions, ignore_index=True)

    performance.to_csv(
        output_dir / "model_performance_with_permutation.csv", index=False, encoding="utf-8-sig"
    )
    increment.to_csv(output_dir / "incremental_brain_value.csv", index=False, encoding="utf-8-sig")
    increment_distribution.to_csv(
        output_dir / "permutation_incremental_brain_distribution.csv", index=False, encoding="utf-8-sig"
    )
    full_sample_weights(data, y).to_csv(
        output_dir / "full_sample_linear_svr_weights.csv", index=False, encoding="utf-8-sig"
    )
    write_summary(data, performance, increment, output_dir)

    print(f"完成分析，纳入长期完整随访个体：{len(data)} 人。")
    print(performance[["Model", "RMSE", "MAE", "CV_R2", "Prediction_r", "Permutation_P_RMSE"]].to_string(index=False))
    print("\n加入基线 DT / FO 的增量评价：")
    print(increment.to_string(index=False))
    print("\n汇总结果已保存。")


if __name__ == "__main__":
    main()
