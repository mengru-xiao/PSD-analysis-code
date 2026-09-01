"""AHBA 提取、PLS1、空间零模型与富集分析的共用计算函数。

正式空间检验由脚本 07 的皮层 surface-spin 分支执行。本文件不包含绘图；
保留 Moran/BrainSMASH 函数仅供敏感性分析复核。

预先指定的 State 7 与 AHBA 转录表达分析脚本。

研究目的
--------
本脚本仅针对 AHBA 分析前预先指定的 LEiDA State 7，
分析其左半球空间状态向量与 Allen Human Brain Atlas (AHBA) 基因表达
模式的关联，并输出适合后续 Metascape 功能富集和细胞类型富集的基因集。

本分析保留 Brainnetome 左半球全部 ROI，包括皮层及皮层下区域。因此，
不能直接使用仅适于皮层表面旋转的 spin test。脚本提供两套可分别运行、
随后比较的空间自相关零模型：

    --null-model moran
        Moran spectral randomization (MSR)。由 ROI 质心欧氏距离构建
        逆距离权重矩阵；矩阵对角线设为 0，表示不引入 ROI 自连接；
        使用 BrainSpace 先拟合 Moran eigenvector map 后生成 surrogate。

    --null-model brainsmash
        BrainSMASH / Burt2020 surrogate maps。直接基于 ROI 间距离矩阵
        模拟匹配空间自相关的表型图。

注意：两种方法用于方法探索与后续选择；最终论文应预先明确采用其中
一套空间校正方法，不应仅依据哪一套更显著而选择结论。

AHBA 表达矩阵构建
-----------------
1. 输入图谱为 Brainnetome 246 分区 NIfTI 文件。
2. 脚本根据 ``BN_Atlas_246_LUT.txt`` 自动生成 abagen 所需的
   ``atlas_info``：
       - ROI 1-210 标记为 ``cortex``；
       - ROI 211-246 标记为 ``subcortex/brainstem``；
       - 根据标签 ``_L`` / ``_R`` 标记半球。
3. 使用 ``abagen.get_expression_data(..., missing="centroids")`` 获得
   完整表达矩阵。对无直接组织样本覆盖的 ROI，abagen 在相同半球及
   结构限制下以质心最近样本补全。
4. 正式分析仅保留左半球全部 123 个 Brainnetome ROI。
5. 脚本保存 ``AHBA_ROI_Sample_Coverage.csv`` 与 abagen 方法报告，便于
   核查哪些 ROI 需要 centroid 补全。

PLS1 分析与空间检验
--------------------
对每一个状态分别使用单成分偏最小二乘回归（PLS1）：

    X = 左半球 ROI x 基因表达矩阵
    Y = 相同 ROI 顺序下的原始 LEiDA 状态向量

脚本输出：

    - PLS1 对状态向量 Y 的样本内解释比例 ``PLS1_Y_R2``；
    - 原始 PLS1 score 与原始状态向量的 Pearson / Spearman 相关；
    - 空间零模型下 R2 的经验 P 值（主要空间统计量）；
    - 空间零模型下 |Spearman r| 的经验 P 值（辅助统计量）。

每个零模型默认生成 5000 个 surrogate maps。P 值采用
``(极端次数 + 1) / (重复次数 + 1)``，因此不会输出不合法的 P=0。
State 7 的整体空间检验直接使用经验 P 值，不进行跨状态校正。

PLS1 方向解释
-------------
输入状态向量的正负方向始终保持原样。由于 PLS1 成分的整体正负号在
数学上可任意翻转，脚本同时保存：

    - 软件原始输出的 PLS1 score / gene weight；
    - 为解释而定向后的权重：使定向 PLS1 score 与原始状态向量正相关。

定向操作不改变模型拟合、R2 或空间 P 值，仅用于生成稳定可解释的文件：

    - ``Genes_Associated_With_Higher_State_Vector_Values``：
      更倾向于在原始状态向量值较高脑区表达的基因；
    - ``Genes_Associated_With_Lower_State_Vector_Values``：
      更倾向于在原始状态向量值较低脑区表达的基因。

基因筛选与 Metascape 输入
-------------------------
使用脑区 bootstrap 计算每个基因的 Bootstrap ratio / approximate Z，
并在每个状态内对基因层面 P 值进行 BH-FDR 校正。输出两类列表：

1. ``Gene_Selection_Thresholded``（参考论文风格）：
       - 主阈值：|Bootstrap Z| > 3 且 gene-level FDR < 0.05；
       - 探索阈值：|Bootstrap Z| > 2.58 且 gene-level FDR < 0.05。
2. ``Gene_Selection_Ranked_Percentage``（排序探索版本）：
       - 与状态向量高值/低值相关的 Top/Bottom 2.5%、5%、10% 基因。
       - 这些列表不是统计显著基因，仅为 ranked exploratory input。

细胞类型富集
------------
脚本首次运行时自动下载 Seidlitz et al. 2020 补充材料中已发表的
人类脑细胞类型基因集，并解析七类细胞：

    Astrocyte, Endothelial, Microglia, Excitatory neuron,
    Inhibitory neuron, Oligodendrocyte, OPC.

细胞类型分析采用与参考论文同类的基因集重采样富集框架：对主阈值
``|Z| > 3 且 FDR < 0.05`` 的高值/低值相关基因列表，计算其与每类
细胞特异性基因集的重叠富集，并通过默认 5000 次随机抽取等大小基因集
生成经验 P 值；所有 State x Direction x Cell type 检验统一进行 BH-FDR
校正。即使某状态未通过整体空间 PLS 检验，脚本仍输出下游结果，但标记
为探索性，不能作为已通过整体空间关联检验的结果解释。

运行方式
--------
本脚本不会自动同时运行两个耗时零模型。请分别运行：

    python 06_ahba_pls_shared.py --null-model moran
    python 06_ahba_pls_shared.py --null-model brainsmash

调试时可减少重复次数：

    python 06_ahba_pls_shared.py --null-model moran \
        --n-spatial 100 --n-bootstrap 100 --n-cell-perm 100

正式分析默认次数：

    空间零模型 5000 次；PLS 权重 bootstrap 5000 次；
    细胞类型富集随机重采样 5000 次。
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
EIGENVECTOR_FILE = Path()
LUT_FILE = Path()
ATLAS_FILE = Path()
ABAGEN_DATA_DIR = Path()
GENERATED_INPUT_DIR = Path()
OUTPUT_ROOT = Path()
ATLAS_INFO_FILE = Path()
SEIDLITZ_FILE = Path()
SEIDLITZ_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2F"
    "s41467-020-17051-5/MediaObjects/41467_2020_17051_MOESM8_ESM.xlsx"
)

STATES = ["State_7"]
THRESHOLDS = [3.0, 2.58]
RANKED_PERCENTAGES = [0.025, 0.05, 0.10]
DEFAULT_SEED = 20260527

CELL_TYPES = {
    "Astrocyte": ["astrocyte", "astro"],
    "Endothelial": ["endothelial", "endo"],
    "Microglia": ["microglia", "micro"],
    "Excitatory_Neuron": ["excitatory", "excit", "glutamatergic", "glut"],
    "Inhibitory_Neuron": ["inhibitory", "inhib", "gabaergic", "gaba"],
    "Oligodendrocyte": ["oligodendrocyte", "oligo"],
    "OPC": ["opc", "precursor", "progenitor"],
}

# Seidlitz 2020 补充表中的 Class 标签与预定义七类细胞的对应关系。
# 未细分兴奋/抑制性的 `Neuro` 以及不属于预定义七类的 `Per`
# 仅记录在质控文件中，不强行并入主要细胞类型分析。
SEIDLITZ_CLASS_MAP = {
    "Astro": "Astrocyte",
    "Endo": "Endothelial",
    "Micro": "Microglia",
    "Neuro-Ex": "Excitatory_Neuron",
    "Neuro-In": "Inhibitory_Neuron",
    "Oligo": "Oligodendrocyte",
    "OPC": "OPC",
}


def parse_args() -> argparse.Namespace:
    """读取运行参数；每次运行只执行一个空间零模型。"""
    parser = argparse.ArgumentParser(description="预先指定的 LEiDA State 7 与 AHBA 分析")
    parser.add_argument("--eigenvectors", type=Path, required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--abagen-data", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--null-model",
        choices=["moran", "brainsmash"],
        required=True,
        help="空间零模型：moran 或 brainsmash",
    )
    parser.add_argument("--n-spatial", type=int, default=5000, help="空间零模型随机次数")
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="PLS 基因权重 bootstrap 次数")
    parser.add_argument("--n-cell-perm", type=int, default=5000, help="细胞类型富集随机重采样次数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    return parser.parse_args()


def configure_paths(
    eigenvectors: Path,
    lut: Path,
    atlas: Path,
    abagen_data: Path,
    output_root: Path,
) -> None:
    """由命令行参数设置输入和输出位置，不在代码中保存本机路径。"""
    global EIGENVECTOR_FILE, LUT_FILE, ATLAS_FILE, ABAGEN_DATA_DIR
    global GENERATED_INPUT_DIR, OUTPUT_ROOT, ATLAS_INFO_FILE, SEIDLITZ_FILE
    EIGENVECTOR_FILE = eigenvectors.resolve()
    LUT_FILE = lut.resolve()
    ATLAS_FILE = atlas.resolve()
    ABAGEN_DATA_DIR = abagen_data.resolve()
    OUTPUT_ROOT = output_root.resolve()
    GENERATED_INPUT_DIR = OUTPUT_ROOT / "Inputs_Generated"
    ATLAS_INFO_FILE = GENERATED_INPUT_DIR / "BN_Atlas_246_atlas_info_abagen.csv"
    SEIDLITZ_FILE = GENERATED_INPUT_DIR / "Seidlitz2020_CellType_GeneSets.xlsx"


def check_dependencies(null_model: str) -> None:
    """在运行开始前检查零模型和 AHBA 提取所需依赖。"""
    requirements = ["abagen", "openpyxl"]
    requirements.append("brainspace" if null_model == "moran" else "brainsmash")
    missing = [package for package in requirements if importlib.util.find_spec(package) is None]
    if missing:
        raise ImportError(
            "缺少本次分析依赖包："
            + ", ".join(missing)
            + "。请根据 README.md 安装后重新运行。"
        )


def ensure_inputs() -> None:
    """检查项目内的固定输入文件。"""
    missing = [
        path for path in [EIGENVECTOR_FILE, LUT_FILE, ATLAS_FILE, ABAGEN_DATA_DIR] if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 项 AHBA 必要输入。")
    GENERATED_INPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_brainnetome_lut() -> pd.DataFrame:
    """读取 Brainnetome LUT 并生成包含 broad structure 信息的表格。"""
    rows = []
    with LUT_FILE.open("r", encoding="utf-8") as stream:
        for line in stream:
            values = line.strip().split()
            if not values or int(values[0]) == 0:
                continue
            roi_id = int(values[0])
            label = values[1]
            if label.endswith("_L"):
                hemisphere = "L"
            elif label.endswith("_R"):
                hemisphere = "R"
            else:
                raise ValueError(f"ROI 标签无法识别半球：{label}")
            structure = "cortex" if roi_id <= 210 else "subcortex/brainstem"
            rows.append(
                {"id": roi_id, "label": label, "hemisphere": hemisphere, "structure": structure}
            )
    atlas_info = pd.DataFrame(rows)
    if len(atlas_info) != 246:
        raise ValueError(f"预期读取 246 个 Brainnetome ROI，实际读取 {len(atlas_info)} 个。")
    if not (atlas_info.loc[atlas_info["id"] <= 210, "structure"] == "cortex").all():
        raise ValueError("Brainnetome 皮层分类失败。")
    atlas_info.to_csv(ATLAS_INFO_FILE, index=False, encoding="utf-8-sig")
    return atlas_info


def enable_abagen_pandas_compatibility() -> None:
    """兼容 abagen 0.1.3 与 pandas >= 2.0 的已移除 API。

    abagen 0.1.3 在 probe collapse 阶段调用
    ``DataFrame.set_axis(..., inplace=False)``；pandas 2.0 起移除了该
    参数；在 centroid 补全阶段还调用了已移除的 ``DataFrame.append``。
    此兼容层仅在当前运行进程中恢复这两种旧调用的等价行为，不修改
    环境文件，也不改变表达矩阵计算。
    """
    if (
        "inplace" not in inspect.signature(pd.DataFrame.set_axis).parameters
        and not getattr(pd.DataFrame.set_axis, "_abagen_compat_patched", False)
    ):
        pandas_set_axis = pd.DataFrame.set_axis

        def set_axis_compat(
            self: pd.DataFrame,
            labels: object,
            *,
            axis: int | str = 0,
            copy: bool | None = None,
            inplace: bool | None = None,
        ) -> pd.DataFrame | None:
            result = pandas_set_axis(self, labels, axis=axis, copy=copy)
            if inplace:
                self._update_inplace(result)
                return None
            return result

        set_axis_compat._abagen_compat_patched = True  # type: ignore[attr-defined]
        pd.DataFrame.set_axis = set_axis_compat  # type: ignore[method-assign]

    if not hasattr(pd.DataFrame, "append"):

        def append_compat(
            self: pd.DataFrame,
            other: object,
            ignore_index: bool = False,
            verify_integrity: bool = False,
            sort: bool = False,
        ) -> pd.DataFrame:
            objects = [self] + (list(other) if isinstance(other, (list, tuple)) else [other])
            return pd.concat(
                objects,
                ignore_index=ignore_index,
                verify_integrity=verify_integrity,
                sort=sort,
            )

        pd.DataFrame.append = append_compat  # type: ignore[attr-defined]


def extract_ahba_expression(atlas_info: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """通过 abagen 提取完整表达矩阵，并保存左半球 ROI 样本覆盖情况。"""
    enable_abagen_pandas_compatibility()
    import abagen

    result = abagen.get_expression_data(
        str(ATLAS_FILE),
        atlas_info=atlas_info[["id", "hemisphere", "structure"]],
        missing="centroids",
        data_dir=str(ABAGEN_DATA_DIR),
        return_counts=True,
        return_report=True,
        verbose=1,
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError("abagen 未按预期返回 expression、counts 和 report，请核查版本。")
    expression_all, counts, report = result
    expression_all.index = expression_all.index.astype(int)
    counts.index = counts.index.astype(int)
    left_info = atlas_info[atlas_info["hemisphere"] == "L"].copy().set_index("id")
    left_ids = left_info.index.tolist()
    missing_expression = sorted(set(left_ids).difference(expression_all.index))
    if missing_expression:
        raise ValueError(f"启用 centroid 补全后仍缺少左侧 ROI 表达：{missing_expression}")
    expression = expression_all.loc[left_ids].copy()
    coverage = left_info.reset_index()
    left_counts = counts.reindex(left_ids).fillna(0)
    coverage["Matched_Sample_Count_Total"] = left_counts.sum(axis=1).to_numpy()
    coverage["Centroid_Imputation_Required"] = coverage["Matched_Sample_Count_Total"] == 0
    coverage.to_csv(output_dir / "AHBA_ROI_Sample_Coverage.csv", index=False, encoding="utf-8-sig")
    (output_dir / "abagen_processing_report.txt").write_text(str(report), encoding="utf-8-sig")
    expression.to_csv(output_dir / "AHBA_Expression_Left_Hemisphere_Dense.csv", encoding="utf-8-sig")
    return expression, coverage


def align_states(expression: pd.DataFrame, atlas_info: pd.DataFrame) -> pd.DataFrame:
    """将原始状态向量按 AHBA 表达矩阵 ROI 顺序严格对齐。"""
    state_vectors = pd.read_csv(EIGENVECTOR_FILE, index_col=0)
    missing_states = sorted(set(STATES).difference(state_vectors.columns))
    if missing_states:
        raise ValueError(f"状态向量文件缺少列：{missing_states}")
    labels = atlas_info.set_index("id").loc[expression.index, "label"]
    missing_labels = sorted(set(labels).difference(state_vectors.index))
    if missing_labels:
        raise ValueError(f"状态向量文件缺少 ROI 标签：{missing_labels}")
    aligned = state_vectors.loc[labels, STATES].copy()
    aligned.index = expression.index
    aligned.index.name = "id"
    return aligned


def roi_centroids(roi_ids: list[int], output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """计算左侧 ROI 质心坐标及 ROI 间物理距离矩阵。"""
    image = nib.load(str(ATLAS_FILE))
    atlas_data = image.get_fdata()
    affine = image.affine
    coords = []
    for roi_id in roi_ids:
        voxel_points = np.argwhere(atlas_data == roi_id)
        if len(voxel_points) == 0:
            raise ValueError(f"图谱 NIfTI 中未找到 ROI ID：{roi_id}")
        voxel_centroid = voxel_points.mean(axis=0)
        coords.append((affine @ np.r_[voxel_centroid, 1])[:3])
    coords = np.asarray(coords, dtype=float)
    distances = squareform(pdist(coords))
    pd.DataFrame(coords, index=roi_ids, columns=["X_mm", "Y_mm", "Z_mm"]).to_csv(
        output_dir / "ROI_Centroid_Coordinates_Left.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(distances, index=roi_ids, columns=roi_ids).to_csv(
        output_dir / "ROI_Euclidean_Distance_Matrix_Left.csv", encoding="utf-8-sig"
    )
    return coords, distances


def inverse_distance_weights(distances: np.ndarray) -> np.ndarray:
    """生成用于 Moran MSR 的无自连接逆距离权重矩阵。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1.0 / distances
    weights[~np.isfinite(weights)] = 0.0
    np.fill_diagonal(weights, 0.0)
    return weights


def generate_spatial_surrogates(
    y: np.ndarray,
    distances: np.ndarray,
    args: argparse.Namespace,
    state_seed: int,
) -> np.ndarray:
    """依据选择的零模型生成保留空间自相关的状态向量 surrogate maps。"""
    if args.null_model == "moran":
        from brainspace.null_models import MoranRandomization

        weights = inverse_distance_weights(distances)
        model = MoranRandomization(n_rep=args.n_spatial, random_state=state_seed)
        model.fit(weights)
        surrogates = model.randomize(y)
    else:
        from brainsmash.mapgen.base import Base

        model = Base(x=y, D=distances, seed=state_seed)
        surrogates = model(n=args.n_spatial)
    surrogates = np.asarray(surrogates, dtype=float)
    if surrogates.shape != (args.n_spatial, len(y)):
        raise ValueError(f"空间 surrogate 形状异常：{surrogates.shape}")
    return surrogates


def fit_pls1(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """拟合 PLS1，并生成原始及面向状态向量解释的定向结果。"""
    model = PLSRegression(n_components=1, scale=True)
    model.fit(x, y.reshape(-1, 1))
    raw_scores = model.x_scores_[:, 0]
    raw_weights = model.x_weights_[:, 0]
    predicted = model.predict(x).reshape(-1)
    raw_pearson = pearsonr(raw_scores, y).statistic
    raw_spearman = spearmanr(raw_scores, y).statistic
    sign = 1.0 if raw_spearman >= 0 else -1.0
    oriented_scores = raw_scores * sign
    oriented_weights = raw_weights * sign
    return {
        "model": model,
        "PLS1_Y_R2": float(r2_score(y, predicted)),
        "Raw_Pearson_R": float(raw_pearson),
        "Raw_Spearman_R": float(raw_spearman),
        "PLS1_Sign_Flipped_For_Interpretation": bool(sign < 0),
        "Oriented_Pearson_R": float(pearsonr(oriented_scores, y).statistic),
        "Oriented_Spearman_R": float(spearmanr(oriented_scores, y).statistic),
        "Raw_Weights": raw_weights,
        "Oriented_Weights": oriented_weights,
        "Raw_Scores": raw_scores,
        "Oriented_Scores": oriented_scores,
        "Predicted_Y": predicted,
    }


def empirical_p(observed: float, null_values: np.ndarray, two_sided: bool = False) -> float:
    """通过加一连续校正计算经验 P 值。"""
    if two_sided:
        extreme = np.sum(np.abs(null_values) >= abs(observed))
    else:
        extreme = np.sum(null_values >= observed)
    return float((extreme + 1) / (len(null_values) + 1))


def spatial_null_statistics(
    x: np.ndarray,
    surrogates: np.ndarray,
    observed: dict[str, object],
) -> tuple[dict[str, float], pd.DataFrame]:
    """对 surrogate maps 重新拟合 PLS1，检验 R2 与相关强度。"""
    null_r2 = np.empty(len(surrogates), dtype=float)
    null_spearman = np.empty(len(surrogates), dtype=float)
    for index, y_null in enumerate(tqdm(surrogates, desc="空间零模型 PLS1")):
        null_result = fit_pls1(x, y_null)
        null_r2[index] = null_result["PLS1_Y_R2"]
        null_spearman[index] = null_result["Raw_Spearman_R"]
    summary = {
        "Spatial_P_R2_Raw": empirical_p(observed["PLS1_Y_R2"], null_r2),
        "Spatial_P_AbsSpearman_Raw": empirical_p(
            observed["Raw_Spearman_R"], null_spearman, two_sided=True
        ),
    }
    null_table = pd.DataFrame({"Null_PLS1_Y_R2": null_r2, "Null_Spearman_R": null_spearman})
    return summary, null_table


def bootstrap_gene_weights(
    x: np.ndarray,
    y: np.ndarray,
    oriented_weights: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """通过 ROI 重采样计算与定向真实权重对齐的 bootstrap 权重矩阵。"""
    rng = np.random.default_rng(seed)
    weights = np.full((n_bootstrap, x.shape[1]), np.nan, dtype=float)
    for index in tqdm(range(n_bootstrap), desc="PLS 基因权重 Bootstrap"):
        sample = rng.integers(0, x.shape[0], size=x.shape[0])
        try:
            boot = fit_pls1(x[sample], y[sample])
            current = np.asarray(boot["Raw_Weights"], dtype=float)
            if np.corrcoef(current, oriented_weights)[0, 1] < 0:
                current = -current
            weights[index] = current
        except Exception:
            continue
    valid = weights[~np.isnan(weights).all(axis=1)]
    if len(valid) < n_bootstrap * 0.95:
        raise RuntimeError(f"有效 bootstrap 次数仅 {len(valid)} / {n_bootstrap}，请核查数据。")
    return valid


def gene_weight_table(
    gene_names: pd.Index,
    real_result: dict[str, object],
    bootstrap_weights: np.ndarray,
) -> pd.DataFrame:
    """建立包含原始方向、定向方向、BSR 与基因级 FDR 的完整基因表。"""
    standard_error = bootstrap_weights.std(axis=0, ddof=1)
    oriented_weights = np.asarray(real_result["Oriented_Weights"])
    bootstrap_z = oriented_weights / np.where(standard_error == 0, np.nan, standard_error)
    gene_p = 2 * norm.sf(np.abs(bootstrap_z))
    gene_q = multipletests(gene_p, method="fdr_bh")[1]
    table = pd.DataFrame(
        {
            "Gene": gene_names,
            "Raw_PLS1_Weight": real_result["Raw_Weights"],
            "Oriented_Weight_Higher_State_Vector": oriented_weights,
            "Bootstrap_SE": standard_error,
            "Bootstrap_Z": bootstrap_z,
            "Gene_P_TwoSided": gene_p,
            "Gene_FDR_BH": gene_q,
        }
    )
    return table.sort_values("Bootstrap_Z", ascending=False).reset_index(drop=True)


def write_gene_list(path: Path, genes: pd.Series) -> None:
    """保存供 Metascape 或富集分析读取的纯基因名文本列表。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(genes.astype(str).tolist()), encoding="utf-8")


def output_gene_selections(gene_table: pd.DataFrame, state_dir: Path) -> dict[str, pd.DataFrame]:
    """输出阈值法及排序比例法基因列表，并返回主阈值方向列表。"""
    threshold_dir = state_dir / "Gene_Selection_Thresholded"
    ranked_dir = state_dir / "Gene_Selection_Ranked_Percentage"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    ranked_dir.mkdir(parents=True, exist_ok=True)
    threshold_summary = []
    primary_lists: dict[str, pd.DataFrame] = {}

    for threshold in THRESHOLDS:
        label = str(threshold).replace(".", "p")
        high = gene_table[
            (gene_table["Bootstrap_Z"] > threshold) & (gene_table["Gene_FDR_BH"] < 0.05)
        ]
        low = gene_table[
            (gene_table["Bootstrap_Z"] < -threshold) & (gene_table["Gene_FDR_BH"] < 0.05)
        ]
        write_gene_list(
            threshold_dir / f"Genes_Associated_With_Higher_State_Vector_Zgt{label}_FDR05.txt",
            high["Gene"],
        )
        write_gene_list(
            threshold_dir / f"Genes_Associated_With_Lower_State_Vector_ZltMinus{label}_FDR05.txt",
            low["Gene"],
        )
        threshold_summary.extend(
            [
                {"Threshold_Z": threshold, "Direction": "Higher_State_Vector", "Gene_Count": len(high)},
                {"Threshold_Z": threshold, "Direction": "Lower_State_Vector", "Gene_Count": len(low)},
            ]
        )
        if threshold == 3.0:
            primary_lists = {"Higher_State_Vector": high, "Lower_State_Vector": low}

    pd.DataFrame(threshold_summary).to_csv(
        threshold_dir / "Threshold_Selection_Summary.csv", index=False, encoding="utf-8-sig"
    )

    ranked_summary = []
    for fraction in RANKED_PERCENTAGES:
        count = max(1, math.ceil(len(gene_table) * fraction))
        label = str(fraction * 100).replace(".", "p")
        high = gene_table.head(count)
        low = gene_table.tail(count).sort_values("Bootstrap_Z")
        write_gene_list(
            ranked_dir / f"Genes_Associated_With_Higher_State_Vector_Top{label}Percent.txt",
            high["Gene"],
        )
        write_gene_list(
            ranked_dir / f"Genes_Associated_With_Lower_State_Vector_Bottom{label}Percent.txt",
            low["Gene"],
        )
        ranked_summary.extend(
            [
                {"Fraction": fraction, "Direction": "Higher_State_Vector", "Gene_Count": len(high)},
                {"Fraction": fraction, "Direction": "Lower_State_Vector", "Gene_Count": len(low)},
            ]
        )
    pd.DataFrame(ranked_summary).to_csv(
        ranked_dir / "Ranked_Selection_Summary.csv", index=False, encoding="utf-8-sig"
    )
    return primary_lists


def download_seidlitz_workbook() -> Path:
    """首次运行时下载参考论文同源的 Seidlitz 细胞类型基因集附件。"""
    if not SEIDLITZ_FILE.exists():
        print("正在下载 Seidlitz 2020 细胞类型基因集补充文件...")
        try:
            urllib.request.urlretrieve(SEIDLITZ_URL, SEIDLITZ_FILE)
        except Exception as exc:
            raise RuntimeError("无法取得 Seidlitz 补充文件，请改用本地参考工作簿。") from exc
    return SEIDLITZ_FILE


def match_cell_type(text: object) -> str | None:
    """根据标签文字匹配七类细胞；优先识别 OPC，避免被 oligo 抢先匹配。"""
    value = str(text).lower()
    if any(keyword in value for keyword in CELL_TYPES["OPC"]):
        return "OPC"
    for cell_type, keywords in CELL_TYPES.items():
        if cell_type == "OPC":
            continue
        if any(keyword in value for keyword in keywords):
            return cell_type
    return None


def is_gene_symbol(value: object) -> bool:
    """宽松判断单元格是否可能为基因符号。"""
    if pd.isna(value):
        return False
    text = str(value).strip()
    return text != "" and " " not in text and len(text) <= 40 and not text.isnumeric()


def load_seidlitz_gene_sets(background_genes: set[str], output_dir: Path) -> dict[str, set[str]]:
    """读取 Seidlitz 补充表，将簇级标记基因合并为预定义七类基因集。

    下载的工作簿为簇级宽表：`Class` 列给出该行所属细胞类，标记基因自
    `Genes` 列开始横向排列至其后的未命名列。这里按 Class 聚合所有行，
    再限制在本次 AHBA 表达矩阵实际存在的背景基因范围内。
    """
    workbook = pd.read_excel(download_seidlitz_workbook(), sheet_name=None)
    collected: dict[str, set[str]] = {name: set() for name in CELL_TYPES}

    source_sheet = None
    source_frame = None
    for sheet_name, dataframe in workbook.items():
        if {"Class", "Genes"}.issubset(dataframe.columns):
            source_sheet = sheet_name
            source_frame = dataframe.copy()
            break
    if source_frame is None:
        details = {sheet: dataframe.columns.astype(str).tolist() for sheet, dataframe in workbook.items()}
        (output_dir / "Seidlitz_Workbook_Sheets_For_Debugging.json").write_text(
            json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise ValueError("Seidlitz 工作簿未找到同时包含 Class 和 Genes 列的数据表，无法按已验证结构解析。")

    first_gene_column = source_frame.columns.get_loc("Genes")
    gene_columns = source_frame.columns[first_gene_column:]
    class_rows = []
    for source_class, rows in source_frame.groupby("Class", dropna=False):
        source_class = str(source_class).strip()
        mapped_cell_type = SEIDLITZ_CLASS_MAP.get(source_class)
        raw_genes = {
            str(value).strip()
            for value in rows.loc[:, gene_columns].to_numpy().ravel()
            if is_gene_symbol(value)
        }
        retained_genes = raw_genes & background_genes
        if mapped_cell_type is not None:
            collected[mapped_cell_type].update(retained_genes)
        class_rows.append(
            {
                "Source_Sheet": source_sheet,
                "Source_Class": source_class,
                "Mapped_Cell_Type": (
                    mapped_cell_type if mapped_cell_type is not None else "Excluded_not_in_predefined_7_types"
                ),
                "Cluster_Row_Count": len(rows),
                "Raw_Unique_Gene_Count": len(raw_genes),
                "Gene_Count_In_AHBA_Background": len(retained_genes),
            }
        )

    pd.DataFrame(class_rows).to_csv(
        output_dir / "Seidlitz_Source_Class_Mapping_And_Coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing = [cell_type for cell_type, genes in collected.items() if len(genes) == 0]
    if missing:
        raise ValueError(
            "Seidlitz 工作簿已按 Class 标签解析，但以下预定义细胞类型在 AHBA 背景基因中没有匹配基因："
            + ", ".join(missing)
            + "。请检查 Seidlitz_Source_Class_Mapping_And_Coverage.csv。"
        )
    rows = []
    for cell_type, genes in collected.items():
        write_gene_list(output_dir / "Seidlitz_Gene_Sets" / f"{cell_type}.txt", pd.Series(sorted(genes)))
        rows.append({"Cell_Type": cell_type, "Gene_Count_In_AHBA_Background": len(genes)})
    pd.DataFrame(rows).to_csv(
        output_dir / "Seidlitz_Cell_Type_Gene_Set_Summary.csv", index=False, encoding="utf-8-sig"
    )
    return collected


def cell_type_enrichment(
    state: str,
    direction: str,
    selected_genes: set[str],
    background_genes: np.ndarray,
    cell_gene_sets: dict[str, set[str]],
    n_perm: int,
    seed: int,
    overall_pass: bool,
) -> list[dict[str, object]]:
    """按参考论文字面策略检验状态相关基因集合的细胞类型富集。

    Higher/Lower state-vector genes 固定不动；对每一种 Seidlitz 细胞
    类型，从 AHBA 背景中重采样与该细胞类型基因集等大小的模拟参考
    集合，并计算其与固定状态相关基因列表的重叠数。
    """
    results = []
    background_set = set(background_genes)
    selected_genes = selected_genes & background_set
    rng = np.random.default_rng(seed)
    for cell_type, genes in cell_gene_sets.items():
        genes = genes & background_set
        observed_overlap = len(selected_genes & genes)
        expected_overlap = len(selected_genes) * len(genes) / len(background_genes) if selected_genes else np.nan
        fold_enrichment = observed_overlap / expected_overlap if expected_overlap and expected_overlap > 0 else np.nan
        if not selected_genes:
            p_value = np.nan
        else:
            null_overlap = np.empty(n_perm, dtype=int)
            for index in range(n_perm):
                sampled_reference = set(
                    rng.choice(background_genes, size=len(genes), replace=False)
                )
                null_overlap[index] = len(selected_genes & sampled_reference)
            p_value = float((np.sum(null_overlap >= observed_overlap) + 1) / (n_perm + 1))
        results.append(
            {
                "State": state,
                "Direction": direction,
                "Cell_Type": cell_type,
                "Selected_Gene_Count": len(selected_genes),
                "Cell_Type_Gene_Count": len(genes),
                "Observed_Overlap": observed_overlap,
                "Expected_Overlap": expected_overlap,
                "Fold_Enrichment": fold_enrichment,
                "P_Random_GeneSet_Raw": p_value,
                "Null_Resampling_Strategy": (
                    "Fixed_state_gene_list_resample_cell_type_reference_genes"
                ),
                "Overall_State_Spatial_Passed": overall_pass,
                "Interpretation_Level": (
                    "Downstream_of_spatially_supported_state"
                    if overall_pass
                    else "Exploratory_only_overall_spatial_test_not_passed"
                ),
            }
        )
    return results


def main() -> None:
    """运行一套指定空间零模型的完整 AHBA-PLS 重分析。"""
    args = parse_args()
    if min(args.n_spatial, args.n_bootstrap, args.n_cell_perm) < 1:
        raise ValueError("随机重复次数必须均为正整数。")
    configure_paths(
        args.eigenvectors,
        args.lut,
        args.atlas,
        args.abagen_data,
        args.output_directory,
    )
    check_dependencies(args.null_model)
    ensure_inputs()

    run_dir = OUTPUT_ROOT / args.null_model
    common_dir = run_dir / "Common_Inputs_And_QC"
    common_dir.mkdir(parents=True, exist_ok=True)
    print(f"空间零模型：{args.null_model}。")

    atlas_info = parse_brainnetome_lut()
    atlas_info.to_csv(common_dir / ATLAS_INFO_FILE.name, index=False, encoding="utf-8-sig")
    expression, coverage = extract_ahba_expression(atlas_info, common_dir)
    state_vectors = align_states(expression, atlas_info)
    state_vectors.to_csv(common_dir / "Aligned_State_Vectors_Left_Hemisphere.csv", encoding="utf-8-sig")
    _, distances = roi_centroids(expression.index.tolist(), common_dir)
    x = expression.to_numpy(dtype=float)
    background_genes = expression.columns.astype(str).to_numpy()

    model_rows: list[dict[str, object]] = []
    state_gene_primary: dict[str, dict[str, pd.DataFrame]] = {}
    for state_index, state in enumerate(STATES):
        print(f"\n开始分析 {state}...")
        state_dir = run_dir / state
        state_dir.mkdir(parents=True, exist_ok=True)
        y = state_vectors[state].to_numpy(dtype=float)
        real = fit_pls1(x, y)
        print(
            f"  正在生成 {args.n_spatial} 个 {args.null_model} 空间替代图；"
            "BrainSMASH 在此阶段拟合空间变异函数时可能等待较久..."
        )
        surrogate = generate_spatial_surrogates(
            y, distances, args, state_seed=args.seed + state_index
        )
        print("  空间替代图生成完成，开始逐次重新拟合 PLS1 以建立零分布...")
        spatial_stats, null_table = spatial_null_statistics(x, surrogate, real)
        null_table.to_csv(state_dir / "Spatial_Null_Distribution.csv", index=False, encoding="utf-8-sig")
        scores = pd.DataFrame(
            {
                "id": expression.index,
                "Original_State_Vector": y,
                "Raw_PLS1_Score": real["Raw_Scores"],
                "Oriented_PLS1_Score_Higher_State_Vector": real["Oriented_Scores"],
                "PLS1_Predicted_State_Vector": real["Predicted_Y"],
            }
        )
        scores.to_csv(state_dir / "ROI_PLS1_Scores_And_State_Vector.csv", index=False, encoding="utf-8-sig")
        print(f"  正在执行 {args.n_bootstrap} 次 PLS 基因权重 bootstrap...")
        boot_weights = bootstrap_gene_weights(
            x,
            y,
            np.asarray(real["Oriented_Weights"]),
            args.n_bootstrap,
            args.seed + 100 + state_index,
        )
        genes = gene_weight_table(expression.columns, real, boot_weights)
        genes.to_csv(state_dir / "Gene_Weights_Full_Oriented_To_State_Vector.csv", index=False, encoding="utf-8-sig")
        state_gene_primary[state] = output_gene_selections(genes, state_dir)
        model_rows.append(
            {
                "State": state,
                "Null_Model": args.null_model,
                "ROI_Count": len(y),
                "Gene_Count": x.shape[1],
                "PLS1_Y_R2": real["PLS1_Y_R2"],
                "Raw_Pearson_R": real["Raw_Pearson_R"],
                "Raw_Spearman_R": real["Raw_Spearman_R"],
                "PLS1_Sign_Flipped_For_Interpretation": real["PLS1_Sign_Flipped_For_Interpretation"],
                "Oriented_Pearson_R": real["Oriented_Pearson_R"],
                "Oriented_Spearman_R": real["Oriented_Spearman_R"],
                **spatial_stats,
                "N_Spatial_Randomizations": args.n_spatial,
                "N_Bootstrap": args.n_bootstrap,
            }
        )

    model_summary = pd.DataFrame(model_rows)
    model_summary["Overall_State_Spatial_Passed"] = model_summary["Spatial_P_R2_Raw"] < 0.05
    model_summary.to_csv(run_dir / "Model_Summary_State7.csv", index=False, encoding="utf-8-sig")
    for state in STATES:
        passed = bool(
            model_summary.loc[
                model_summary["State"] == state, "Overall_State_Spatial_Passed"
            ].iloc[0]
        )
        status = pd.DataFrame(
            [
                {
                    "State": state,
                    "Overall_State_Spatial_Passed": passed,
                    "Downstream_Gene_And_Cell_Results_Interpretation": (
                        "Downstream results may be interpreted after an overall spatially supported PLS1 association."
                        if passed
                        else "Exploratory only: the prespecified State 7 spatial test did not pass."
                    ),
                }
            ]
        )
        status.to_csv(
            run_dir / state / "Downstream_Interpretation_Status.csv",
            index=False,
            encoding="utf-8-sig",
        )

    cell_dir = run_dir / "Cell_Type_Enrichment_Seidlitz"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell_gene_sets = load_seidlitz_gene_sets(set(background_genes), cell_dir)
    enrichment_rows: list[dict[str, object]] = []
    for state_index, state in enumerate(STATES):
        passed = bool(
            model_summary.loc[
                model_summary["State"] == state, "Overall_State_Spatial_Passed"
            ].iloc[0]
        )
        for direction_index, (direction, genes) in enumerate(state_gene_primary[state].items()):
            enrichment_rows.extend(
                cell_type_enrichment(
                    state=state,
                    direction=direction,
                    selected_genes=set(genes["Gene"]),
                    background_genes=background_genes,
                    cell_gene_sets=cell_gene_sets,
                    n_perm=args.n_cell_perm,
                    seed=args.seed + 1000 + state_index * 10 + direction_index,
                    overall_pass=passed,
                )
            )
    enrichment = pd.DataFrame(enrichment_rows)
    valid = enrichment["P_Random_GeneSet_Raw"].notna()
    enrichment["P_FDR_BH_Across_All_Cell_Tests"] = np.nan
    if valid.any():
        enrichment.loc[valid, "P_FDR_BH_Across_All_Cell_Tests"] = multipletests(
            enrichment.loc[valid, "P_Random_GeneSet_Raw"], method="fdr_bh"
        )[1]
    enrichment.sort_values(
        ["P_FDR_BH_Across_All_Cell_Tests", "P_Random_GeneSet_Raw"], na_position="last"
    ).to_csv(cell_dir / "Cell_Type_Enrichment_State7_FDR.csv", index=False, encoding="utf-8-sig")

    run_config = {
        "null_model": args.null_model,
        "n_spatial": args.n_spatial,
        "n_bootstrap": args.n_bootstrap,
        "n_cell_perm": args.n_cell_perm,
        "seed": args.seed,
        "states": STATES,
        "missing_roi_strategy": "abagen missing='centroids'",
        "cell_type_reference": "Seidlitz2020_MOESM8",
        "primary_gene_threshold": "|Bootstrap_Z| > 3 and Gene_FDR_BH < 0.05",
    }
    (run_dir / "Run_Configuration.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n分析完成。请首先查看 Model_Summary_State7.csv，再解释下游富集结果。")


if __name__ == "__main__":
    main()
