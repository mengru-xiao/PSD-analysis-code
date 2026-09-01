"""预先指定的 State 7 与 AHBA 的 Brainnetome 皮层 surface-spin 分析。

研究目的
--------
本脚本建立一个与参考论文空间检验原理一致的皮层分析分支：仅使用
Brainnetome 左半球 105 个皮层 ROI，将其原始 LEiDA 状态向量与 AHBA
转录表达进行 PLS1 分析，并使用表面球面旋转（surface-based spin
permutation）检验 PLS1 对状态向量空间变异的解释比例。

与包含皮层下 ROI 的 Moran/BrainSMASH 主脚本不同，本脚本主动排除
Brainnetome 皮层下分区，理由是球面旋转只对皮层表面分区成立。本分支
不更换 Brainnetome 图谱：它复现参考论文的空间置换原则，而不是复现
参考论文使用的具体分区模板。

表面 spin 空间置换方法
----------------------
1. 读取项目内 Brainnetome FreeSurfer 表面文件：
       - ``100307/label/lh.BN_Atlas.annot``
       - ``100307/surf/lh.sphere.reg``
2. 对左半球 105 个皮层 ROI，在注册球面上计算 parcel centroid。
3. 每次生成随机三维旋转矩阵，将所有 parcel centroid 在球面上共同旋转。
4. 通过一对一 Hungarian 最近匹配，将旋转后的 parcels 重新指派给原始
   parcels，得到保留皮层空间拓扑的替代状态向量。
5. 对每个替代向量重新拟合 PLS1，以 ``PLS1_Y_R2`` 为主要统计量形成
   spin 空间零分布；同时输出 |Spearman r| 的辅助检验。
6. P 值使用 ``(extreme + 1) / (n_spin + 1)``。State 7 在 AHBA 分析前
   已预先指定，因此不进行跨状态多重比较校正。

后续输出
--------
脚本延续正式重分析的完整结果体系：
    - PLS1 解释比例、相关系数及 spin P 值；
    - ROI-PLS1 score 和每次 spin 的零分布；
    - 5000 次 bootstrap 基因权重；
    - ``|Bootstrap Z| > 3`` / ``2.58`` 且基因级 FDR < 0.05 的列表；
    - Top/Bottom 2.5%、5%、10% 排序列表，供探索性 Metascape 对比；
    - Seidlitz 2020 七类细胞类型基因集的随机重采样富集结果。

运行说明
--------
正式运行（默认每个阶段 5000 次）：

    python 07_ahba_surface_spin.py

仅用于检查依赖和数据映射的调试运行：

    python 07_ahba_surface_spin.py \
        --n-spatial 100 --n-bootstrap 100 --n-cell-perm 100

正式结果写入：

    Results/spin_surface_cortical_brainnetome/

注意：空间检验及 bootstrap 会对高维 AHBA 基因矩阵重复拟合 PLS，
正式 5000 次分析耗时较长属于预期现象。脚本会打印每一耗时阶段的进度。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_PATH = SCRIPT_DIR / "06_ahba_pls_shared.py"


def load_shared_module():
    """加载同目录中的 AHBA 共用计算模块。"""

    spec = importlib.util.spec_from_file_location("ahba_pls_shared", SHARED_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("无法加载 AHBA 共用计算模块。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load_shared_module()


DEFAULT_SEED = shared.DEFAULT_SEED


def parse_args() -> argparse.Namespace:
    """读取运行参数；默认重复次数用于正式分析。"""
    parser = argparse.ArgumentParser(description="Brainnetome 左半球皮层 AHBA surface spin 重分析")
    parser.add_argument("--eigenvectors", type=Path, required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--abagen-data", type=Path, required=True)
    parser.add_argument("--surface-annot", type=Path, required=True)
    parser.add_argument("--surface-sphere", type=Path, required=True)
    parser.add_argument("--seidlitz-file", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--n-spatial", type=int, default=5000, help="表面 spin 旋转置换次数")
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="PLS 基因权重 bootstrap 次数")
    parser.add_argument("--n-cell-perm", type=int, default=5000, help="细胞类型富集随机重采样次数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    return parser.parse_args()


def check_inputs(surface_annot: Path, surface_sphere: Path) -> None:
    """检查体积输入、AHBA 数据和表面 spin 所需的 FreeSurfer 文件。"""
    required = ["abagen", "openpyxl", "nibabel", "scipy"]
    missing_packages = [package for package in required if importlib.util.find_spec(package) is None]
    if missing_packages:
        raise ImportError("缺少 surface spin 分析依赖包：" + ", ".join(missing_packages))
    shared.ensure_inputs()
    missing_surface = [path for path in [surface_annot, surface_sphere] if not path.exists()]
    if missing_surface:
        raise FileNotFoundError(f"缺少 {len(missing_surface)} 项 surface-spin 必要输入。")


def load_cortical_expression(
    atlas_info: pd.DataFrame, common_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载或提取 AHBA 表达矩阵，并严格限定至左半球皮层 105 个 ROI。"""
    print("开始使用 abagen 提取完整左半球 AHBA 表达矩阵...")
    expression_all, coverage_all = shared.extract_ahba_expression(atlas_info, common_dir)

    cortex_info = atlas_info[
        (atlas_info["hemisphere"] == "L") & (atlas_info["structure"] == "cortex")
    ].copy()
    cortical_ids = cortex_info["id"].astype(int).tolist()
    if len(cortical_ids) != 105:
        raise ValueError(f"预期 Brainnetome 左半球皮层包含 105 个 ROI，实际为 {len(cortical_ids)}。")
    missing_ids = sorted(set(cortical_ids).difference(expression_all.index))
    if missing_ids:
        raise ValueError(f"AHBA 表达矩阵缺少左半球皮层 ROI：{missing_ids}")

    expression = expression_all.loc[cortical_ids].copy()
    expression.to_csv(
        common_dir / "AHBA_Expression_Left_Cortical_SurfaceSpin.csv", encoding="utf-8-sig"
    )
    coverage = coverage_all[coverage_all["id"].isin(cortical_ids)].copy()
    coverage.to_csv(
        common_dir / "AHBA_ROI_Sample_Coverage_Left_Cortical_SurfaceSpin.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return expression, cortex_info


def surface_parcel_centroids(
    cortical_info: pd.DataFrame,
    common_dir: Path,
    surface_annot: Path,
    surface_sphere: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 Brainnetome 左侧皮层 parcels 在 FreeSurfer 注册球面上的质心。"""
    sphere_coords, _ = nib.freesurfer.read_geometry(str(surface_sphere))
    annot_labels, _, annot_names = nib.freesurfer.read_annot(str(surface_annot))
    annot_names = [name.decode("utf-8") for name in annot_names]
    sphere_coords = sphere_coords / np.linalg.norm(sphere_coords, axis=1, keepdims=True)

    cortical_ids = cortical_info["id"].astype(int).to_numpy()
    expected_labels = cortical_info.set_index("id")["label"]
    rows = []
    centroids = []
    for roi_id in cortical_ids:
        if roi_id >= len(annot_names) or annot_names[roi_id] != expected_labels.loc[roi_id]:
            raise ValueError(
                f"表面 annotation 与体积 LUT 对应失败：ROI {roi_id}, "
                f"LUT={expected_labels.loc[roi_id]}, "
                f"surface={annot_names[roi_id] if roi_id < len(annot_names) else 'missing'}"
            )
        parcel_vertices = sphere_coords[annot_labels == roi_id]
        if len(parcel_vertices) == 0:
            raise ValueError(f"表面 annotation 未包含左侧皮层 ROI {roi_id}。")
        centroid = parcel_vertices.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids.append(centroid)
        rows.append(
            {
                "id": roi_id,
                "label": expected_labels.loc[roi_id],
                "Surface_Vertex_Count": len(parcel_vertices),
                "Sphere_Centroid_X": centroid[0],
                "Sphere_Centroid_Y": centroid[1],
                "Sphere_Centroid_Z": centroid[2],
            }
        )

    expected_id_set = set(cortical_ids)
    unexpected = []
    unique_labels, counts = np.unique(annot_labels, return_counts=True)
    for label_index, count in zip(unique_labels, counts):
        if int(label_index) not in expected_id_set:
            unexpected.append(
                {
                    "Annotation_Label_Index": int(label_index),
                    "Annotation_Name": (
                        annot_names[int(label_index)]
                        if 0 <= int(label_index) < len(annot_names)
                        else "unlabeled"
                    ),
                    "Surface_Vertex_Count": int(count),
                    "Included_In_Analysis": False,
                }
            )
    pd.DataFrame(rows).to_csv(
        common_dir / "Surface_Sphere_Parcel_Centroids_Left_Cortical.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(unexpected).to_csv(
        common_dir / "Surface_Annotation_Excluded_Labels_QC.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return np.asarray(centroids, dtype=float), cortical_ids


def generate_spin_mappings(
    centroids: np.ndarray, roi_ids: np.ndarray, n_spatial: int, seed: int, common_dir: Path
) -> np.ndarray:
    """生成球面旋转后的一对一 parcel 重指派索引，并保存可复现映射。"""
    rng = np.random.default_rng(seed)
    source_index_for_target = np.empty((n_spatial, len(roi_ids)), dtype=int)
    print(f"正在基于 lh.sphere.reg 生成 {n_spatial} 次表面球面旋转映射...")
    for iteration in tqdm(range(n_spatial), desc="表面 spin 旋转"):
        rotation = Rotation.random(random_state=rng).as_matrix()
        rotated_centroids = centroids @ rotation.T
        source_indices, target_indices = linear_sum_assignment(cdist(rotated_centroids, centroids))
        source_index_for_target[iteration, target_indices] = source_indices
    if not np.all(np.sort(source_index_for_target, axis=1) == np.arange(len(roi_ids))):
        raise RuntimeError("spin 映射不是一对一 parcel 重排，请停止分析并检查表面输入。")
    mapped_ids = roi_ids[source_index_for_target]
    pd.DataFrame(mapped_ids, columns=[f"Target_ROI_{roi_id}" for roi_id in roi_ids]).to_csv(
        common_dir / "Surface_Spin_Source_ROI_For_Each_Target.csv",
        index_label="Spin_Iteration",
        encoding="utf-8-sig",
    )
    return source_index_for_target


def write_interpretation_status(model_summary: pd.DataFrame, run_dir: Path) -> None:
    """为每个状态写出整体空间结果通过情况和下游解释边界。"""
    for state in shared.STATES:
        passed = bool(
            model_summary.loc[
                model_summary["State"] == state, "Overall_State_Spatial_Passed"
            ].iloc[0]
        )
        pd.DataFrame(
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
        ).to_csv(
            run_dir / state / "Downstream_Interpretation_Status.csv",
            index=False,
            encoding="utf-8-sig",
        )


def run_cell_type_enrichment(
    model_summary: pd.DataFrame,
    state_gene_primary: dict[str, dict[str, pd.DataFrame]],
    background_genes: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    """按参考论文字面重采样策略运行 Seidlitz 七类细胞类型富集分析。"""
    cell_dir = run_dir / "Cell_Type_Enrichment_Seidlitz"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cell_gene_sets = shared.load_seidlitz_gene_sets(set(background_genes), cell_dir)
    enrichment_rows: list[dict[str, object]] = []
    print(
        "正在进行 Seidlitz 细胞类型富集分析：固定状态相关基因，"
        f"对细胞类型参考基因集执行每项 {args.n_cell_perm} 次随机重采样..."
    )
    for state_index, state in enumerate(shared.STATES):
        passed = bool(
            model_summary.loc[
                model_summary["State"] == state, "Overall_State_Spatial_Passed"
            ].iloc[0]
        )
        for direction_index, (direction, genes) in enumerate(state_gene_primary[state].items()):
            enrichment_rows.extend(
                shared.cell_type_enrichment(
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


def main() -> None:
    """运行 Brainnetome 左侧皮层表面 spin 的完整 AHBA-PLS 重分析。"""
    args = parse_args()
    if min(args.n_spatial, args.n_bootstrap, args.n_cell_perm) < 1:
        raise ValueError("随机重复次数必须均为正整数。")
    run_dir = args.output_directory.resolve()
    surface_annot = args.surface_annot.resolve()
    surface_sphere = args.surface_sphere.resolve()
    shared.configure_paths(
        args.eigenvectors,
        args.lut,
        args.atlas,
        args.abagen_data,
        run_dir.parent,
    )
    shared.SEIDLITZ_FILE = args.seidlitz_file.resolve()
    check_inputs(surface_annot, surface_sphere)
    common_dir = run_dir / "Common_Inputs_And_QC"
    common_dir.mkdir(parents=True, exist_ok=True)
    print("空间零模型：surface spin；正式状态：State 7。")
    print("本分支只分析 Brainnetome 左半球皮层 ROI，皮层下 ROI 不参与 PLS 或富集。")

    atlas_info = shared.parse_brainnetome_lut()
    atlas_info.to_csv(common_dir / shared.ATLAS_INFO_FILE.name, index=False, encoding="utf-8-sig")
    expression, cortical_info = load_cortical_expression(atlas_info, common_dir)
    state_vectors = shared.align_states(expression, atlas_info)
    state_vectors.to_csv(
        common_dir / "Aligned_State_Vectors_Left_Cortical_SurfaceSpin.csv", encoding="utf-8-sig"
    )
    centroids, roi_ids = surface_parcel_centroids(
        cortical_info, common_dir, surface_annot, surface_sphere
    )
    spin_mappings = generate_spin_mappings(centroids, roi_ids, args.n_spatial, args.seed, common_dir)

    x = expression.to_numpy(dtype=float)
    background_genes = expression.columns.astype(str).to_numpy()
    model_rows: list[dict[str, object]] = []
    state_gene_primary: dict[str, dict[str, pd.DataFrame]] = {}
    for state_index, state in enumerate(shared.STATES):
        print(f"\n开始分析 {state}（左半球皮层 surface spin）...")
        state_dir = run_dir / state
        state_dir.mkdir(parents=True, exist_ok=True)
        y = state_vectors[state].to_numpy(dtype=float)
        observed = shared.fit_pls1(x, y)
        surrogates = y[spin_mappings]
        print(f"  已基于共同球面旋转映射生成 {args.n_spatial} 个替代状态图，开始逐次 PLS1 空间检验...")
        spatial_stats, null_table = shared.spatial_null_statistics(x, surrogates, observed)
        null_table.to_csv(state_dir / "Spatial_Null_Distribution.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(
            {
                "id": expression.index,
                "Original_State_Vector": y,
                "Raw_PLS1_Score": observed["Raw_Scores"],
                "Oriented_PLS1_Score_Higher_State_Vector": observed["Oriented_Scores"],
                "PLS1_Predicted_State_Vector": observed["Predicted_Y"],
            }
        ).to_csv(state_dir / "ROI_PLS1_Scores_And_State_Vector.csv", index=False, encoding="utf-8-sig")
        print(f"  正在执行 {args.n_bootstrap} 次 PLS 基因权重 bootstrap...")
        boot_weights = shared.bootstrap_gene_weights(
            x,
            y,
            np.asarray(observed["Oriented_Weights"]),
            args.n_bootstrap,
            args.seed + 100 + state_index,
        )
        genes = shared.gene_weight_table(expression.columns, observed, boot_weights)
        genes.to_csv(
            state_dir / "Gene_Weights_Full_Oriented_To_State_Vector.csv",
            index=False,
            encoding="utf-8-sig",
        )
        state_gene_primary[state] = shared.output_gene_selections(genes, state_dir)
        model_rows.append(
            {
                "State": state,
                "Null_Model": "surface_spin_brainnetome_lh_cortex",
                "ROI_Count": len(y),
                "Gene_Count": x.shape[1],
                "PLS1_Y_R2": observed["PLS1_Y_R2"],
                "Raw_Pearson_R": observed["Raw_Pearson_R"],
                "Raw_Spearman_R": observed["Raw_Spearman_R"],
                "PLS1_Sign_Flipped_For_Interpretation": observed[
                    "PLS1_Sign_Flipped_For_Interpretation"
                ],
                "Oriented_Pearson_R": observed["Oriented_Pearson_R"],
                "Oriented_Spearman_R": observed["Oriented_Spearman_R"],
                **spatial_stats,
                "N_Spatial_Randomizations": args.n_spatial,
                "N_Bootstrap": args.n_bootstrap,
            }
        )

    model_summary = pd.DataFrame(model_rows)
    model_summary["Overall_State_Spatial_Passed"] = model_summary["Spatial_P_R2_Raw"] < 0.05
    model_summary.to_csv(run_dir / "Model_Summary_State7.csv", index=False, encoding="utf-8-sig")
    write_interpretation_status(model_summary, run_dir)
    run_cell_type_enrichment(model_summary, state_gene_primary, background_genes, args, run_dir)

    configuration = {
        "null_model": "surface_spin_brainnetome_lh_cortex",
        "surface_spin_method": (
            "random 3D rotations of lh.sphere.reg parcel centroids with one-to-one "
            "Hungarian reassignment to original Brainnetome parcels"
        ),
        "surface_annot_name": surface_annot.name,
        "surface_sphere_name": surface_sphere.name,
        "roi_scope": "Brainnetome left cortical ROIs only; subcortical ROIs excluded",
        "n_spatial": args.n_spatial,
        "n_bootstrap": args.n_bootstrap,
        "n_cell_perm": args.n_cell_perm,
        "seed": args.seed,
        "states": shared.STATES,
        "missing_roi_strategy": "abagen missing='centroids'",
        "cell_type_reference": "Seidlitz2020_MOESM8",
        "cell_type_null_resampling_strategy": (
            "Fixed state-related gene list; resample equal-sized cell-type "
            "reference gene sets from AHBA background, as in the reference article"
        ),
        "primary_gene_threshold": "|Bootstrap_Z| > 3 and Gene_FDR_BH < 0.05",
    }
    (run_dir / "Run_Configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nSurface spin 分析完成。请首先查看 Model_Summary_State7.csv。")


if __name__ == "__main__":
    main()
