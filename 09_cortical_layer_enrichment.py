"""预先指定 State 7 的 L1–L6 皮层层级富集，不包含绘图。"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


REFERENCE_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fnn.4548/"
    "MediaObjects/41593_2017_BFnn4548_MOESM255_ESM.xlsx"
)
STATES = ("State_7",)
LAYERS = ("L1", "L2", "L3", "L4", "L5", "L6")
DIRECTIONS = ("Higher_State_Vector", "Lower_State_Vector")
GENE_LISTS = {
    "Higher_State_Vector": "Genes_Associated_With_Higher_State_Vector_Zgt3p0_FDR05.txt",
    "Lower_State_Vector": "Genes_Associated_With_Lower_State_Vector_ZltMinus3p0_FDR05.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 State 7 皮层层级基因富集。")
    parser.add_argument("--spin-results", type=Path, required=True)
    parser.add_argument(
        "--reference-file",
        type=Path,
        required=True,
    )
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def ensure_reference_file(path: Path) -> Path:
    """使用本地参考工作簿；缺失时下载公开附件。"""

    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print("正在下载皮层层级标记基因参考工作簿。")
    try:
        urllib.request.urlretrieve(REFERENCE_URL, path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download layer markers. Place the workbook manually at {path}."
        ) from exc
    return path


def background_genes(expression_file: Path) -> np.ndarray:
    if not expression_file.exists():
        raise FileNotFoundError("找不到 AHBA 表达矩阵。")
    columns = pd.read_csv(expression_file, nrows=0).columns.astype(str).tolist()
    if columns and (columns[0].startswith("Unnamed:") or columns[0] in {"label", "Label"}):
        columns = columns[1:]
    genes = np.asarray(columns, dtype=str)
    if len(genes) == 0 or len(set(genes)) != len(genes):
        raise ValueError("AHBA background genes are empty or duplicated.")
    return genes


def layer_gene_sets(reference_file: Path, background: set[str]) -> tuple[dict[str, set[str]], pd.DataFrame]:
    data = pd.read_excel(reference_file)
    required = {"Gene symbol", "Layer marker in human"}
    if not required.issubset(data.columns):
        raise ValueError(f"Layer workbook must contain columns {sorted(required)}.")

    gene_sets: dict[str, set[str]] = {}
    rows = []
    for layer in LAYERS:
        published = set(
            data.loc[data["Layer marker in human"].eq(layer), "Gene symbol"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        retained = published & background
        if not retained:
            raise ValueError(f"No {layer} marker genes remain in the AHBA background.")
        gene_sets[layer] = retained
        rows.append(
            {
                "Cortical_Layer": layer,
                "Published_Marker_Gene_Count": len(published),
                "Marker_Gene_Count_In_AHBA_Background": len(retained),
            }
        )
    return gene_sets, pd.DataFrame(rows)


def selected_genes(spin_dir: Path, state: str, direction: str) -> set[str]:
    path = spin_dir / state / "Gene_Selection_Thresholded" / GENE_LISTS[direction]
    if not path.exists():
        raise FileNotFoundError("找不到 State 7 基因列表。")
    genes = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not genes:
        raise ValueError("State 7 基因列表为空。")
    return genes


def one_test(
    state: str,
    direction: str,
    selected: set[str],
    layer: str,
    markers: set[str],
    background: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """计算一次重叠富集效应及随机参考集经验 P 值。"""

    background_set = set(background)
    selected = selected & background_set
    markers = markers & background_set
    observed = len(selected & markers)
    expected = len(selected) * len(markers) / len(background)
    null_overlap = np.empty(n_permutations, dtype=int)
    for index in range(n_permutations):
        random_markers = set(rng.choice(background, size=len(markers), replace=False))
        null_overlap[index] = len(selected & random_markers)
    p_raw = float((np.sum(null_overlap >= observed) + 1) / (n_permutations + 1))
    return {
        "State": state,
        "Direction": direction,
        "Cortical_Layer": layer,
        "Selected_Gene_Count": len(selected),
        "Layer_Marker_Gene_Count": len(markers),
        "Observed_Overlap": observed,
        "Expected_Overlap": expected,
        "Fold_Enrichment": observed / expected if expected > 0 else np.nan,
        "P_Random_GeneSet_Raw": p_raw,
        "Null_Resampling_Strategy": "Fixed_state_gene_list_resample_layer_reference_genes",
    }


def main() -> None:
    args = parse_args()
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be positive.")
    spin_dir = args.spin_results.resolve()
    output_dir = spin_dir / "Cell_Type_Enrichment_Seidlitz" / "Cortical_Layer_Enrichment"
    output_dir.mkdir(parents=True, exist_ok=True)

    expression_file = (
        spin_dir / "Common_Inputs_And_QC" / "AHBA_Expression_Left_Cortical_SurfaceSpin.csv"
    )
    model_file = spin_dir / "Model_Summary_State7.csv"
    if not model_file.exists():
        raise FileNotFoundError("找不到 State 7 空间模型汇总表。")
    model_summary = pd.read_csv(model_file).set_index("State")
    for state in STATES:
        if state not in model_summary.index or not bool(
            model_summary.loc[state, "Overall_State_Spatial_Passed"]
        ):
            raise ValueError("State 7 的预设整体空间检验未通过，不能进行正式层级富集解释。")

    background = background_genes(expression_file)
    markers, marker_summary = layer_gene_sets(
        ensure_reference_file(args.reference_file.resolve()), set(background)
    )
    marker_summary.to_csv(
        output_dir / "Cortical_Layer_Marker_Gene_Set_Summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rows = []
    for state_index, state in enumerate(STATES):
        for direction_index, direction in enumerate(DIRECTIONS):
            selected = selected_genes(spin_dir, state, direction)
            rng = np.random.default_rng(args.seed + state_index * 10 + direction_index)
            for layer in LAYERS:
                rows.append(
                    one_test(
                        state,
                        direction,
                        selected,
                        layer,
                        markers[layer],
                        background,
                        args.n_permutations,
                        rng,
                    )
                )

    results = pd.DataFrame(rows)
    results["P_FDR_BH_Across_All_Layer_Tests"] = multipletests(
        results["P_Random_GeneSet_Raw"], method="fdr_bh"
    )[1]
    results["Significant_FDR_0p05"] = results["P_FDR_BH_Across_All_Layer_Tests"] < 0.05
    results["N_Permutations"] = args.n_permutations
    results["Random_Seed"] = args.seed
    output_file = output_dir / "Cortical_Layer_Enrichment_State7_FDR.csv"
    results.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已保存 {len(results)} 项皮层层级富集检验。")


if __name__ == "__main__":
    main()
