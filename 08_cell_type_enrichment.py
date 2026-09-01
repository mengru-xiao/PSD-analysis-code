"""预先指定 State 7 的 Seidlitz 细胞类型富集，不包含绘图。"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_PATH = SCRIPT_DIR / "06_ahba_pls_shared.py"
FORMAL_STATES = ("State_7",)
GENE_FILES = {
    "Higher_State_Vector": "Genes_Associated_With_Higher_State_Vector_Zgt3p0_FDR05.txt",
    "Lower_State_Vector": "Genes_Associated_With_Lower_State_Vector_ZltMinus3p0_FDR05.txt",
}


def load_shared_module():
    spec = importlib.util.spec_from_file_location("ahba_pls_shared", SHARED_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("无法加载 AHBA 共用计算模块。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load_shared_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 State 7 Seidlitz 细胞类型富集。")
    parser.add_argument("--spin-results", type=Path, required=True)
    parser.add_argument("--seidlitz-file", type=Path, required=True)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def read_background_genes(expression_file: Path) -> np.ndarray:
    """只读取表达矩阵的基因列名。"""

    if not expression_file.exists():
        raise FileNotFoundError("找不到 AHBA 表达矩阵。")
    columns = pd.read_csv(expression_file, nrows=0).columns.astype(str).tolist()
    if columns and (columns[0].startswith("Unnamed:") or columns[0] in {"label", "Label"}):
        columns = columns[1:]
    genes = np.asarray(columns, dtype=str)
    if len(genes) == 0 or len(set(genes)) != len(genes):
        raise ValueError("AHBA expression table must contain unique gene-name columns.")
    return genes


def read_selected_genes(spin_dir: Path, state: str, direction: str) -> set[str]:
    path = spin_dir / state / "Gene_Selection_Thresholded" / GENE_FILES[direction]
    if not path.exists():
        raise FileNotFoundError("找不到 State 7 基因列表。")
    return {gene.strip() for gene in path.read_text(encoding="utf-8").splitlines() if gene.strip()}


def main() -> None:
    args = parse_args()
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be positive.")
    spin_dir = args.spin_results.resolve()
    model_file = spin_dir / "Model_Summary_State7.csv"
    expression_file = (
        spin_dir / "Common_Inputs_And_QC" / "AHBA_Expression_Left_Cortical_SurfaceSpin.csv"
    )
    if not model_file.exists():
        raise FileNotFoundError("找不到 State 7 空间模型汇总表。")

    shared.SEIDLITZ_FILE = args.seidlitz_file.resolve()
    model_summary = pd.read_csv(model_file).set_index("State")
    missing_states = sorted(set(FORMAL_STATES).difference(model_summary.index))
    if missing_states:
        raise ValueError(f"Model summary is missing states: {missing_states}")

    background = read_background_genes(expression_file)
    output_dir = spin_dir / "Cell_Type_Enrichment_Seidlitz"
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_gene_sets = shared.load_seidlitz_gene_sets(set(background), output_dir)

    rows: list[dict[str, object]] = []
    for state_index, state in enumerate(FORMAL_STATES):
        overall_pass = bool(model_summary.loc[state, "Overall_State_Spatial_Passed"])
        if not overall_pass:
            raise ValueError(
                "State 7 的预设整体空间检验未通过，不能进行正式下游富集解释。"
            )
        for direction_index, direction in enumerate(GENE_FILES):
            selected = read_selected_genes(spin_dir, state, direction)
            rows.extend(
                shared.cell_type_enrichment(
                    state=state,
                    direction=direction,
                    selected_genes=selected,
                    background_genes=background,
                    cell_gene_sets=cell_gene_sets,
                    n_perm=args.n_permutations,
                    seed=args.seed + state_index * 10 + direction_index,
                    overall_pass=overall_pass,
                )
            )

    results = pd.DataFrame(rows)
    valid = results["P_Random_GeneSet_Raw"].notna()
    results["P_FDR_BH_Across_All_Cell_Tests"] = np.nan
    results.loc[valid, "P_FDR_BH_Across_All_Cell_Tests"] = multipletests(
        results.loc[valid, "P_Random_GeneSet_Raw"], method="fdr_bh"
    )[1]
    results["Significant_FDR_0p05"] = results["P_FDR_BH_Across_All_Cell_Tests"] < 0.05
    results["N_Permutations"] = args.n_permutations
    results["Random_Seed"] = args.seed
    results = results.sort_values(
        ["P_FDR_BH_Across_All_Cell_Tests", "P_Random_GeneSet_Raw"], na_position="last"
    )
    output_file = output_dir / "Cell_Type_Enrichment_State7_FDR.csv"
    results.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已保存 {len(results)} 项细胞类型富集检验。")


if __name__ == "__main__":
    main()
