"""DSRS28 分析变量的表格化正态性检查，不包含绘图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


VARIABLES = {
    "DSRS28": "Day-28 DSRS; higher values indicate worse swallowing outcome.",
    "DSRS": "Baseline DSRS used as the baseline clinical covariate.",
    "age": "Age in years, entered as a continuous covariate.",
    "PL_state_2_dwell_post_pre": "State 2 dwell-time change (post minus pre).",
    "PL_state_6_dwell_post_pre": "State 6 dwell-time change (post minus pre).",
    "PL_state_7_dwell_post_pre": "State 7 dwell-time change (post minus pre).",
    "PL_state_2_occupancies_post_pre": "State 2 fractional-occupancy change (post minus pre).",
    "PL_state_6_occupancies_post_pre": "State 6 fractional-occupancy change (post minus pre).",
    "PL_state_7_occupancies_post_pre": "State 7 fractional-occupancy change (post minus pre).",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 DSRS28 分析变量的正态性。")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def load_complete_cases(path: Path) -> tuple[pd.DataFrame, int]:
    if not path.exists():
        raise FileNotFoundError("找不到正态性检查输入表。")
    data = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(set(VARIABLES).difference(data.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")
    candidates = data.dropna(subset=list(VARIABLES), how="all").copy()
    incomplete = int(candidates[list(VARIABLES)].isna().any(axis=1).sum())
    complete = candidates.dropna(subset=list(VARIABLES)).copy()
    if len(complete) < 3:
        raise ValueError(f"Shapiro-Wilk requires at least 3 complete observations; found {len(complete)}.")
    return complete, incomplete


def describe_and_test(values: pd.Series, variable: str) -> dict[str, object]:
    clean = values.astype(float)
    if clean.nunique() < 2:
        raise ValueError(f"{variable} is constant and cannot be meaningfully tested.")
    shapiro_w, shapiro_p = stats.shapiro(clean)
    return {
        "Variable": variable,
        "Variable_Meaning": VARIABLES[variable],
        "N": len(clean),
        "Mean": float(clean.mean()),
        "SD": float(clean.std(ddof=1)),
        "Median": float(clean.median()),
        "Q1": float(clean.quantile(0.25)),
        "Q3": float(clean.quantile(0.75)),
        "Minimum": float(clean.min()),
        "Maximum": float(clean.max()),
        "Skewness": float(stats.skew(clean, bias=False)),
        "Excess_Kurtosis": float(stats.kurtosis(clean, fisher=True, bias=False)),
        "Shapiro_W": float(shapiro_w),
        "P_Shapiro": float(shapiro_p),
        "NonNormal_at_0p05": bool(shapiro_p < 0.05),
        "Interpretation": (
            "Evidence of departure from normality"
            if shapiro_p < 0.05
            else "No statistically detectable departure from normality"
        ),
    }


def main() -> None:
    args = parse_args()
    data, incomplete_count = load_complete_cases(args.input.resolve())
    results = pd.DataFrame(
        [describe_and_test(data[variable], variable) for variable in VARIABLES]
    )
    results["Complete_Case_N"] = len(data)
    results["Rows_Excluded_For_Any_Missing_Analysis_Variable"] = incomplete_count

    output_dir = args.output_directory.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "normality_test_results.csv"
    results.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已保存 {len(results)} 项正态性检查（完整样本 N={len(data)}）。")


if __name__ == "__main__":
    main()
