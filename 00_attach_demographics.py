"""将年龄和性别协变量合并到三个 K=8 动态指标表。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_FILES = {
    "occupancies": "occupancies.csv",
    "dwell_times": "dwell_times.csv",
    "transitions_probabilities": "transitions_probabilities.csv",
}


def read_metric_table(path: Path) -> pd.DataFrame:
    """兼容逗号或制表符分隔的 pyleida 指标表。"""
    data = pd.read_csv(path, sep=None, engine="python")
    if len(data.columns) == 1:
        raise ValueError("无法识别指标表的分隔符。")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并扫描级人口学协变量。")
    parser.add_argument(
        "--demographics",
        type=Path,
        required=True,
        help="私有 CSV，包含匿名 subject_id、Age 和 Sex。",
    )
    parser.add_argument("--metric-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def read_demographics(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {"subject_id", "Age", "Sex"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"人口学表缺少字段：{missing}")
    data = data[["subject_id", "Age", "Sex"]].copy()
    if data["subject_id"].isna().any() or data["subject_id"].duplicated().any():
        raise ValueError("人口学表中的匿名 subject_id 必须完整且唯一。")
    if data[["Age", "Sex"]].isna().any().any():
        raise ValueError("每条记录的 Age 和 Sex 必须完整。")
    return data


def main() -> None:
    args = parse_args()
    demographics = read_demographics(args.demographics.resolve())
    input_dir = args.metric_directory.resolve()
    output_dir = args.output_directory.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, filename in METRIC_FILES.items():
        input_file = input_dir / filename
        if not input_file.exists():
            raise FileNotFoundError(f"缺少指标文件：{filename}")
        values = read_metric_table(input_file)
        if "subject_id" not in values.columns:
            raise ValueError(f"{metric} 表缺少 subject_id 字段。")
        if values["subject_id"].duplicated().any():
            raise ValueError(f"{metric} 表中存在重复匿名ID。")

        merged = values.merge(
            demographics,
            on="subject_id",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        unmatched_count = int((merged["_merge"] != "both").sum())
        if unmatched_count:
            raise ValueError(f"{metric} 有 {unmatched_count} 条记录未匹配人口学信息。")
        merged = merged.drop(columns="_merge")
        output_file = output_dir / f"{metric}_age_sex.csv"
        merged.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"{metric}：已保存 {len(merged)} 条记录。")


if __name__ == "__main__":
    main()
