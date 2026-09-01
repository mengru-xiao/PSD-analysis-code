"""将 DPABI ROI 时间序列转换为匿名化的 pyleida 输入。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


N_ROIS = 246


@dataclass(frozen=True)
class InputBlock:
    folder: Path
    time_label: str
    source_group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成匿名化的 pyleida 输入文件。")
    parser.add_argument("--psd-pre", type=Path, required=True)
    parser.add_argument("--psd-post", type=Path, required=True)
    parser.add_argument("--no-psd", type=Path, required=True)
    parser.add_argument("--healthy-controls", type=Path, required=True)
    parser.add_argument("--id-map", type=Path, required=True, help="仅在本地保存的私有ID映射表。")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_private_id_map(path: Path) -> pd.DataFrame:
    """读取私有映射；映射内容绝不写入分析输出。"""
    data = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    required = ["source_group", "source_id", "anonymous_id", "intervention"]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"私有ID映射表缺少字段：{missing}")
    data = data[required].copy()
    if (data[["source_group", "source_id", "anonymous_id"]] == "").any().any():
        raise ValueError("私有ID映射表中的分组、原始标识和匿名ID必须完整。")
    if data.duplicated(["source_group", "source_id"]).any():
        raise ValueError("私有ID映射表存在重复原始标识。")
    if data["anonymous_id"].duplicated().any():
        raise ValueError("私有ID映射表中的 anonymous_id 必须唯一。")
    if data["anonymous_id"].str.contains(r"[^A-Za-z0-9-]", regex=True).any():
        raise ValueError("anonymous_id 只能包含字母、数字和连字符。")
    psd = data["source_group"].eq("PSD")
    invalid = psd & ~data["intervention"].isin(["Acupuncture", "Sham"])
    if invalid.any():
        raise ValueError("PSD 映射记录的 intervention 必须为 Acupuncture 或 Sham。")
    return data.set_index(["source_group", "source_id"])


def load_roi_signals(mat_file: Path) -> np.ndarray:
    mat = loadmat(mat_file)
    if "ROISignals" not in mat:
        raise KeyError("某个 MAT 文件缺少 ROISignals 变量。")
    signals = np.asarray(mat["ROISignals"], dtype=float)
    if signals.ndim != 2 or signals.shape[1] != N_ROIS:
        raise ValueError(f"某个 ROI 矩阵应为 time × {N_ROIS}，实际为 {signals.shape}。")
    if not np.isfinite(signals).all():
        raise ValueError("某个 ROI 矩阵包含 NaN 或无穷值。")
    return signals.T


def process_block(
    block: InputBlock,
    time_series_dir: Path,
    id_map: pd.DataFrame,
) -> list[dict[str, object]]:
    if not block.folder.is_dir():
        raise FileNotFoundError(f"缺少 {block.source_group}-{block.time_label} 输入目录。")
    files = sorted(block.folder.glob("ROISignals_*.mat"))
    if not files:
        raise FileNotFoundError(f"{block.source_group}-{block.time_label} 中没有 ROI MAT 文件。")

    rows: list[dict[str, object]] = []
    missing_map_count = 0
    for mat_file in files:
        source_id = mat_file.stem.removeprefix("ROISignals_")
        key = (block.source_group, source_id)
        if key not in id_map.index:
            missing_map_count += 1
            continue
        mapping = id_map.loc[key]
        anonymous_id = str(mapping["anonymous_id"])
        subject_id = f"{anonymous_id}_{block.time_label}"
        condition = (
            f"{mapping['intervention']}_{block.time_label}"
            if block.source_group == "PSD"
            else block.source_group
        )

        signals = load_roi_signals(mat_file)
        subject_dir = time_series_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(signals).to_csv(
            subject_dir / f"{subject_id}.csv", index=False, header=False
        )
        rows.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "n_rois": signals.shape[0],
                "n_timepoints": signals.shape[1],
            }
        )

    if missing_map_count:
        raise ValueError(f"有 {missing_map_count} 个输入文件未在私有ID映射表中找到。")
    print(f"{block.source_group}-{block.time_label}：已处理 {len(rows)} 个扫描。")
    return rows


def main() -> None:
    args = parse_args()
    output_dir = args.output.resolve()
    time_series_dir = output_dir / "time_series"
    time_series_dir.mkdir(parents=True, exist_ok=True)
    id_map = read_private_id_map(args.id_map.resolve())

    blocks = [
        InputBlock(args.psd_pre.resolve(), "Pre", "PSD"),
        InputBlock(args.psd_post.resolve(), "Post", "PSD"),
        InputBlock(args.no_psd.resolve(), "Baseline", "noPSD"),
        InputBlock(args.healthy_controls.resolve(), "Baseline", "HC"),
    ]
    metadata_rows: list[dict[str, object]] = []
    for block in blocks:
        metadata_rows.extend(process_block(block, time_series_dir, id_map))

    metadata = pd.DataFrame(metadata_rows)
    duplicate_count = int(metadata["subject_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"生成了 {duplicate_count} 条重复匿名扫描ID。")

    metadata.to_csv(output_dir / "metadata_with_qc.csv", index=False, encoding="utf-8-sig")
    metadata[["subject_id", "condition"]].to_csv(
        output_dir / "metadata.csv", index=False, encoding="utf-8-sig"
    )
    print(f"已生成 {len(metadata)} 条匿名扫描记录。")


if __name__ == "__main__":
    main()
