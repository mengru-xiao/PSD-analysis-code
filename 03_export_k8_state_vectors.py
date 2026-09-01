"""导出 K=8 聚类中心，并转置为 ROI × State 矩阵。"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 K=8 LEiDA 状态向量。")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--roi-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError("找不到 K=8 模型文件。")
    if not args.roi_labels.exists():
        raise FileNotFoundError("找不到 ROI 标签文件。")

    # 仅加载自己生成且可信的 pickle 文件。
    with args.model.open("rb") as handle:
        model = pickle.load(handle)
    if not hasattr(model, "cluster_centers_"):
        raise AttributeError("The saved K-means model has no cluster_centers_ attribute.")

    centers = np.asarray(model.cluster_centers_, dtype=float)
    labels = [line.strip() for line in args.roi_labels.read_text(encoding="utf-8").splitlines()]
    if centers.shape != (8, 246):
        raise ValueError(f"Expected 8 states x 246 ROIs, found {centers.shape}.")
    if len(labels) != 246 or len(set(labels)) != 246:
        raise ValueError("ROI label file must contain 246 unique non-empty labels.")
    if not all(labels) or not np.isfinite(centers).all():
        raise ValueError("Empty ROI labels or non-finite centroid values detected.")

    state_vectors = pd.DataFrame(
        centers.T,
        index=pd.Index(labels, name="Label"),
        columns=[f"State_{number}" for number in range(1, 9)],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state_vectors.to_csv(args.output, encoding="utf-8-sig")
    print(f"已导出 {state_vectors.shape[0]} 个 ROI × {state_vectors.shape[1]} 个状态。")


if __name__ == "__main__":
    main()
