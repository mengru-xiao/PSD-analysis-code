"""运行合并样本的 LEiDA 分解及动态状态指标计算，不包含绘图。"""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# 兼容旧版 pyleida 对 NumPy 类型别名的调用，不改变计算。
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

from pyleida import Leida  # noqa: E402  (import after NumPy compatibility shim)


REQUIRED_FILES = ("metadata.csv", "rois_labels.txt", "rois_coordinates.CSV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行合并样本的 LEiDA 分析。")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--working-directory",
        type=Path,
        required=True,
        help="pyleida 写入分析结果的目录。",
    )
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def validate_input(data_dir: Path) -> dict[str, int]:
    """检查 pyleida 输入结构和矩阵维度。"""

    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"输入目录缺少必要文件：{missing}")

    time_series_dir = data_dir / "time_series"
    if not time_series_dir.is_dir():
        raise FileNotFoundError("输入目录缺少 time_series 子目录。")
    csv_files = sorted(time_series_dir.glob("*/*.csv"))
    if not csv_files:
        raise FileNotFoundError("time_series 中没有可分析的 CSV 文件。")

    shapes: set[tuple[int, int]] = set()
    for csv_file in csv_files:
        array = np.loadtxt(csv_file, delimiter=",")
        if array.ndim != 2 or array.shape[0] != 246:
            raise ValueError(f"某个时间序列应为 246 x time，实际为 {array.shape}。")
        if not np.isfinite(array).all():
            raise ValueError("某个时间序列包含 NaN 或无穷值。")
        shapes.add(array.shape)
    if len({shape[1] for shape in shapes}) != 1:
        raise ValueError("各扫描的时间点数量不一致。")
    return {"n_scans": len(csv_files), "n_rois": 246, "n_timepoints": next(iter(shapes))[1]}


def package_version() -> str:
    try:
        from importlib.metadata import version

        return version("pyleida")
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be positive.")

    data_dir = args.data.resolve()
    working_dir = args.working_directory.resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    input_qc = validate_input(data_dir)

    os.chdir(working_dir)
    analysis = Leida(str(data_dir))
    analysis.fit_predict(
        TR=None,                 # dwell time remains in acquired volumes
        paired_tests=False,     # LMM is the prespecified longitudinal test
        n_perm=args.n_permutations,
        save_results=True,
        random_state=args.random_seed,
    )

    run_record = {
        "analysis": "pooled LEiDA; all scans share one state solution",
        "candidate_k": "2-20 (pyleida default used in the study)",
        "selected_k": 8,
        "selection_criterion": "minimum Davies-Bouldin index",
        "TR": None,
        "paired_tests": False,
        "n_permutations": args.n_permutations,
        "random_seed": args.random_seed,
        "input_qc": input_qc,
        "python": platform.python_version(),
        "pyleida": package_version(),
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    record_path = working_dir / "LEiDA_results" / "run_configuration.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("LEiDA 分析完成，已保存运行参数。")


if __name__ == "__main__":
    main()
