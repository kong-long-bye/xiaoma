from __future__ import annotations

"""MLP-only 基线模型预测入口。

加载训练阶段保存的 MLP 模型和 pipeline，对输入 CSV 进行坐标预测。
不需要 PyTorch 和 Transformer。

支持两种启动方式：
    python src/mlp-single/mlp-single-predict.py --input data/validationData.csv
    cd src/mlp-single && python mlp-single-predict.py --input ../data/validationData.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# sys.path 处理
# ---------------------------------------------------------------------------
_current_file = Path(__file__).resolve()
_mlp_single_dir = _current_file.parent
_src_dir = _mlp_single_dir.parent
_project_root = _src_dir.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# mlp-single 含连字符，不是合法包名，将目录加入 sys.path 后直接导入模块
if str(_mlp_single_dir) not in sys.path:
    sys.path.insert(0, str(_mlp_single_dir))

# ---------------------------------------------------------------------------
# 导入项目现有模块
# ---------------------------------------------------------------------------
from src.config import PROJECT_ROOT
from mlp_single_config import MODEL_ROOT
from src.utils import (
    create_run_paths,
    get_git_branch,
    latest_run_dir,
    setup_logging,
)


def resolve_model_dir(
    model_dir: str,
    model_root: Path,
    branch: str | None,
) -> Path:
    """解析用户指定的模型目录。

    model_dir 不是 latest:
        直接使用明确指定的目录。

    model_dir 是 latest:
        优先寻找当前分支的最新模型；
        如果当前分支没有模型，再寻找整个 model/mlp-single 目录中的最新模型。
    """

    if model_dir != "latest":
        resolved = Path(model_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Model directory not found: {resolved}"
            )
        return resolved

    branch_name = branch or get_git_branch(PROJECT_ROOT)

    try:
        return latest_run_dir(model_root, branch_name)
    except FileNotFoundError:
        # 非 Git 环境或切换分支后，允许回退到全局最新模型。
        candidates = sorted(
            path
            for path in model_root.glob("*/*")
            if (
                path.is_dir()
                and path.parent.name != "log"
            )
        )

        if not candidates:
            raise FileNotFoundError(
                f"No trained MLP-only model was found under {model_root}."
            )

        return candidates[-1]


def predict(
    input_csv: Path,
    model_dir: Path,
    output_csv: Path,
    log_file: Path,
) -> Path:
    """加载 MLP-only 模型并对输入 CSV 执行预测。"""

    logger = setup_logging(log_file)

    logger.info("Loading model from %s", model_dir)

    # 加载 pipeline
    pipeline_path = model_dir / "pipeline.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"pipeline.joblib not found in {model_dir}."
        )

    pipeline = joblib.load(pipeline_path)

    # 检查模型类型
    if pipeline.get("model_type") != "mlp-single":
        raise ValueError(
            f"pipeline.joblib model_type is "
            f"'{pipeline.get('model_type')}', expected 'mlp-single'. "
            f"This pipeline is not an MLP-only model."
        )

    # 加载 MLP 模型
    model_path = model_dir / str(pipeline.get("model_file", "mlp.joblib"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"MLP model file not found: {model_path}"
        )

    mlp = joblib.load(model_path)

    feature_names: list[str] = list(pipeline["feature_names"])
    feature_scaler = pipeline["feature_scaler"]
    target_scaler = pipeline["target_scaler"]
    raw_missing_value = float(pipeline["raw_missing_value"])
    filled_missing_value = float(pipeline["filled_missing_value"])

    logger.info("Loaded MLP model from %s", model_path)
    logger.info("Model expects %d features.", len(feature_names))

    # 读取输入 CSV
    logger.info("Reading input: %s", input_csv)
    frame = pd.read_csv(input_csv)
    logger.info("Input shape: %s", frame.shape)

    # 检查必需的 WAP 列
    missing_columns = [
        col for col in feature_names if col not in frame.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Input CSV is missing {len(missing_columns)} "
            f"required WAP columns: {missing_columns[:5]}"
        )

    # 使用与训练时相同的缺失值替换逻辑
    features = (
        frame[feature_names]
        .replace(raw_missing_value, filled_missing_value)
        .fillna(filled_missing_value)
    )

    # 特征缩放
    scaled = feature_scaler.transform(
        features.to_numpy()
    ).astype(np.float32)

    # MLP 预测（标准化坐标空间）
    prediction_scaled = mlp.predict(scaled)

    # 恢复原始坐标尺度
    prediction = target_scaler.inverse_transform(prediction_scaled)

    # 构造输出 DataFrame
    output = frame.copy()
    output["PRED_LONGITUDE"] = prediction[:, 0]
    output["PRED_LATITUDE"] = prediction[:, 1]

    # 输入包含真实坐标时计算误差
    if {"LONGITUDE", "LATITUDE"}.issubset(output.columns):
        truth = output[["LONGITUDE", "LATITUDE"]].to_numpy()
        output["ERROR_DISTANCE"] = np.linalg.norm(
            truth - prediction, axis=1
        )

    # 保存结果
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8")

    logger.info("Saved %d predictions to %s", len(output), output_csv)
    return output_csv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """定义 MLP-only 预测命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Predict coordinates with a trained MLP-only model.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入 CSV 文件路径。",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="latest",
        help="模型目录路径，或 'latest' 使用最新的 MLP-only 模型。",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=MODEL_ROOT,
        help="模型根目录（默认 model/mlp-single）。",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Git 分支名。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 CSV 路径（默认自动生成）。",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """MLP-only 预测命令行主函数。"""

    args = parse_args(argv)

    try:
        model_dir = resolve_model_dir(
            args.model_dir,
            args.model_root,
            args.branch,
        )

        # 只创建日志目录，不创建空模型目录
        log_paths = create_run_paths(
            model_root=args.model_root,
            project_root=PROJECT_ROOT,
            branch=args.branch,
            log_suffix="predict",
            create_artifact_dir=False,
        )

        output = args.output or (
            model_dir
            / (
                f"predictions_{args.input.stem}_"
                f"{log_paths.timestamp}.csv"
            )
        )

        result = predict(
            input_csv=args.input,
            model_dir=model_dir,
            output_csv=output,
            log_file=log_paths.log_file,
        )
    except Exception:
        logging.getLogger("xiaoma").exception("MLP-only prediction failed.")
        raise

    print(f"Prediction file: {result}")


if __name__ == "__main__":
    main()
