from __future__ import annotations

"""MLP-only 基线模型预测入口。

加载训练阶段保存的 PyTorch MLP 模型和 pipeline，对输入 CSV 进行坐标预测。
支持 GPU（通过 --device 参数）。

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
import torch
import torch.nn as nn

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
    select_device,
    setup_logging,
)

# ---------------------------------------------------------------------------
# 激活函数映射（与训练脚本保持一致）
# ---------------------------------------------------------------------------
_ACTIVATION_MAP: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "logistic": nn.Sigmoid,
}


class MLPModel(nn.Module):
    """纯 MLP 回归器，与训练脚本中的定义完全一致。"""

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: tuple[int, ...],
        output_dim: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        act_cls = _ACTIVATION_MAP.get(activation)
        if act_cls is None:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Supported: {list(_ACTIVATION_MAP)}"
            )

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        candidates = sorted(
            path
            for path in model_root.glob("*/*")
            if (path.is_dir() and path.parent.name != "log")
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
    device: torch.device,
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

    # 加载 PyTorch 模型
    model_path = model_dir / str(pipeline.get("model_file", "mlp.pt"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"MLP model file not found: {model_path}"
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    input_dim = int(checkpoint["input_dim"])
    hidden_layer_sizes = tuple(checkpoint["hidden_layer_sizes"])
    activation = str(checkpoint["activation"])
    dropout = float(checkpoint["dropout"])

    model = MLPModel(
        input_dim=input_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

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
    scaled = feature_scaler.transform(features.to_numpy()).astype(np.float32)

    # MLP 预测（标准化坐标空间）
    with torch.inference_mode():
        tensor = torch.from_numpy(scaled).to(device, non_blocking=True)
        prediction_scaled = model(tensor).cpu().numpy()

    # 恢复原始坐标尺度
    prediction = target_scaler.inverse_transform(prediction_scaled)

    # 构造输出 DataFrame
    output = frame.copy()
    output["PRED_LONGITUDE"] = prediction[:, 0]
    output["PRED_LATITUDE"] = prediction[:, 1]

    # 输入包含真实坐标时计算误差
    if {"LONGITUDE", "LATITUDE"}.issubset(output.columns):
        truth = output[["LONGITUDE", "LATITUDE"]].to_numpy()
        output["ERROR_DISTANCE"] = np.linalg.norm(truth - prediction, axis=1)

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
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='PyTorch 设备，auto 优先 CUDA，不可用时回退 CPU。',
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

        device = select_device(args.device)

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
            device=device,
            log_file=log_paths.log_file,
        )
    except Exception:
        logging.getLogger("xiaoma").exception("MLP-only prediction failed.")
        raise

    print(f"Prediction file: {result}")


if __name__ == "__main__":
    main()
