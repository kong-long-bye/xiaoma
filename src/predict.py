from __future__ import annotations

"""模型预测入口。

该文件只负责加载训练产物并生成坐标预测，不重新训练模型。
默认可以使用当前 Git 分支下最新一次训练结果。
"""

import argparse
from dataclasses import fields
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import ModelConfig, PROJECT_ROOT
from src.model import WiFiTransformerAutoencoder
from src.utils import (
    create_run_paths,
    get_git_branch,
    latest_run_dir,
    select_device,
    set_seed,
    setup_logging,
)


def _load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, object]:
    """兼容不同 PyTorch 版本加载 transformer.pt。"""

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # 较旧版本的 PyTorch 没有 weights_only 参数。
        return torch.load(
            path,
            map_location=device,
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
        如果当前分支没有模型，再寻找整个 model 目录中的最新模型。
    """

    if model_dir != "latest":
        resolved = Path(model_dir).expanduser().resolve()

        if not resolved.exists():
            raise FileNotFoundError(resolved)

        return resolved

    branch_name = branch or get_git_branch(PROJECT_ROOT)

    try:
        return latest_run_dir(
            model_root,
            branch_name,
        )
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
            raise

        return candidates[-1]


@torch.inference_mode()
def extract_latent(
    model: WiFiTransformerAutoencoder,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """分批提取预测数据的 Transformer latent 特征。"""

    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )

    chunks: list[np.ndarray] = []
    model.eval()

    for (batch,) in loader:
        latent = model.encode(batch.to(device))
        chunks.append(latent.cpu().numpy())

    return np.concatenate(
        chunks,
        axis=0,
    ).astype(np.float32)


def predict(
    input_csv: Path,
    model_dir: Path,
    output_csv: Path,
    device_name: str,
    batch_size: int,
    log_file: Path,
) -> Path:
    """加载模型并对输入 CSV 执行预测。"""

    logger = setup_logging(log_file)

    # 预测阶段同样设置线程数，避免部分 CPU 环境线程调度过慢。
    set_seed(0)
    device = select_device(device_name)

    logger.info("Loading model from %s", model_dir)

    # 加载 sklearn 预处理器、特征列表和 SVR。
    pipeline = joblib.load(
        model_dir / "pipeline.joblib"
    )

    # 加载 Transformer 结构配置和权重。
    checkpoint = _load_checkpoint(
        model_dir / str(pipeline["transformer_file"]),
        device,
    )

    # 只读取当前 ModelConfig 中仍然存在的字段，
    # 使旧模型在增加非必要配置项后仍有机会被加载。
    valid_names = {
        item.name for item in fields(ModelConfig)
    }

    model_config = ModelConfig(
        **{
            key: value
            for key, value
            in checkpoint["model_config"].items()
            if key in valid_names
        }
    )

    model = WiFiTransformerAutoencoder(
        input_dim=int(checkpoint["input_dim"]),
        config=model_config,
    ).to(device)

    model.load_state_dict(
        checkpoint["state_dict"],
        strict=False,
    )

    frame = pd.read_csv(input_csv)
    feature_names = list(pipeline["feature_names"])

    # 预测数据必须至少包含训练时保留下来的全部 WAP 特征。
    missing_columns = [
        column
        for column in feature_names
        if column not in frame.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Input CSV is missing "
            f"{len(missing_columns)} required WAP columns."
        )

    # 预测阶段必须使用与训练阶段完全相同的缺失值规则和 scaler。
    features = (
        frame[feature_names]
        .replace(
            pipeline["raw_missing_value"],
            pipeline["filled_missing_value"],
        )
        .fillna(
            pipeline["filled_missing_value"]
        )
    )

    scaled = pipeline["feature_scaler"].transform(
        features.to_numpy()
    ).astype(np.float32)

    regressor_type = pipeline.get(
        "regressor_type", "svr"
    )

    if regressor_type == "mlp":
        # MLP 模式：使用模型的回归头直接预测坐标。
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(scaled)
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        model.eval()
        chunks: list[np.ndarray] = []

        with torch.inference_mode():
            for (batch,) in loader:
                coords = model(
                    batch.to(device), joint=True
                )
                chunks.append(
                    coords.cpu().numpy()
                )

        prediction = np.concatenate(
            chunks, axis=0
        ).astype(np.float32)

        # 将标准化后的坐标预测还原为原始坐标量级
        target_scaler = pipeline.get(
            "target_scaler"
        )
        if target_scaler is not None:
            prediction = (
                target_scaler.inverse_transform(
                    prediction
                )
            )

    else:
        # SVR 模式：提取 latent 后用 sklearn 回归器。
        latent = extract_latent(
            model,
            scaled,
            device,
            batch_size,
        )

        prediction = pipeline["regressor"].predict(
            latent
        )

    # 在原始 CSV 后追加预测坐标，方便继续分析其他字段。
    output = frame.copy()
    output["PRED_LONGITUDE"] = prediction[:, 0]
    output["PRED_LATITUDE"] = prediction[:, 1]

    # 输入数据包含真实坐标时，额外计算每条记录的二维距离误差。
    if {"LONGITUDE", "LATITUDE"}.issubset(
        output.columns
    ):
        truth = output[
            ["LONGITUDE", "LATITUDE"]
        ].to_numpy()

        output["ERROR_DISTANCE"] = np.linalg.norm(
            truth - prediction,
            axis=1,
        )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
    )

    logger.info(
        "Saved %d predictions to %s",
        len(output),
        output_csv,
    )

    return output_csv


def parse_args() -> argparse.Namespace:
    """定义预测命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Predict coordinates with a trained "
            "Transformer + SVR/MLP model."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="latest",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "model",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    return parser.parse_args()


def main() -> None:
    """预测命令行主函数。"""

    args = parse_args()

    model_dir = resolve_model_dir(
        args.model_dir,
        args.model_root,
        args.branch,
    )

    # 预测只创建日志目录，不创建新的空模型目录。
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
        device_name=args.device,
        batch_size=args.batch_size,
        log_file=log_paths.log_file,
    )

    print(f"Prediction file: {result}")


if __name__ == "__main__":
    main()
