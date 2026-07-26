from __future__ import annotations

"""MLP-only 基线模型训练入口。

用于 Transformer 消融实验，完全移除 Transformer 编码器，
直接将预处理后的 WAP 特征输入 sklearn MLPRegressor 预测坐标。

支持两种启动方式：
    python src/mlp-single/mlp-single.py
    cd src/mlp-single && python mlp-single.py
"""

import argparse
import copy
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# sys.path 处理：保证无论从项目根目录还是从 src/mlp-single/ 启动都能导入 src.*
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
from src.config import DataConfig, PROJECT_ROOT
from mlp_single_config import (
    MLPSingleConfig,
    MODEL_ROOT,
    parse_layer_sizes,
)
from src.train import prepare_data
from src.utils import (
    create_run_paths,
    distance_metrics,
    save_json,
    set_seed,
    setup_logging,
)


def _prediction_frame(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """将预测坐标和误差附加到原始数据表。"""
    distances = np.linalg.norm(y_true - y_pred, axis=1)
    output = metadata.copy()
    output["PRED_LONGITUDE"] = y_pred[:, 0]
    output["PRED_LATITUDE"] = y_pred[:, 1]
    output["ERROR_DISTANCE"] = distances
    return output


def train_mlp(
    mlp_config: MLPSingleConfig,
    data,
    logger: logging.Logger,
) -> tuple[MLPRegressor, StandardScaler, pd.DataFrame, int, float]:
    """训练 MLP 回归器并返回最佳模型、target_scaler、训练历史等。

    Parameters
    ----------
    mlp_config : MLPSingleConfig
        MLP 超参数配置。
    data : PreparedData
        prepare_data() 返回的数据对象。
    logger : logging.Logger
        日志记录器。

    Returns
    -------
    best_model : MLPRegressor
        验证损失最佳的模型（deepcopy 恢复后的状态）。
    target_scaler : StandardScaler
        在 y_train 上拟合的标准化器。
    history_df : pd.DataFrame
        每轮训练/验证损失和相关信息。
    best_epoch : int
        取得最佳验证损失的 epoch 编号。
    best_val_loss : float
        最佳验证损失（标准化空间）。
    """
    # -----------------------------------------------------------------------
    # 目标值缩放：只使用 y_train 拟合 scaler，避免数据泄漏
    # -----------------------------------------------------------------------
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(data.y_train)
    y_validation_scaled = target_scaler.transform(data.y_validation)
    y_evaluation_scaled = target_scaler.transform(data.y_evaluation)

    logger.info("Target scaler fitted on y_train (mean=%s, std=%s)",
                target_scaler.mean_, target_scaler.scale_)

    # -----------------------------------------------------------------------
    # 构造 MLPRegressor：warm_start=True 逐 epoch 训练
    # -----------------------------------------------------------------------
    mlp = MLPRegressor(
        hidden_layer_sizes=mlp_config.hidden_layer_sizes,
        activation=mlp_config.activation,
        solver=mlp_config.solver,
        alpha=mlp_config.alpha,
        learning_rate_init=mlp_config.learning_rate_init,
        batch_size=mlp_config.batch_size,
        max_iter=1,
        warm_start=True,
        early_stopping=False,
        random_state=mlp_config.seed,
        verbose=False,
    )

    # 计算模型参数量
    n_input = data.X_train.shape[1]
    n_output = 2
    param_count = 0
    layers = [n_input] + list(mlp_config.hidden_layer_sizes) + [n_output]
    for i in range(len(layers) - 1):
        n_params = layers[i] * layers[i + 1] + layers[i + 1]  # weights + biases
        param_count += n_params
        logger.info("  Layer %d: %d -> %d (%d parameters)",
                     i + 1, layers[i], layers[i + 1], n_params)

    logger.info("MLP total parameters: %d", param_count)
    logger.info("MLP config: %s", mlp_config.to_dict())

    # -----------------------------------------------------------------------
    # 逐 epoch 训练循环
    # -----------------------------------------------------------------------
    best_val_loss = float("inf")
    best_model: MLPRegressor | None = None
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, mlp_config.epochs + 1):
        # 训练一个迭代
        mlp.fit(data.X_train, y_train_scaled)

        # 计算训练损失（标准化空间）
        y_train_pred = mlp.predict(data.X_train)
        train_loss = float(mean_squared_error(y_train_scaled, y_train_pred))

        # 计算验证损失（标准化空间）
        y_val_pred = mlp.predict(data.X_validation)
        val_loss = float(mean_squared_error(y_validation_scaled, y_val_pred))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
        })

        logger.info(
            "Epoch %03d/%03d | train_loss=%.8f | validation_loss=%.8f",
            epoch,
            mlp_config.epochs,
            train_loss,
            val_loss,
        )

        # 检查是否是最佳验证损失
        if val_loss < best_val_loss - mlp_config.min_delta:
            best_val_loss = val_loss
            best_model = copy.deepcopy(mlp)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= mlp_config.early_stopping_patience:
                logger.info(
                    "Early stopping triggered at epoch %d "
                    "(no improvement for %d consecutive epochs).",
                    epoch,
                    mlp_config.early_stopping_patience,
                )
                break

    if best_model is None:
        raise RuntimeError(
            "MLP training did not produce a valid best model."
        )

    logger.info(
        "Best model at epoch %d with validation_loss=%.8f",
        best_epoch,
        best_val_loss,
    )

    history_df = pd.DataFrame(history)

    # 验证评估集损失（标准化空间）
    y_eval_pred = best_model.predict(data.X_evaluation)
    eval_loss = float(mean_squared_error(y_evaluation_scaled, y_eval_pred))
    logger.info("Evaluation loss (scaled space): %.8f", eval_loss)

    return best_model, target_scaler, history_df, best_epoch, best_val_loss


def run_training(
    mlp_config: MLPSingleConfig,
    data_config: DataConfig,
    model_root: Path,
    branch: str | None,
    smoke_test: bool,
    smoke_train_rows: int,
    smoke_eval_rows: int,
) -> Path:
    """执行完整 MLP-only 训练流程并返回产物目录。"""

    paths = create_run_paths(
        model_root=model_root,
        project_root=PROJECT_ROOT,
        branch=branch,
    )

    logger = setup_logging(paths.log_file)

    logger.info("Branch: %s", paths.branch)
    logger.info("Artifacts: %s", paths.artifact_dir)
    logger.info("UTF-8 log: %s", paths.log_file)

    set_seed(mlp_config.seed)
    logger.info("Random seed: %d", mlp_config.seed)

    # -------------------------------------------------------------------
    # 数据准备：直接复用现有 prepare_data()
    # -------------------------------------------------------------------
    data = prepare_data(
        data_config,
        seed=mlp_config.seed,
        smoke_train_rows=(smoke_train_rows if smoke_test else None),
        smoke_eval_rows=(smoke_eval_rows if smoke_test else None),
    )

    logger.info("Training samples: %d", len(data.X_train))
    logger.info("Validation samples: %d", len(data.X_validation))
    logger.info("Evaluation samples: %d", len(data.X_evaluation))
    logger.info("Input features: %d", data.X_train.shape[1])

    # -------------------------------------------------------------------
    # 训练 MLP
    # -------------------------------------------------------------------
    best_model, target_scaler, history_df, best_epoch, best_val_loss = train_mlp(
        mlp_config=mlp_config,
        data=data,
        logger=logger,
    )

    # -------------------------------------------------------------------
    # 恢复最佳 MLP 并在原始坐标空间计算指标
    # -------------------------------------------------------------------
    validation_pred_scaled = best_model.predict(data.X_validation)
    evaluation_pred_scaled = best_model.predict(data.X_evaluation)

    validation_pred = target_scaler.inverse_transform(validation_pred_scaled)
    evaluation_pred = target_scaler.inverse_transform(evaluation_pred_scaled)

    metrics: dict[str, Any] = {
        "validation": distance_metrics(data.y_validation, validation_pred),
        "evaluation": distance_metrics(data.y_evaluation, evaluation_pred),
        "preprocessing": data.summary,
        "model_type": "mlp-single",
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
    }

    logger.info("Validation metrics: %s", metrics["validation"])
    logger.info("Evaluation metrics: %s", metrics["evaluation"])

    # -------------------------------------------------------------------
    # 保存产物
    # -------------------------------------------------------------------
    joblib.dump(best_model, paths.artifact_dir / "mlp.joblib")

    joblib.dump(
        {
            "model_type": "mlp-single",
            "model_file": "mlp.joblib",
            "feature_names": data.feature_names,
            "feature_scaler": data.scaler,
            "target_scaler": target_scaler,
            "raw_missing_value": float(data_config.raw_missing_value),
            "filled_missing_value": float(data_config.filled_missing_value),
        },
        paths.artifact_dir / "pipeline.joblib",
    )

    history_df.to_csv(
        paths.artifact_dir / "training_history.csv",
        index=False,
        encoding="utf-8",
    )

    _prediction_frame(
        data.validation_metadata,
        data.y_validation,
        validation_pred,
    ).to_csv(
        paths.artifact_dir / "predictions_validation.csv",
        index=False,
        encoding="utf-8",
    )

    _prediction_frame(
        data.evaluation_metadata,
        data.y_evaluation,
        evaluation_pred,
    ).to_csv(
        paths.artifact_dir / "predictions_evaluation.csv",
        index=False,
        encoding="utf-8",
    )

    save_json(mlp_config.to_dict(), paths.artifact_dir / "config.json")
    save_json(metrics, paths.artifact_dir / "metrics.json")

    logger.info("Training completed successfully.")
    return paths.artifact_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """定义 MLP-only 训练命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Train an MLP-only baseline for UJIIndoorLoc.",
    )

    parser.add_argument(
        "--train-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "trainingData.csv",
        help="训练集 CSV 文件路径。",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "validationData.csv",
        help="官方评估集 CSV 文件路径。",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=MODEL_ROOT,
        help="模型产物输出根目录（默认 model/mlp-single）。",
    )
    parser.add_argument(
        "--building-id",
        type=int,
        default=-1,
        help="建筑编号筛选：-1 使用全部，0/1/2 只使用指定建筑。",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="运行分支名，用于组织产物子目录。",
    )
    parser.add_argument(
        "--hidden-layer-sizes",
        type=str,
        default="256,128,64",
        help="MLP 隐藏层大小，逗号分隔，如 256,128,64。",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="MLP 激活函数（relu, tanh, logistic）。",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="MLP L2 正则化参数。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="MLP 初始学习率。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="MLP mini-batch 大小。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="最大训练轮数。",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="验证损失无改善时最多等待的轮数。",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-5,
        help="验证损失改善的最小阈值。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="开启冒烟测试，只读取少量数据快速验证全流程。",
    )
    parser.add_argument(
        "--smoke-train-rows",
        type=int,
        default=512,
        help="冒烟测试时读取的训练集行数。",
    )
    parser.add_argument(
        "--smoke-eval-rows",
        type=int,
        default=128,
        help="冒烟测试时读取的评估集行数。",
    )

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> tuple[MLPSingleConfig, DataConfig, Path]:
    """用命令行参数构造 MLP 和数据配置。"""

    mlp_config = MLPSingleConfig(
        hidden_layer_sizes=parse_layer_sizes(args.hidden_layer_sizes),
        activation=args.activation,
        alpha=args.alpha,
        learning_rate_init=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        seed=args.seed,
    )

    data_config = DataConfig(
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        building_id=args.building_id,
    )

    model_root = args.model_root

    return mlp_config, data_config, model_root


def main(argv: list[str] | None = None) -> None:
    """MLP-only 训练命令行主函数。"""

    args = parse_args(argv)
    mlp_config, data_config, model_root = build_config(args)

    try:
        artifact_dir = run_training(
            mlp_config=mlp_config,
            data_config=data_config,
            model_root=model_root,
            branch=args.branch,
            smoke_test=args.smoke_test,
            smoke_train_rows=args.smoke_train_rows,
            smoke_eval_rows=args.smoke_eval_rows,
        )
    except Exception:
        logging.getLogger("xiaoma").exception("MLP-only training failed.")
        raise

    print(f"Model artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
