from __future__ import annotations

"""模型训练入口。

完整流程:
1. 读取 UJIIndoorLoc CSV；
2. 清洗缺失 WiFi 信号并删除几乎全缺失的 WAP；
3. 划分训练集和内部验证集；
4. 使用 Transformer 自编码器学习 WiFi 表示；
5. 选择回归器：
   - "svr": 提取 latent -> 多输出 SVR 回归经纬度；
   - "mlp": 使用 MLP 回归头端到端联合训练；
6. 保存模型、配置、指标、预测结果和 UTF-8 日志。
"""

import argparse
import copy
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    AppConfig,
    DataConfig,
    TrainingConfig,
)
from src.model import WiFiTransformerAutoencoder
from src.utils import (
    create_run_paths,
    distance_metrics,
    save_json,
    select_device,
    set_seed,
    setup_logging,
)


@dataclass
class PreparedData:
    """预处理完成后，训练流程需要使用的全部数据。"""

    X_train: np.ndarray
    X_validation: np.ndarray
    X_evaluation: np.ndarray

    y_train: np.ndarray
    y_validation: np.ndarray
    y_evaluation: np.ndarray

    # 保存原始元数据，便于最终预测 CSV 保留 BUILDINGID、FLOOR 等字段。
    validation_metadata: pd.DataFrame
    evaluation_metadata: pd.DataFrame

    feature_names: list[str]
    scaler: MinMaxScaler
    summary: dict[str, Any]


def _stratify_labels(frame: pd.DataFrame) -> np.ndarray | None:
    """构造 BUILDINGID + FLOOR 分层标签。

    如果某些类别样本过少，train_test_split 无法分层，
    此时返回 None，让 sklearn 使用普通随机划分。
    """

    required = {"BUILDINGID", "FLOOR"}
    if not required.issubset(frame.columns):
        return None

    labels = (
        frame[["BUILDINGID", "FLOOR"]]
        .astype(int)
        .astype(str)
        .agg("_".join, axis=1)
        .to_numpy()
    )

    _, counts = np.unique(labels, return_counts=True)
    return labels if len(counts) > 1 and counts.min() >= 2 else None


def _load_frame(
    path: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """读取 CSV；冒烟测试时可以只读取前 max_rows 行。"""

    return pd.read_csv(path, nrows=max_rows)


def prepare_data(
    config: DataConfig,
    seed: int,
    smoke_train_rows: int | None = None,
    smoke_eval_rows: int | None = None,
) -> PreparedData:
    """完成数据读取、筛选、划分和缩放。"""

    logger = logging.getLogger("xiaoma")

    train_df = _load_frame(config.train_csv, smoke_train_rows)
    eval_df = _load_frame(config.eval_csv, smoke_eval_rows)

    logger.info("Loaded training data: %s", train_df.shape)
    logger.info("Loaded evaluation data: %s", eval_df.shape)

    # UJIIndoorLoc 一共有 3 个建筑，编号为 0、1、2。
    if config.building_id not in {-1, 0, 1, 2}:
        raise ValueError("building_id must be one of -1, 0, 1, 2.")

    if config.building_id != -1:
        train_df = train_df.loc[
            train_df["BUILDINGID"] == config.building_id
        ].reset_index(drop=True)

        eval_df = eval_df.loc[
            eval_df["BUILDINGID"] == config.building_id
        ].reset_index(drop=True)

    if train_df.empty or eval_df.empty:
        raise ValueError(
            "No rows remain after applying the building filter."
        )

    # 自动识别所有 WAP 特征，避免在代码中手写 WAP001...WAP520。
    wap_columns = [
        column
        for column in train_df.columns
        if str(column).upper().startswith(
            config.wap_prefix.upper()
        )
    ]

    if not wap_columns:
        raise ValueError("No WAP feature columns were found.")

    # 将数据集中的缺失标记 100 转换成弱信号值 -105。
    train_features = (
        train_df[wap_columns]
        .replace(
            config.raw_missing_value,
            config.filled_missing_value,
        )
        .fillna(config.filled_missing_value)
    )

    eval_features = (
        eval_df[wap_columns]
        .replace(
            config.raw_missing_value,
            config.filled_missing_value,
        )
        .fillna(config.filled_missing_value)
    )

    if config.enable_feature_filtering:
        # 按列计算每个 WAP 在全部训练样本中的缺失比例。
        missing_ratio = (
                train_features == config.filled_missing_value
        ).mean(axis=0)

        feature_names = missing_ratio[
            missing_ratio < config.feature_missing_threshold
            ].index.tolist()

        if not feature_names:
            raise ValueError(
                "Feature filtering removed every WAP column."
            )
    else:
        # 论文复现模式：完整保留 WAP001-WAP520。
        feature_names = wap_columns.copy()

    train_features = train_features[feature_names]
    eval_features = eval_features[feature_names]

    # 目标是二维坐标：经度和纬度。
    targets = train_df[
        ["LONGITUDE", "LATITUDE"]
    ].to_numpy(dtype=np.float32)

    eval_targets = eval_df[
        ["LONGITUDE", "LATITUDE"]
    ].to_numpy(dtype=np.float32)

    # 官方 validationData.csv 用作最终评估。
    # 同时从 trainingData.csv 内部划分一份验证集，用于早停。
    indices = np.arange(len(train_df))
    train_idx, validation_idx = train_test_split(
        indices,
        test_size=config.validation_size,
        random_state=seed,
        shuffle=True,
        stratify=_stratify_labels(train_df),
    )

    # WAP RSSI 特征使用训练集拟合 MinMaxScaler。
    # 验证集和官方评估集只做 transform，避免数据泄漏。
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(
        train_features.iloc[train_idx].to_numpy()
    ).astype(np.float32)

    X_validation = scaler.transform(
        train_features.iloc[validation_idx].to_numpy()
    ).astype(np.float32)

    X_evaluation = scaler.transform(
        eval_features.to_numpy()
    ).astype(np.float32)

    summary = {
        "training_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "evaluation_rows": int(len(eval_df)),
        "features_before_filtering": int(len(wap_columns)),
        "features_after_filtering": int(len(feature_names)),
        "feature_missing_threshold": (
            config.feature_missing_threshold
        ),
        "building_id": config.building_id,
    }

    logger.info("Prepared data summary: %s", summary)

    return PreparedData(
        X_train=X_train,
        X_validation=X_validation,
        X_evaluation=X_evaluation,
        y_train=targets[train_idx],
        y_validation=targets[validation_idx],
        y_evaluation=eval_targets,
        validation_metadata=(
            train_df.iloc[validation_idx].reset_index(drop=True)
        ),
        evaluation_metadata=eval_df.reset_index(drop=True),
        feature_names=feature_names,
        scaler=scaler,
        summary=summary,
    )


def train_autoencoder(
    model: WiFiTransformerAutoencoder,
    data: PreparedData,
    config: TrainingConfig,
    device: torch.device,
    target_scaler: StandardScaler | None = None,
) -> pd.DataFrame:
    """训练 Transformer。

    两种训练模式：

    1. regressor == "svr"
       训练 Transformer 自编码器，按照内部验证集的重建
       validation_loss 保存最佳模型。

    2. regressor == "mlp"
       Transformer + MLP 端到端预测坐标，按照内部验证集的
       distance_mean 保存最佳模型；mean 基本相同时比较 median。
    """

    logger = logging.getLogger("xiaoma")
    pin_memory = device.type == "cuda"

    joint = config.regressor == "mlp"

    # MLP 模式的坐标经过 StandardScaler 标准化。
    # 每个 epoch 计算米制距离时必须使用它恢复原始坐标。
    if joint and target_scaler is None:
        raise ValueError(
            "target_scaler is required when regressor='mlp'."
        )

    # ------------------------------------------------------------------
    # 构造训练集与验证集
    # ------------------------------------------------------------------
    if joint:
        logger.info(
            "Joint training mode: Transformer + MLP regression "
            "(checkpoint selected by validation distance)."
        )

        train_dataset = TensorDataset(
            torch.from_numpy(data.X_train),
            torch.from_numpy(data.y_train),
        )
        validation_dataset = TensorDataset(
            torch.from_numpy(data.X_validation),
            torch.from_numpy(data.y_validation),
        )
    else:
        logger.info(
            "Autoencoder training mode "
            "(checkpoint selected by reconstruction loss)."
        )

        train_dataset = TensorDataset(
            torch.from_numpy(data.X_train)
        )
        validation_dataset = TensorDataset(
            torch.from_numpy(data.X_validation)
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    # SVR 模式：重建输入特征的 MSE。
    # MLP 模式：标准化坐标的 MSE。
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # MLP 模式下 scheduler.step() 使用 distance_mean。
    # SVR 模式下 scheduler.step() 使用 validation_loss。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(
            1,
            config.early_stopping_patience // 2,
        ),
    )

    # SVR/自编码器模式使用。
    best_loss = float("inf")

    # MLP 模式使用。
    best_distance_mean = float("inf")
    best_distance_median = float("inf")

    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    patience = 0
    history: list[dict[str, float | int]] = []

    # ------------------------------------------------------------------
    # Epoch 循环
    # ------------------------------------------------------------------
    for epoch in range(1, config.epochs + 1):
        # ==============================================================
        # 训练阶段
        # ==============================================================
        model.train()

        train_loss_sum = 0.0
        train_items = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            if joint:
                batch_x, batch_y = batch

                batch_x = batch_x.to(
                    device,
                    non_blocking=True,
                )

                batch_y = batch_y.to(
                    device,
                    non_blocking=True,
                )

                # MLP 模式直接预测标准化后的二维坐标。
                coordinate_prediction = model(
                    batch_x,
                    joint=True,
                )

                loss = criterion(
                    coordinate_prediction,
                    batch_y,
                )
            else:
                (batch_x,) = batch

                batch_x = batch_x.to(
                    device,
                    non_blocking=True,
                )

                # SVR 模式先训练自编码器。
                reconstruction = model(batch_x)

                loss = criterion(
                    reconstruction,
                    batch_x,
                )

            loss.backward()

            # 防止 Transformer 出现梯度爆炸。
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = len(batch_x)

            train_loss_sum += (
                float(loss.item()) * batch_size
            )

            train_items += batch_size

        train_loss = (
            train_loss_sum / max(train_items, 1)
        )

        # ==============================================================
        # 验证阶段
        # ==============================================================
        model.eval()

        validation_loss_sum = 0.0
        validation_items = 0

        # 只有 MLP 模式需要收集坐标预测。
        validation_prediction_chunks: list[np.ndarray] = []
        validation_target_chunks: list[np.ndarray] = []

        with torch.inference_mode():
            for batch in validation_loader:
                if joint:
                    batch_x, batch_y = batch

                    batch_x = batch_x.to(
                        device,
                        non_blocking=True,
                    )

                    batch_y = batch_y.to(
                        device,
                        non_blocking=True,
                    )

                    coordinate_prediction = model(
                        batch_x,
                        joint=True,
                    )

                    loss = criterion(
                        coordinate_prediction,
                        batch_y,
                    )

                    # 保存标准化坐标，验证循环结束后统一恢复原始坐标。
                    validation_prediction_chunks.append(
                        coordinate_prediction
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    validation_target_chunks.append(
                        batch_y
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    (batch_x,) = batch

                    batch_x = batch_x.to(
                        device,
                        non_blocking=True,
                    )

                    reconstruction = model(batch_x)

                    loss = criterion(
                        reconstruction,
                        batch_x,
                    )

                batch_size = len(batch_x)

                validation_loss_sum += (
                    float(loss.item()) * batch_size
                )

                validation_items += batch_size

        validation_loss = (
            validation_loss_sum
            / max(validation_items, 1)
        )

        # 默认设置为 NaN。
        # SVR 模式不会在这里产生定位距离。
        validation_distance_mean = float("nan")
        validation_distance_median = float("nan")

        # ==============================================================
        # MLP 模式：计算当前 epoch 的真实定位距离
        # ==============================================================
        if joint:
            if not validation_prediction_chunks:
                raise RuntimeError(
                    "No validation predictions were produced."
                )

            validation_prediction_scaled = np.concatenate(
                validation_prediction_chunks,
                axis=0,
            )

            validation_target_scaled = np.concatenate(
                validation_target_chunks,
                axis=0,
            )

            # 标准化坐标恢复成原始 LONGITUDE、LATITUDE。
            validation_prediction_original = (
                target_scaler.inverse_transform(
                    validation_prediction_scaled
                )
            )

            validation_target_original = (
                target_scaler.inverse_transform(
                    validation_target_scaled
                )
            )

            validation_metrics = distance_metrics(
                validation_target_original,
                validation_prediction_original,
            )

            validation_distance_mean = float(
                validation_metrics["distance_mean"]
            )

            validation_distance_median = float(
                validation_metrics["distance_median"]
            )

            # MLP 模式：学习率根据真实定位距离调整。
            scheduler.step(validation_distance_mean)
        else:
            # 自编码器模式：学习率根据重建损失调整。
            scheduler.step(validation_loss)

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        # ==============================================================
        # 保存训练历史
        # ==============================================================
        history_row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": current_lr,
        }

        if joint:
            history_row["validation_distance_mean"] = (
                validation_distance_mean
            )

            history_row["validation_distance_median"] = (
                validation_distance_median
            )

        history.append(history_row)

        # ==============================================================
        # 打印当前 epoch 日志
        # ==============================================================
        if joint:
            logger.info(
                "Epoch %03d/%03d | "
                "train_loss=%.8f | "
                "validation_loss=%.8f | "
                "distance_mean=%.6f | "
                "distance_median=%.6f | "
                "lr=%.3e",
                epoch,
                config.epochs,
                train_loss,
                validation_loss,
                validation_distance_mean,
                validation_distance_median,
                current_lr,
            )
        else:
            logger.info(
                "Epoch %03d/%03d | "
                "train_loss=%.8f | "
                "validation_loss=%.8f | "
                "lr=%.3e",
                epoch,
                config.epochs,
                train_loss,
                validation_loss,
                current_lr,
            )

        # ==============================================================
        # 判断当前 epoch 是否为最佳模型
        # ==============================================================
        if joint:
            # distance_mean 是主要保存指标。
            mean_improved = (
                validation_distance_mean
                < best_distance_mean - config.min_delta
            )

            # mean 的差距不超过 min_delta 时，认为两者基本相同。
            mean_tied = (
                abs(
                    validation_distance_mean
                    - best_distance_mean
                )
                <= config.min_delta
            )

            # mean 基本相同时，比较 median。
            median_improved = (
                validation_distance_median
                < best_distance_median
            )

            improved = (
                mean_improved
                or (
                    mean_tied
                    and median_improved
                )
            )

            if improved:
                best_distance_mean = (
                    validation_distance_mean
                )

                best_distance_median = (
                    validation_distance_median
                )

                best_epoch = epoch

                best_state = copy.deepcopy(
                    model.state_dict()
                )

                patience = 0

                logger.info(
                    "New best MLP checkpoint | "
                    "epoch=%d | "
                    "distance_mean=%.6f | "
                    "distance_median=%.6f",
                    best_epoch,
                    best_distance_mean,
                    best_distance_median,
                )
            else:
                patience += 1
        else:
            # SVR/自编码器模式保持原来的保存逻辑。
            improved = (
                validation_loss
                < best_loss - config.min_delta
            )

            if improved:
                best_loss = validation_loss
                best_epoch = epoch

                best_state = copy.deepcopy(
                    model.state_dict()
                )

                patience = 0

                logger.info(
                    "New best autoencoder checkpoint | "
                    "epoch=%d | "
                    "validation_loss=%.8f",
                    best_epoch,
                    best_loss,
                )
            else:
                patience += 1

        # ==============================================================
        # Early stopping
        # ==============================================================
        if patience >= config.early_stopping_patience:
            if joint:
                logger.info(
                    "Early stopping at epoch %d | "
                    "best_epoch=%d | "
                    "best_distance_mean=%.6f | "
                    "best_distance_median=%.6f",
                    epoch,
                    best_epoch,
                    best_distance_mean,
                    best_distance_median,
                )
            else:
                logger.info(
                    "Early stopping at epoch %d | "
                    "best_epoch=%d | "
                    "best_validation_loss=%.8f",
                    epoch,
                    best_epoch,
                    best_loss,
                )

            break

    # ------------------------------------------------------------------
    # 恢复最佳 epoch 的权重
    # ------------------------------------------------------------------
    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid checkpoint."
        )

    model.load_state_dict(best_state)

    if joint:
        logger.info(
            "Restored best MLP checkpoint | "
            "epoch=%d | "
            "distance_mean=%.6f | "
            "distance_median=%.6f",
            best_epoch,
            best_distance_mean,
            best_distance_median,
        )
    else:
        logger.info(
            "Restored best autoencoder checkpoint | "
            "epoch=%d | "
            "validation_loss=%.8f",
            best_epoch,
            best_loss,
        )

    return pd.DataFrame(history)



@torch.inference_mode()
def extract_latent(
    model: WiFiTransformerAutoencoder,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """批量提取 Transformer latent 特征。"""

    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    chunks: list[np.ndarray] = []

    for (batch,) in loader:
        latent = model.encode(batch.to(device))
        chunks.append(latent.cpu().numpy())

    return np.concatenate(
        chunks,
        axis=0,
    ).astype(np.float32)


@torch.inference_mode()
def predict_regression(
    model: WiFiTransformerAutoencoder,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """使用模型回归头批量预测坐标。"""

    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    chunks: list[np.ndarray] = []

    for (batch,) in loader:
        coords = model(
            batch.to(device), joint=True
        )
        chunks.append(coords.cpu().numpy())

    return np.concatenate(
        chunks,
        axis=0,
    ).astype(np.float32)


def fit_svr(
    latent_features: np.ndarray,
    targets: np.ndarray,
    config: TrainingConfig,
) -> TransformedTargetRegressor:
    """使用 sklearn 官方组合器拟合经纬度 SVR。"""

    logger = logging.getLogger("xiaoma")
    X_fit = latent_features
    y_fit = targets

    # SVR 的训练复杂度随样本量增长较快。
    # 用户明确设置上限时，使用固定随机种子进行无放回采样。
    if (
        config.svr_max_train_samples > 0
        and len(X_fit) > config.svr_max_train_samples
    ):
        rng = np.random.default_rng(config.seed)
        selected = rng.choice(
            len(X_fit),
            size=config.svr_max_train_samples,
            replace=False,
        )

        X_fit = X_fit[selected]
        y_fit = y_fit[selected]

        logger.info(
            "SVR training was sampled down to %d rows.",
            len(X_fit),
        )

    # MultiOutputRegressor 自动为经度和纬度各训练一个 SVR。
    # TransformedTargetRegressor 自动标准化目标值，并在预测后恢复原尺度。
    regressor = TransformedTargetRegressor(
        regressor=MultiOutputRegressor(
            SVR(
                kernel="rbf",
                C=config.svr_c,
                epsilon=config.svr_epsilon,
                gamma=config.svr_gamma,
            )
        ),
        transformer=StandardScaler(),
    )

    regressor.fit(X_fit, y_fit)
    return regressor


def _prediction_frame(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """将预测坐标和误差附加到原始数据表。"""

    distances = np.linalg.norm(
        y_true - y_pred,
        axis=1,
    )

    output = metadata.copy()
    output["PRED_LONGITUDE"] = y_pred[:, 0]
    output["PRED_LATITUDE"] = y_pred[:, 1]
    output["ERROR_DISTANCE"] = distances

    return output


def run_training(
    config: AppConfig,
    branch: str | None,
    smoke_test: bool,
    smoke_train_rows: int,
    smoke_eval_rows: int,
) -> Path:
    """执行完整训练任务并返回模型产物目录。"""

    paths = create_run_paths(
        model_root=config.model_root,
        project_root=PROJECT_ROOT,
        branch=branch,
    )

    logger = setup_logging(paths.log_file)

    logger.info("Branch: %s", paths.branch)
    logger.info("Artifacts: %s", paths.artifact_dir)
    logger.info("UTF-8 log: %s", paths.log_file)

    set_seed(config.training.seed)
    device = select_device(config.training.device)

    logger.info("Device: %s", device)

    data = prepare_data(
        config.data,
        seed=config.training.seed,
        smoke_train_rows=(
            smoke_train_rows if smoke_test else None
        ),
        smoke_eval_rows=(
            smoke_eval_rows if smoke_test else None
        ),
    )

    # 保存原始坐标副本，用于最终指标计算和 CSV 输出
    #（MLP 模式下 y 会被标准化，但指标和 CSV 需要原始坐标）
    y_validation_orig = data.y_validation.copy()
    y_evaluation_orig = data.y_evaluation.copy()

    if config.training.regressor == "mlp":
        # MLP 模式：对坐标做标准化，让 MSE 落在合理量级
        target_scaler = StandardScaler()
        data.y_train = (
            target_scaler.fit_transform(data.y_train).astype(np.float32)
        )
        data.y_validation = (
            target_scaler.transform(data.y_validation).astype(np.float32)
        )
        data.y_evaluation = (
            target_scaler.transform(data.y_evaluation).astype(np.float32)
        )

    model = WiFiTransformerAutoencoder(
        input_dim=data.X_train.shape[1],
        config=config.model,
    ).to(device)

    logger.info(
        "Transformer parameters: %d",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    history = train_autoencoder(
        model,
        data,
        config.training,
        device,
        target_scaler=(
            target_scaler
            if config.training.regressor == "mlp"
            else None
        ),
    )

    if config.training.regressor == "mlp":
        logger.info(
            "Predicting coordinates via model regression head."
        )

        # MLP 模式不需要显式提取 latent，回归头直接在 forward 中产出坐标。
        validation_prediction = predict_regression(
            model,
            data.X_validation,
            device,
            config.training.feature_batch_size,
        )
        evaluation_prediction = predict_regression(
            model,
            data.X_evaluation,
            device,
            config.training.feature_batch_size,
        )

        # 将标准化后的坐标预测还原为原始坐标量级
        validation_prediction = (
            target_scaler.inverse_transform(
                validation_prediction
            )
        )
        evaluation_prediction = (
            target_scaler.inverse_transform(
                evaluation_prediction
            )
        )

    else:
        # 自编码器训练结束后，分别提取三个数据集的 latent 特征。
        train_latent = extract_latent(
            model,
            data.X_train,
            device,
            config.training.feature_batch_size,
        )

        validation_latent = extract_latent(
            model,
            data.X_validation,
            device,
            config.training.feature_batch_size,
        )

        evaluation_latent = extract_latent(
            model,
            data.X_evaluation,
            device,
            config.training.feature_batch_size,
        )

        logger.info(
            "Fitting MultiOutputRegressor(SVR) "
            "with target scaling."
        )

        regressor = fit_svr(
            train_latent,
            data.y_train,
            config.training,
        )

        validation_prediction = regressor.predict(
            validation_latent
        )
        evaluation_prediction = regressor.predict(
            evaluation_latent
        )

    metrics = {
        "validation": distance_metrics(
            y_validation_orig,
            validation_prediction,
        ),
        "evaluation": distance_metrics(
            y_evaluation_orig,
            evaluation_prediction,
        ),
        "preprocessing": data.summary,
    }

    logger.info(
        "Validation metrics: %s",
        metrics["validation"],
    )
    logger.info(
        "Evaluation metrics: %s",
        metrics["evaluation"],
    )

    # 保存 Transformer 配置和权重。
    torch.save(
        model.checkpoint_payload(),
        paths.artifact_dir / "transformer.pt",
    )

    # pipeline.joblib 保存预测所需的非神经网络对象。
    pipeline: dict[str, object] = {
        "feature_names": data.feature_names,
        "feature_scaler": data.scaler,
        "raw_missing_value": (
            config.data.raw_missing_value
        ),
        "filled_missing_value": (
            config.data.filled_missing_value
        ),
        "transformer_file": "transformer.pt",
    }

    if config.training.regressor == "mlp":
        pipeline["regressor_type"] = "mlp"
        pipeline["target_scaler"] = target_scaler
    else:
        pipeline["regressor_type"] = "svr"
        pipeline["regressor"] = regressor

    joblib.dump(
        pipeline,
        paths.artifact_dir / "pipeline.joblib",
    )

    # 所有文本文件都显式使用 UTF-8。
    history.to_csv(
        paths.artifact_dir / "training_history.csv",
        index=False,
        encoding="utf-8",
    )

    _prediction_frame(
        data.validation_metadata,
        y_validation_orig,
        validation_prediction,
    ).to_csv(
        paths.artifact_dir / "predictions_validation.csv",
        index=False,
        encoding="utf-8",
    )

    _prediction_frame(
        data.evaluation_metadata,
        y_evaluation_orig,
        evaluation_prediction,
    ).to_csv(
        paths.artifact_dir / "predictions_evaluation.csv",
        index=False,
        encoding="utf-8",
    )

    save_json(
        config.to_dict(),
        paths.artifact_dir / "config.json",
    )
    save_json(
        metrics,
        paths.artifact_dir / "metrics.json",
    )

    logger.info("Training completed successfully.")
    return paths.artifact_dir


def parse_args() -> argparse.Namespace:
    """定义训练命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Train the refactored UJIIndoorLoc "
            "Transformer + SVR/MLP model."
        )
    )

    parser.add_argument(
        "--regressor",
        type=str,
        default=DEFAULT_CONFIG.training.regressor,
        choices={"svr", "mlp"},
        help="回归器类型：svr（两阶段 SVR）或 mlp（端到端 MLP 回归头）。",
    )
    parser.add_argument(
        "--regression-loss-weight",
        type=float,
        default=(
            DEFAULT_CONFIG.training.regression_loss_weight
        ),
        help="MLP 联合训练时回归损失的权重。",
    )
    parser.add_argument(
        "--mlp-hidden-sizes",
        type=str,
        default=DEFAULT_CONFIG.model.mlp_hidden_sizes,
        help="MLP 回归头隐藏层大小，逗号分隔，如 128,64。",
    )

    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_CONFIG.data.train_csv,
        help="训练集 CSV 文件路径（UJIIndoorLoc trainingData.csv）。",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=DEFAULT_CONFIG.data.eval_csv,
        help="官方评估集 CSV 文件路径（validationData.csv）。",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_CONFIG.model_root,
        help="模型产物、预测结果和日志的输出根目录。",
    )
    parser.add_argument(
        "--building-id",
        type=int,
        default=DEFAULT_CONFIG.data.building_id,
        help="建筑编号筛选：-1 使用全部建筑，0/1/2 只使用指定建筑。",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="运行分支名，用于组织产物子目录（None 则自动生成时间戳）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_CONFIG.training.device,
        help="PyTorch 设备，auto 优先使用 CUDA，不可用时回退到 CPU。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG.training.epochs,
        help="Transformer 自编码器最大训练轮数。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG.training.batch_size,
        help="自编码器训练阶段的 mini-batch 大小。",
    )
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=(
            DEFAULT_CONFIG.training.feature_batch_size
        ),
        help="提取 latent 特征时的推理 batch 大小。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_CONFIG.training.learning_rate,
        help="AdamW 优化器的初始学习率。",
    )
    parser.add_argument(
        "--svr-c",
        type=float,
        default=DEFAULT_CONFIG.training.svr_c,
        help="SVR 正则化参数 C，越大则对训练误差的惩罚越重。",
    )
    parser.add_argument(
        "--svr-epsilon",
        type=float,
        default=DEFAULT_CONFIG.training.svr_epsilon,
        help="SVR 的 epsilon 参数，控制 epsilon-insensitive 管道宽度。",
    )
    parser.add_argument(
        "--svr-max-train-samples",
        type=int,
        default=(
            DEFAULT_CONFIG.training.svr_max_train_samples
        ),
        help="拟合 SVR 的最大样本数，0 表示使用全部样本（数据量大时可限制以加速）。",
    )

    # 冒烟测试只读取少量数据，用来快速验证代码能否完整运行。
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

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    """用命令行参数覆盖默认 dataclass 配置。"""

    data_config = replace(
        DEFAULT_CONFIG.data,
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        building_id=args.building_id,
    )

    training_config = replace(
        DEFAULT_CONFIG.training,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        feature_batch_size=args.feature_batch_size,
        learning_rate=args.learning_rate,
        regressor=args.regressor,
        regression_loss_weight=args.regression_loss_weight,
        svr_c=args.svr_c,
        svr_epsilon=args.svr_epsilon,
        svr_max_train_samples=(
            args.svr_max_train_samples
        ),
    )

    model_config = replace(
        DEFAULT_CONFIG.model,
        mlp_hidden_sizes=args.mlp_hidden_sizes,
    )

    return replace(
        DEFAULT_CONFIG,
        data=data_config,
        model=model_config,
        training=training_config,
        model_root=args.model_root,
    )


def main() -> None:
    """训练命令行主函数。"""

    args = parse_args()
    config = build_config(args)

    try:
        artifact_dir = run_training(
            config=config,
            branch=args.branch,
            smoke_test=args.smoke_test,
            smoke_train_rows=args.smoke_train_rows,
            smoke_eval_rows=args.smoke_eval_rows,
        )
    except Exception:
        logging.getLogger("xiaoma").exception(
            "Training failed."
        )
        raise

    print(f"Model artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
