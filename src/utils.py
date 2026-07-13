import json
import logging
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DATASET_URL,
    DATA_DIR,
    LOG_DIR,
    MODEL_DIR,
    SEED,
    TARGET_COLUMNS,
    TEST_CSV,
    TRAIN_CSV,
    WAP_DIM,
)


def ensure_directories() -> None:
    """创建项目要求的目录。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> None:
    """缺少数据时，从 UCI 官方地址下载并提取数据。"""
    ensure_directories()
    if TRAIN_CSV.exists() and TEST_CSV.exists():
        return

    zip_path = DATA_DIR / "ujiindoorloc.zip"
    extract_dir = DATA_DIR / "_extract"
    print("未检测到 UJIIndoorLoc 数据，正在从 UCI 下载……")
    try:
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)

        train_source = next(extract_dir.rglob("trainingData.csv"))
        test_source = next(extract_dir.rglob("validationData.csv"))
        shutil.copy2(train_source, TRAIN_CSV)
        shutil.copy2(test_source, TEST_CSV)
    except Exception as exc:
        raise RuntimeError(
            "数据自动下载失败。请手动将 trainingData.csv 和 validationData.csv 放入 data 目录。"
        ) from exc
    finally:
        if zip_path.exists():
            zip_path.unlink()
        if extract_dir.exists():
            shutil.rmtree(extract_dir)


def set_seed(seed: int = SEED) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_logger(log_path: Path) -> logging.Logger:
    """同时输出到终端和日志文件。"""
    logger = logging.getLogger("uji_transformer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def get_wap_columns() -> list[str]:
    """返回 WAP001 到 WAP520。"""
    return [f"WAP{i:03d}" for i in range(1, WAP_DIM + 1)]


def load_data(
    building_id: int | None = None,
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取官方训练集和 validationData，并按 BuildingID 同步过滤。"""
    download_dataset()
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    if building_id is not None:
        train_df = train_df[train_df["BUILDINGID"] == building_id]
        test_df = test_df[test_df["BUILDINGID"] == building_id]

    if train_df.empty or test_df.empty:
        raise ValueError("指定 BuildingID 后没有可用数据")

    if smoke_test:
        train_count = min(8, len(train_df))
        test_count = min(4, len(test_df))
        train_df = train_df.sample(train_count, random_state=SEED)
        test_df = test_df.sample(test_count, random_state=SEED)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _prepare_rssi(df: pd.DataFrame) -> np.ndarray:
    """将未检测到信号的 100 替换为 -105 dB。"""
    wap_columns = get_wap_columns()
    missing_columns = [column for column in wap_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"数据缺少 WAP 列：{missing_columns[:3]}")

    values = df[wap_columns].replace(100, -105).to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("RSSI 数据包含 NaN 或无穷值")
    return values


def fit_preprocessors(
    train_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, StandardScaler]:
    """缩放器只在训练集上拟合，避免验证集信息泄漏。"""
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = StandardScaler()

    x_train = feature_scaler.fit_transform(_prepare_rssi(train_df)).astype(np.float32)
    y_train = target_scaler.fit_transform(
        train_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    return x_train, y_train, feature_scaler, target_scaler


def transform_dataframe(
    df: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    target_scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """用训练集缩放器转换验证数据或待预测数据。"""
    x_data = feature_scaler.transform(_prepare_rssi(df)).astype(np.float32)
    y_data = None
    if target_scaler is not None and all(column in df.columns for column in TARGET_COLUMNS):
        y_data = target_scaler.transform(
            df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        ).astype(np.float32)
    return x_data, y_data


def save_preprocessors(
    feature_scaler: MinMaxScaler,
    target_scaler: StandardScaler,
    path: Path,
) -> None:
    joblib.dump(
        {"feature_scaler": feature_scaler, "target_scaler": target_scaler},
        path,
    )


def load_preprocessors(path: Path) -> tuple[MinMaxScaler, StandardScaler]:
    preprocessors = joblib.load(path)
    return preprocessors["feature_scaler"], preprocessors["target_scaler"]


def make_autoencoder_loaders(
    x_data: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """从 trainingData 内部划分自编码器训练集和早停验证集。"""
    val_ratio = 0.2 if len(x_data) < 200 else 0.1
    x_train, x_val = train_test_split(
        x_data,
        test_size=val_ratio,
        random_state=SEED,
    )
    train_dataset = TensorDataset(torch.from_numpy(x_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def encode_features(
    model: torch.nn.Module,
    x_data: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """批量提取 CLS 编码特征。"""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_data)),
        batch_size=batch_size,
        shuffle=False,
    )
    encoded_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            encoded = model.encode(batch_x.to(device))
            encoded_batches.append(encoded.cpu().numpy())
    return np.concatenate(encoded_batches, axis=0)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """UJIIndoorLoc 坐标本身采用米制平面坐标，可直接计算欧氏距离。"""
    distances = np.linalg.norm(y_pred - y_true, axis=1)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_position_error_m": float(np.mean(distances)),
        "median_position_error_m": float(np.median(distances)),
        "std_position_error_m": float(np.std(distances)),
    }


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
