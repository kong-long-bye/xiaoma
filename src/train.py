import argparse
import copy

import joblib
import numpy as np
import torch
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DEFAULT_CONFIG,
    DEVICE,
    LOG_DIR,
    MODEL_DIR,
    SEED,
    SMOKE_CONFIG,
    WAP_DIM,
)
from model import CoordinateMLP, TransformerAutoencoder
from utils import (
    build_logger,
    calculate_metrics,
    encode_features,
    ensure_directories,
    fit_preprocessors,
    load_data,
    make_autoencoder_loaders,
    save_json,
    save_preprocessors,
    set_seed,
    transform_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Transformer 自编码器 + SVR/MLP")
    parser.add_argument("--regressor", choices=["svr", "mlp"], default="svr")
    parser.add_argument("--building-id", choices=["all", "0", "1", "2"], default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def train_autoencoder(
    model: TransformerAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    logger,
) -> list[dict[str, float]]:
    """使用 MSE、AdamW 和早停训练自编码器。"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                reconstructed, _ = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                val_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        logger.info(
            "自编码器 Epoch %d/%d - train_loss=%.6f - val_loss=%.6f",
            epoch,
            config["epochs"],
            train_loss,
            val_loss,
        )

        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= config["patience"]:
                logger.info("触发早停，停止自编码器训练")
                break

    model.load_state_dict(best_state)
    return history


def _split_mlp_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(SEED)
    indices = generator.permutation(len(x_data))
    val_size = max(1, int(len(indices) * 0.1))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return (
        x_data[train_indices],
        x_data[val_indices],
        y_data[train_indices],
        y_data[val_indices],
    )


def train_mlp_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    config: dict,
    device: torch.device,
    logger,
) -> CoordinateMLP:
    """训练可选的 MLP 坐标回归器。"""
    x_fit, x_val, y_fit, y_val = _split_mlp_data(x_train, y_train)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    model = CoordinateMLP(input_dim, config["mlp_hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["mlp_learning_rate"])
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improvement = 0

    for epoch in range(1, config["mlp_epochs"] + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_loss += (
                    criterion(model(batch_x), batch_y).item() * batch_x.size(0)
                )
        val_loss /= len(val_loader.dataset)

        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                "MLP Epoch %d/%d - val_loss=%.6f",
                epoch,
                config["mlp_epochs"],
                val_loss,
            )

        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= config["mlp_patience"]:
                logger.info("触发早停，停止 MLP 训练")
                break

    model.load_state_dict(best_state)
    return model


def main() -> None:
    args = parse_args()
    ensure_directories()
    set_seed()

    config = dict(SMOKE_CONFIG if args.smoke_test else DEFAULT_CONFIG)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    building_id = None if args.building_id == "all" else int(args.building_id)
    run_name = f"building_{args.building_id}_{args.regressor}"
    logger = build_logger(LOG_DIR / f"{run_name}.log")
    device = torch.device(DEVICE)

    logger.info("设备：%s", device)
    logger.info("训练建筑：%s", "全部" if building_id is None else building_id)
    train_df, test_df = load_data(
        building_id=building_id,
        smoke_test=args.smoke_test,
    )
    logger.info(
        "训练样本：%d，validationData 测试样本：%d",
        len(train_df),
        len(test_df),
    )

    x_train, y_train, feature_scaler, target_scaler = fit_preprocessors(train_df)
    x_test, y_test = transform_dataframe(test_df, feature_scaler, target_scaler)
    if y_test is None:
        raise ValueError("测试数据缺少 LONGITUDE 和 LATITUDE")

    save_preprocessors(
        feature_scaler,
        target_scaler,
        MODEL_DIR / "preprocessors.joblib",
    )
    train_loader, val_loader = make_autoencoder_loaders(
        x_train,
        config["batch_size"],
    )

    model_params = {
        "input_dim": WAP_DIM,
        "d_model": config["d_model"],
        "nhead": config["nhead"],
        "num_layers": config["num_layers"],
        "dim_feedforward": config["dim_feedforward"],
        "dropout": config["dropout"],
    }
    autoencoder = TransformerAutoencoder(**model_params).to(device)
    history = train_autoencoder(
        autoencoder,
        train_loader,
        val_loader,
        config,
        device,
        logger,
    )
    torch.save(
        {"state_dict": autoencoder.state_dict(), "model_params": model_params},
        MODEL_DIR / "transformer_autoencoder.pt",
    )
    save_json({"history": history}, LOG_DIR / "loss_history.json")

    train_features = encode_features(
        autoencoder,
        x_train,
        device,
        config["batch_size"],
    )
    test_features = encode_features(
        autoencoder,
        x_test,
        device,
        config["batch_size"],
    )
    logger.info("SVR/MLP 输入特征维度：%d", train_features.shape[1])

    if args.regressor == "svr":
        regressor = MultiOutputRegressor(
            SVR(
                kernel="rbf",
                C=config["svr_c"],
                epsilon=config["svr_epsilon"],
                gamma=config["svr_gamma"],
            )
        )
        regressor.fit(train_features, y_train)
        pred_scaled = regressor.predict(test_features)
        joblib.dump(regressor, MODEL_DIR / "svr_regressor.joblib")
    else:
        regressor = train_mlp_regressor(
            train_features,
            y_train,
            model_params["d_model"],
            config,
            device,
            logger,
        )
        regressor.eval()
        with torch.no_grad():
            pred_scaled = regressor(
                torch.from_numpy(test_features).to(device)
            ).cpu().numpy()
        torch.save(regressor.state_dict(), MODEL_DIR / "mlp_regressor.pt")

    y_true = target_scaler.inverse_transform(y_test)
    y_pred = target_scaler.inverse_transform(pred_scaled)
    metrics = calculate_metrics(y_true, y_pred)
    save_json(metrics, LOG_DIR / "metrics.json")

    run_config = {
        "regressor": args.regressor,
        "building_id": building_id,
        "model_params": model_params,
        "training_config": config,
    }
    save_json(run_config, MODEL_DIR / "run_config.json")

    logger.info("测试指标：%s", metrics)
    print("训练完成，模型与日志已保存到 model 目录。")


if __name__ == "__main__":
    main()
