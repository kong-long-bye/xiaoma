import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from config import DEVICE, LOG_DIR, MODEL_DIR, TEST_CSV
from model import CoordinateMLP, TransformerAutoencoder
from utils import (
    calculate_metrics,
    download_dataset,
    encode_features,
    load_preprocessors,
    transform_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用已训练模型预测 validationData")
    parser.add_argument("--input-csv", type=Path, default=TEST_CSV)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=LOG_DIR / "predictions.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_csv == TEST_CSV and not TEST_CSV.exists():
        download_dataset()

    with (MODEL_DIR / "run_config.json").open("r", encoding="utf-8") as file:
        run_config = json.load(file)

    data = pd.read_csv(args.input_csv)
    building_id = run_config["building_id"]

    # 模型若只在 Building 0 上训练，预测时也只能评估 validationData 中的 Building 0。
    if building_id is not None:
        if "BUILDINGID" not in data.columns:
            raise ValueError(
                "当前模型是单建筑模型，但输入 CSV 缺少 BUILDINGID，无法安全过滤。"
            )
        before_count = len(data)
        data = data[data["BUILDINGID"] == building_id].reset_index(drop=True)
        print(
            f"按训练配置过滤 BuildingID={building_id}："
            f"{before_count} 条 -> {len(data)} 条"
        )

    if data.empty:
        raise ValueError("输入数据在 BuildingID 过滤后为空")

    feature_scaler, target_scaler = load_preprocessors(
        MODEL_DIR / "preprocessors.joblib"
    )
    x_data, y_scaled = transform_dataframe(
        data,
        feature_scaler,
        target_scaler,
    )

    device = torch.device(DEVICE)
    checkpoint = torch.load(
        MODEL_DIR / "transformer_autoencoder.pt",
        map_location=device,
        weights_only=False,
    )
    autoencoder = TransformerAutoencoder(**checkpoint["model_params"]).to(device)
    autoencoder.load_state_dict(checkpoint["state_dict"])

    batch_size = run_config["training_config"]["batch_size"]
    features = encode_features(autoencoder, x_data, device, batch_size)

    if run_config["regressor"] == "svr":
        regressor = joblib.load(MODEL_DIR / "svr_regressor.joblib")
        pred_scaled = regressor.predict(features)
    else:
        training_config = run_config["training_config"]
        regressor = CoordinateMLP(
            checkpoint["model_params"]["d_model"],
            training_config["mlp_hidden_dim"],
        ).to(device)
        regressor.load_state_dict(
            torch.load(
                MODEL_DIR / "mlp_regressor.pt",
                map_location=device,
                weights_only=True,
            )
        )
        regressor.eval()
        with torch.no_grad():
            pred_scaled = regressor(
                torch.from_numpy(features).to(device)
            ).cpu().numpy()

    predictions = target_scaler.inverse_transform(pred_scaled)
    output = pd.DataFrame(
        {
            "PRED_LONGITUDE": predictions[:, 0],
            "PRED_LATITUDE": predictions[:, 1],
        }
    )

    if y_scaled is not None:
        y_true = target_scaler.inverse_transform(y_scaled)
        output["TRUE_LONGITUDE"] = y_true[:, 0]
        output["TRUE_LATITUDE"] = y_true[:, 1]
        output["POSITION_ERROR_M"] = np.linalg.norm(
            predictions - y_true,
            axis=1,
        )
        print("预测指标：", calculate_metrics(y_true, predictions))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"预测结果已保存到：{args.output_csv}")


if __name__ == "__main__":
    main()
