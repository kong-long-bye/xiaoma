from pathlib import Path

import torch

# 项目路径
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"
LOG_DIR = MODEL_DIR / "logs"

# UJIIndoorLoc 数据文件
TRAIN_CSV = DATA_DIR / "trainingData.csv"
TEST_CSV = DATA_DIR / "validationData.csv"
DATASET_URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"

# 基础设置
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAP_DIM = 520
TARGET_COLUMNS = ["LONGITUDE", "LATITUDE"]

# Building 0 的论文参数。
# 论文未给出 dim_feedforward，这里按 d_model 的 4 倍设置为 512。
DEFAULT_CONFIG = {
    "d_model": 128,
    "nhead": 16,
    "num_layers": 15,
    "dim_feedforward": 512,
    "dropout": 0.0446,
    "learning_rate": 0.0022103,
    "weight_decay": 1e-4,
    "batch_size": 16,
    "epochs": 30,
    "patience": 4,
    "svr_c": 223.1289,
    "svr_epsilon": 0.5082,
    "svr_gamma": "scale",
    "mlp_hidden_dim": 128,
    "mlp_learning_rate": 1e-3,
    "mlp_epochs": 100,
    "mlp_patience": 10,
}

# 仅用于检查完整流程，不用于正式结果。
SMOKE_CONFIG = {
    **DEFAULT_CONFIG,
    "d_model": 8,
    "nhead": 1,
    "num_layers": 1,
    "dim_feedforward": 16,
    "batch_size": 2,
    "epochs": 1,
    "patience": 1,
    "svr_c": 10.0,
    "mlp_hidden_dim": 16,
    "mlp_epochs": 2,
    "mlp_patience": 1,
}
