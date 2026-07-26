from __future__ import annotations

"""MLP-only 基线模型配置。

纯 MLP 模型，用于 Transformer 消融实验。
所有参数集中在此文件，避免散布在训练代码中。
"""

from dataclasses import dataclass
from pathlib import Path


# 项目根目录：当前文件所在目录的父目录的父目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MLP-only 模型默认根目录，与 Transformer 模型分开存放
MODEL_ROOT = PROJECT_ROOT / "model" / "mlp-single"


def parse_layer_sizes(value: str) -> tuple[int, ...]:
    """将 "256,128,64" 格式的字符串解析为整数元组。"""
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


@dataclass(frozen=True)
class MLPSingleConfig:
    """MLP-only 基线模型的所有参数。"""

    # MLP 网络结构
    hidden_layer_sizes: tuple[int, ...] = (256, 128, 64)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 1e-4

    # 优化器参数
    learning_rate_init: float = 1e-3
    batch_size: int = 128

    # 训练控制
    epochs: int = 100
    early_stopping_patience: int = 10
    min_delta: float = 1e-5

    # 随机种子
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "min_delta": self.min_delta,
            "seed": self.seed,
        }
