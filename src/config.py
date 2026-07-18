from __future__ import annotations

"""项目配置。

本文件只负责保存配置，不包含数据处理或训练逻辑。
将配置集中在这里，可以避免在 train.py、predict.py 中重复出现魔法数字。
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# 项目根目录：
# src/config.py 的上一级是 src，再上一级就是项目根目录 xiaoma。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataConfig:
    """数据读取和预处理配置。"""

    # 原始训练集和官方验证集的位置。
    train_csv: Path = PROJECT_ROOT / "data" / "trainingData.csv"
    eval_csv: Path = PROJECT_ROOT / "data" / "validationData.csv"

    # -1 表示使用所有建筑；0、1、2 表示只训练指定建筑的数据。
    building_id: int = -1

    # 从 trainingData.csv 中再划分一部分作为训练过程中的验证集。
    validation_size: float = 0.10

    # 如果一个 WAP 特征在训练集中有 98% 以上的值都缺失，则删除该特征。
    feature_missing_threshold: float = 0.98
    # False：保留全部 WAP001-WAP520
    # True：过滤几乎从不出现的 WAP
    enable_feature_filtering: bool = False
    # UJIIndoorLoc 使用 100 表示没有检测到对应的 WiFi 信号。
    raw_missing_value: float = 100.0

    # 将缺失信号转换成较弱的 RSSI 值，便于后续缩放和模型计算。
    filled_missing_value: float = -105.0

    # 数据集中的 WiFi 特征列都以 WAP 开头。
    wap_prefix: str = "WAP"


@dataclass(frozen=True)
class ModelConfig:
    """Transformer 自编码器结构配置。"""

    # Transformer 内部每个 token 的向量维度。
    d_model: int = 64

    # 多头注意力的头数；d_model 必须能够被 nhead 整除。
    nhead: int = 4

    # TransformerEncoderLayer 的堆叠层数。
    num_layers: int = 4

    # Transformer 前馈网络隐藏层维度。
    dim_feedforward: int = 128

    # Transformer 和解码器中的 dropout。
    dropout: float = 0.10

    # 每多少个连续 WAP 特征组成一个 patch。
    # 这样可以显著缩短 Transformer 的序列长度。
    token_mode: str = "wap"   # "wap" 或 "patch"
    patch_size: int = 16

    # Transformer 编码后输出给 SVR/MLP 的最终特征维度。
    latent_dim: int = 64

    # MLP 回归头隐藏层大小（逗号分隔），仅在 regressor="mlp" 时生效。
    mlp_hidden_sizes: str = "128,64"


@dataclass(frozen=True)
class TrainingConfig:
    """训练过程和 SVR 配置。"""

    # 随机种子，用于保证数据划分和初始化尽可能可复现。
    seed: int = 42

    # auto 会优先使用 CUDA，没有 CUDA 时自动回退到 CPU。
    device: str = "auto"

    # 自编码器训练参数。
    epochs: int = 30
    batch_size: int = 128
    feature_batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # 验证损失连续多少轮没有明显下降时停止训练。
    early_stopping_patience: int = 5
    min_delta: float = 1e-5

    # DataLoader 工作进程数。Windows 环境建议保留为 0。
    num_workers: int = 0

    # 回归器类型："svr" 使用两阶段 SVR，"mlp" 使用 MLP 回归头端到端联合训练。
    regressor: str = "svr"

    # MLP 联合训练时回归损失的权重。
    # total_loss = reconstruction_loss + regression_loss_weight * regression_loss
    regression_loss_weight: float = 1.0

    # SVR 参数。
    svr_c: float = 100.0
    svr_epsilon: float = 0.10
    svr_gamma: str = "scale"

    # 0 表示使用全部训练样本拟合 SVR。
    # 数据量很大、CPU 较慢时可以设置一个正整数进行随机采样。
    svr_max_train_samples: int = 0


@dataclass(frozen=True)
class AppConfig:
    """应用总配置。"""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # 所有模型、预测结果和日志都放在该目录下。
    model_root: Path = PROJECT_ROOT / "model"

    def to_dict(self) -> dict[str, Any]:
        """将配置转换成可写入 JSON 的普通字典。"""

        payload = asdict(self)

        # pathlib.Path 不能直接被标准 json 序列化，因此转换成字符串。
        payload["data"]["train_csv"] = str(self.data.train_csv)
        payload["data"]["eval_csv"] = str(self.data.eval_csv)
        payload["model_root"] = str(self.model_root)
        return payload


# 命令行没有覆盖参数时使用这组默认配置。
DEFAULT_CONFIG = AppConfig()
