
from __future__ import annotations

"""模型定义。

模型同时支持两种 token 化方式：

1. wap 模式：
   每一个 WAP 特征作为一个独立 token。

2. patch 模式：
   每 patch_size 个连续 WAP 特征组成一个 token。

两种模式最终都会经过 TransformerEncoder 提取 latent 特征，
再由解码器重建原始 WAP 输入。
"""

import math
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from src.config import ModelConfig


class WiFiTransformerAutoencoder(nn.Module):
    """同时支持 WAP token 和 Patch token 的 Transformer 自编码器。

    输入:
        shape = (batch_size, WAP特征数量)

    输出:
        forward(x) 返回重建后的 WAP 特征；
        forward(x, joint=True) 返回 (重建结果, 坐标预测)；
        encode() 返回给 SVR 使用的 latent 特征；
        decode() 根据 latent 特征重建原始输入。

    token_mode="wap":
        每一个 WAP 数值作为一个独立 token。

    token_mode="patch":
        每 patch_size 个连续 WAP 数值组成一个 token。
    """

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        # 提前检查配置，避免训练时出现难以定位的形状错误。
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if config.d_model % config.nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead."
            )

        if config.token_mode not in {"wap", "patch"}:
            raise ValueError(
                "token_mode must be 'wap' or 'patch'."
            )

        if config.patch_size <= 0:
            raise ValueError(
                "patch_size must be positive."
            )

        self.input_dim = int(input_dim)
        self.config = config

        # 根据 token_mode 决定 Transformer 的序列长度
        # 以及 token embedding 的输入维度。
        if config.token_mode == "wap":
            # 每一个 WAP 数值作为一个 token。
            #
            # (batch, input_dim)
            # ->
            # (batch, input_dim, 1)
            # ->
            # (batch, input_dim, d_model)
            self.num_tokens = self.input_dim
            self.token_embedding = nn.Linear(
                1,
                config.d_model,
            )

        else:
            # 每 patch_size 个连续 WAP 特征组成一个 token。
            self.num_tokens = math.ceil(
                self.input_dim / config.patch_size
            )
            self.padded_dim = (
                self.num_tokens * config.patch_size
            )

            # 每个 patch 包含 patch_size 个 RSSI 数值。
            self.token_embedding = nn.Linear(
                config.patch_size,
                config.d_model,
            )

        # 可学习的位置编码，用来区分不同的 WAP 或 patch。
        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_tokens,
                config.d_model,
            )
        )

        # 使用 PyTorch 官方 TransformerEncoderLayer，
        # 不再手写注意力、残差连接和 LayerNorm。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )

        # 将 Transformer 输出压缩成固定长度的 latent 特征。
        self.latent_head = nn.Sequential(
            nn.Linear(
                config.d_model,
                config.latent_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(config.latent_dim),
        )

        # 根据 latent 特征重建原始 WAP 输入。
        self.decoder = nn.Sequential(
            nn.Linear(
                config.latent_dim,
                config.dim_feedforward,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.dim_feedforward,
                self.input_dim,
            ),
        )

        # MLP 回归头：将 latent 特征映射到坐标 (longitude, latitude)。
        self._init_regression_head(config)

        # 使用较小方差初始化位置编码。
        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02,
        )

    def _init_regression_head(
        self,
        config: ModelConfig,
    ) -> None:
        """初始化 MLP 回归头。"""

        mlp_hidden = [
            int(x)
            for x in config.mlp_hidden_sizes.split(",")
        ]

        layers: list[nn.Module] = []
        prev_dim = config.latent_dim

        for hidden_dim in mlp_hidden:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden_dim

        # 输出二维坐标：经度、纬度。
        layers.append(nn.Linear(prev_dim, 2))
        self.regression_head = nn.Sequential(*layers)

    def _to_tokens(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """根据 token_mode 将二维 WAP 输入转换成 token 序列。"""

        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected shape "
                f"(batch, {self.input_dim}), "
                f"got {tuple(x.shape)}."
            )

        if self.config.token_mode == "wap":
            # 每一个 WAP 增加一个维度，作为独立 token。
            token_values = x.unsqueeze(-1)

        else:
            # Patch 模式下，如果特征数量不能整除 patch_size，
            # 则只在末尾补零。
            if self.padded_dim > self.input_dim:
                x = F.pad(
                    x,
                    (
                        0,
                        self.padded_dim
                        - self.input_dim,
                    ),
                )

            token_values = x.reshape(
                x.shape[0],
                self.num_tokens,
                self.config.patch_size,
            )

        # 将 WAP 标量或 patch 向量映射成 d_model 维 token。
        tokens = self.token_embedding(token_values)

        return tokens + self.position_embedding

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """提取固定长度的 latent 特征，供后续 SVR 使用。"""

        tokens = self._to_tokens(x)
        encoded = self.encoder(tokens)

        # 当前不使用 CLS token，直接对所有 token 做平均池化。
        pooled = encoded.mean(dim=1)

        return self.latent_head(pooled)

    def decode(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """根据 latent 特征重建原始 WAP 输入。"""

        if (
            latent.ndim != 2
            or latent.shape[1] != self.config.latent_dim
        ):
            raise ValueError(
                f"Expected latent shape "
                f"(batch, {self.config.latent_dim}), "
                f"got {tuple(latent.shape)}."
            )

        return self.decoder(latent)

    def forward(
        self,
        x: torch.Tensor,
        joint: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """执行完整自编码器前向传播。

        Args:
            x: 输入 WAP 特征。
            joint: 若为 True，跳过解码，直接用 MLP 回归头输出坐标。

        Returns:
            joint=False: 重建后的 WAP 特征。
            joint=True: 坐标预测 (batch, 2)。
        """

        latent = self.encode(x)

        if joint:
            # MLP 模式：直接回归坐标，不做重建。
            return self.regression_head(latent)

        return self.decode(latent)

    def checkpoint_payload(
        self,
    ) -> dict[str, object]:
        """构造保存到 transformer.pt 的完整检查点。"""

        return {
            "input_dim": self.input_dim,
            "model_config": asdict(self.config),

            # 保存 CPU 权重，便于在不同设备之间加载。
            "state_dict": {
                key: value.detach().cpu()
                for key, value
                in self.state_dict().items()
            },
        }

