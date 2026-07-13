import torch
from torch import nn


class TransformerAutoencoder(nn.Module):
    """使用 CLS token 汇聚 520 个 WAP 的 Transformer 自编码器。"""

    def __init__(
        self,
        input_dim: int = 520,
        d_model: int = 128,
        nhead: int = 16,
        num_layers: int = 15,
        dim_feedforward: int = 512,
        dropout: float = 0.0446,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model 必须能够被 nhead 整除")

        self.input_dim = input_dim
        self.d_model = d_model

        # 每个 RSSI 标量映射到 d_model 维。
        self.input_projection = nn.Linear(1, d_model)

        # WAP 编号必须被保留，否则模型无法区分不同接入点。
        self.ap_embedding = nn.Parameter(torch.zeros(1, input_dim, d_model))

        # CLS token 用于汇聚整条 RSSI 指纹，不再使用 mean pooling。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.ap_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            # 论文描述为 LayerNorm(x + Sublayer(x))，对应 post-norm。
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(d_model)

        # 直接使用 d_model 维 CLS 特征重构 520 维 RSSI。
        # 编码特征也原样交给 SVR，不再增加额外 latent 压缩层。
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 520]
        wap_tokens = self.input_projection(x.unsqueeze(-1))
        wap_tokens = wap_tokens + self.ap_embedding

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_tokens, wap_tokens], dim=1)

        hidden = self.encoder(tokens)

        # 第 0 个位置是 CLS token，形状为 [batch_size, d_model]。
        return self.output_norm(hidden[:, 0, :])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(x)
        reconstructed = self.decoder(encoded)
        return reconstructed, encoded


class CoordinateMLP(nn.Module):
    """使用 Transformer 编码特征预测经纬度的两层 MLP。"""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        second_hidden = max(hidden_dim // 2, 16)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
