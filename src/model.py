"""
model.py
--------
Multi-scale 1D CNN -> Stacked LSTM -> Attention -> RUL Regression

Architecture:
    Static features (op_cluster + fault_mode + subset)
            |
    Linear -> tanh -> h0, c0  (LSTM hidden state init)
            |
    Sequential input -> Multi-scale 1D CNN (kernels 3,5,7)
            |
    Stacked LSTM (2 layers) + MC Dropout
            |
    Temporal Attention
            |
        RUL Head  (single scalar output)

Pure regression. No classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Multi-scale 1D CNN
# ---------------------------------------------------------------------------

class MultiScaleCNN(nn.Module):
    """Parallel Conv1d branches with kernels [3,5,7], concatenated."""

    def __init__(self, input_dim: int, out_channels: int = 64,
                 kernels: list = [3, 5, 7], dropout: float = 0.2):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, out_channels, k, padding=k // 2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for k in kernels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        x   = x.permute(0, 2, 1)                          # [batch, input_dim, seq_len]
        out = torch.cat([b(x) for b in self.branches], 1) # [batch, out*n_kernels, seq_len]
        return out.permute(0, 2, 1)                        # [batch, seq_len, out*n_kernels]


# ---------------------------------------------------------------------------
# Static encoder — seeds LSTM hidden state
# ---------------------------------------------------------------------------

class StaticEncoder(nn.Module):
    def __init__(self, static_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.encoder    = nn.Sequential(nn.Linear(static_dim, hidden_dim), nn.Tanh())

    def forward(self, static: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(static)                                  # [batch, hidden]
        h0      = encoded.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c0      = torch.zeros_like(h0)
        return h0, c0


# ---------------------------------------------------------------------------
# Temporal attention
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.v(torch.tanh(self.W(lstm_out))), dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return context, weights.squeeze(-1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CMAPSS_CNN_LSTM(nn.Module):
    def __init__(
        self,
        input_dim:     int   = 73,
        static_dim:    int   = 9,
        hidden_dim:    int   = 128,
        num_layers:    int   = 2,
        dropout:       float = 0.4,
        cnn_kernels:   list  = [3, 5, 7],
        cnn_channels:  int   = 64,
        attention_dim: int   = 64,
        **kwargs  # absorb unused keys like num_classes
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers

        self.static_encoder = StaticEncoder(static_dim, hidden_dim, num_layers)

        self.cnn = MultiScaleCNN(input_dim, cnn_channels, cnn_kernels, dropout * 0.5)
        cnn_out_dim = cnn_channels * len(cnn_kernels)

        self.lstm = nn.LSTM(
            cnn_out_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.mc_dropout = nn.Dropout(dropout)
        self.attention  = TemporalAttention(hidden_dim, attention_dim)

        head_input_dim = hidden_dim + static_dim
        self.rul_head  = nn.Sequential(
            nn.Linear(head_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.ReLU()   # RUL >= 0
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name: nn.init.xavier_uniform_(param)
                elif "weight_hh" in name: nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    n = param.size(0)
                    param.data[n//4 : n//2].fill_(1.0)  # forget gate bias = 1

    def forward(
        self,
        x:      torch.Tensor,
        static: torch.Tensor,
        hidden: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:       [batch, seq_len, input_dim]
            static:  [batch, static_dim]
        Returns:
            rul_pred:     [batch, 1]
            attn_weights: [batch, seq_len]
        """
        h0, c0   = self.static_encoder(static) if hidden is None else hidden
        cnn_out  = self.cnn(x)
        lstm_out, _ = self.lstm(cnn_out, (h0, c0))
        lstm_out = self.mc_dropout(lstm_out)
        context, attn_weights = self.attention(lstm_out)
        head_input = torch.cat([context, static], dim=-1)
        rul_pred   = self.rul_head(head_input)
        return rul_pred, attn_weights

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss — normalised MSE
# ---------------------------------------------------------------------------

class RULLoss(nn.Module):
    """
    MSE normalised by rul_cap^2 so loss is always in [0, 1] range.
    loss = MSE(pred, target) / rul_cap^2
    """
    def __init__(self, rul_cap: float = 125.0):
        super().__init__()
        self.rul_cap  = rul_cap
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mse   = self.mse_loss(pred.squeeze(-1), target)
        total = mse / (self.rul_cap ** 2)
        return total, mse


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_model(config_path: str = "configs/config.yaml") -> CMAPSS_CNN_LSTM:
    cfg  = load_config(config_path)
    mcfg = cfg["model"]
    return CMAPSS_CNN_LSTM(
        input_dim=mcfg["input_dim"],
        static_dim=mcfg["static_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_layers=mcfg["num_lstm_layers"],
        dropout=mcfg["dropout"],
        cnn_kernels=mcfg["cnn_kernels"],
        cnn_channels=mcfg["cnn_out_channels"],
        attention_dim=mcfg["attention_dim"],
    )


def build_loss(config_path: str = "configs/config.yaml") -> RULLoss:
    cfg = load_config(config_path)
    return RULLoss(rul_cap=float(cfg["data"]["rul_cap"]))
