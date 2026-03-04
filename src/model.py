"""
model.py
--------
Multi-scale 1D CNN -> Stacked LSTM -> Attention -> RUL Regression Head

Architecture:
    Static features (op_cluster + fault_mode + subset)
            |
    Linear -> tanh -> h0, c0  (LSTM hidden state init)
            |
    Sequential input -> Multi-scale 1D CNN (kernels 3,5,7)
            |
    LSTM Layer 1 (h0, c0) + MC Dropout
            |
    LSTM Layer 2 + MC Dropout
            |
    Attention Layer (weighted sum over time steps)
            |
        RUL Head (MSE regression only)

Classification is derived from predicted RUL at test time:
    RUL >= 75  -> Healthy   (class 0)
    RUL >= 50  -> Degrading (class 1)
    RUL >= 25  -> Warning   (class 2)
    RUL < 25   -> Critical  (class 3)

Why pure regression:
    The classification head was competing with the regression head.
    Both were learning from raw sensors independently.
    But class labels ARE derived from RUL by definition.
    Fix RMSE -> classification follows automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import yaml


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Multi-scale 1D CNN Block
# ---------------------------------------------------------------------------

class MultiScaleCNN(nn.Module):
    """
    Parallel 1D convolutions with kernel sizes [3, 5, 7].
    Each branch: Conv1d -> BatchNorm -> GELU -> MaxPool
    Outputs concatenated along channel dimension.

    Input:  [batch, seq_len, input_dim]
    Output: [batch, seq_len, cnn_out_channels * n_kernels]
    """

    def __init__(
        self,
        input_dim:    int,
        out_channels: int = 64,
        kernels:      list = [3, 5, 7],
        dropout:      float = 0.2
    ):
        super().__init__()

        self.branches = nn.ModuleList()
        for k in kernels:
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=out_channels,
                    kernel_size=k,
                    padding=k // 2
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.branches.append(branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        x = x.permute(0, 2, 1)   # -> [batch, input_dim, seq_len]

        branch_outs = []
        for branch in self.branches:
            out = branch(x)       # [batch, out_channels, seq_len]
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)   # [batch, out_channels*n_kernels, seq_len]
        out = out.permute(0, 2, 1)            # [batch, seq_len, out_channels*n_kernels]
        return out


# ---------------------------------------------------------------------------
# Static Encoder — initialises LSTM hidden state from static features
# ---------------------------------------------------------------------------

class StaticEncoder(nn.Module):
    """
    Encodes static features (cluster, fault mode, subset) into
    LSTM initial hidden/cell states h0 and c0.

    This gives the LSTM context-awareness before seeing any sensor data.
    FD001 engine and FD004 engine start with different hidden states.
    """

    def __init__(self, static_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(
        self,
        static: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # static: [batch, static_dim]
        encoded = self.encoder(static)   # [batch, hidden_dim]

        # Expand to [num_layers, batch, hidden_dim]
        h0 = encoded.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        c0 = torch.zeros_like(h0)

        return h0, c0


# ---------------------------------------------------------------------------
# Temporal Attention
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention over LSTM time steps.
    Learns which cycles are most informative for RUL prediction.

    Input:  [batch, seq_len, hidden_dim]
    Output: context [batch, hidden_dim], weights [batch, seq_len]
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        lstm_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_out: [batch, seq_len, hidden_dim]
        scores  = self.v(torch.tanh(self.W(lstm_out)))   # [batch, seq_len, 1]
        weights = F.softmax(scores, dim=1)               # [batch, seq_len, 1]
        context = (weights * lstm_out).sum(dim=1)        # [batch, hidden_dim]
        return context, weights.squeeze(-1)


# ---------------------------------------------------------------------------
# Main Model — Pure RUL Regression
# ---------------------------------------------------------------------------

class CMAPSS_CNN_LSTM(nn.Module):
    """
    Multi-scale 1D CNN -> Stacked LSTM -> Attention -> RUL Head

    Single regression head only.
    Classification derived from predicted RUL at test time.

    Args:
        input_dim:    Number of sequential features (sensors + rolling + op)
        static_dim:   Number of static features (cluster + fault + subset)
        hidden_dim:   LSTM hidden dimension
        num_layers:   Number of stacked LSTM layers
        dropout:      Dropout rate (also used for MC Dropout at inference)
        cnn_kernels:  List of CNN kernel sizes
        cnn_channels: CNN output channels per kernel
        attention_dim: Attention projection dimension
    """

    def __init__(
        self,
        input_dim:     int = 73,
        static_dim:    int = 9,
        hidden_dim:    int = 128,
        num_layers:    int = 2,
        dropout:       float = 0.3,
        cnn_kernels:   list = [3, 5, 7],
        cnn_channels:  int = 64,
        attention_dim: int = 64,
        num_classes:   int = 4   # kept for compatibility, not used in forward
    ):
        super().__init__()

        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.dropout_rate = dropout

        # Static encoder -> LSTM h0, c0
        self.static_encoder = StaticEncoder(
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # Multi-scale CNN
        self.cnn = MultiScaleCNN(
            input_dim=input_dim,
            out_channels=cnn_channels,
            kernels=cnn_kernels,
            dropout=dropout * 0.5
        )
        cnn_out_dim = cnn_channels * len(cnn_kernels)

        # Stacked LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.mc_dropout = nn.Dropout(dropout)

        # Attention
        self.attention = TemporalAttention(
            hidden_dim=hidden_dim,
            attention_dim=attention_dim
        )

        # RUL regression head only
        head_input_dim = hidden_dim + static_dim
        self.rul_head = nn.Sequential(
            nn.Linear(head_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.ReLU()   # RUL is always >= 0
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, orthogonal for LSTM."""
        for name, param in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Forget gate bias = 1 (helps LSTM remember longer)
                    n = param.size(0)
                    param.data[n//4 : n//2].fill_(1.0)

    def forward(
        self,
        x:      torch.Tensor,
        static: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:       [batch, seq_len, input_dim]
            static:  [batch, static_dim]
            hidden:  Optional (h, c) for recursive inference

        Returns:
            rul_pred:     [batch, 1]       predicted RUL
            attn_weights: [batch, seq_len] attention weights
            (h_n, c_n):   LSTM final state
        """
        # Static -> initial hidden state
        if hidden is None:
            h0, c0 = self.static_encoder(static)
        else:
            h0, c0 = hidden

        # CNN
        cnn_out = self.cnn(x)                            # [batch, seq_len, cnn_out_dim]

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_out, (h0, c0))
        lstm_out = self.mc_dropout(lstm_out)

        # Attention
        context, attn_weights = self.attention(lstm_out)  # [batch, hidden_dim]

        # Head
        head_input = torch.cat([context, static], dim=-1)
        rul_pred   = self.rul_head(head_input)            # [batch, 1]

        return rul_pred, attn_weights, (h_n, c_n)

    def predict_class(
        self,
        rul_pred: torch.Tensor,
        bins: Tuple[int, int, int] = (75, 50, 25)
    ) -> torch.Tensor:
        """
        Derive health class from predicted RUL.
        Called at test time — no separate classification head needed.

        Class 0 - Healthy:   RUL >= 75
        Class 1 - Degrading: 50 <= RUL < 75
        Class 2 - Warning:   25 <= RUL < 50
        Class 3 - Critical:  RUL < 25
        """
        rul    = rul_pred.squeeze(-1)
        h, d, w = bins
        classes = torch.zeros(rul.shape, dtype=torch.long, device=rul.device)
        classes[rul < h] = 1
        classes[rul < d] = 2
        classes[rul < w] = 3
        return classes

    def enable_dropout(self):
        """Enable dropout layers for MC Dropout uncertainty estimation."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss — pure MSE on normalised RUL
# ---------------------------------------------------------------------------

class CMAPSSLoss(nn.Module):
    """
    Pure MSE loss on RUL regression.

    Normalised by RUL_CAP^2 so loss scale is ~0.0 to 1.0
    regardless of RUL cap value.

    loss = MSE(pred, target) / (rul_cap^2)
    """

    def __init__(self, rul_cap: float = 125.0):
        super().__init__()
        self.rul_cap  = rul_cap
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        rul_pred:   torch.Tensor,
        rul_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss: normalised MSE (scale 0-1)
            mse:        raw MSE in cycles^2 (for RMSE logging)
        """
        mse   = self.mse_loss(rul_pred.squeeze(-1), rul_target)
        total = mse / (self.rul_cap ** 2)
        return total, mse


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(config_path: str = "configs/config.yaml") -> CMAPSS_CNN_LSTM:
    cfg  = load_config(config_path)
    mcfg = cfg["model"]

    model = CMAPSS_CNN_LSTM(
        input_dim=mcfg["input_dim"],
        static_dim=mcfg["static_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_layers=mcfg["num_lstm_layers"],
        dropout=mcfg["dropout"],
        cnn_kernels=mcfg["cnn_kernels"],
        cnn_channels=mcfg["cnn_out_channels"],
        attention_dim=mcfg["attention_dim"],
        num_classes=mcfg["num_classes"]
    )
    return model


def build_loss(config_path: str = "configs/config.yaml") -> CMAPSSLoss:
    cfg = load_config(config_path)
    return CMAPSSLoss(rul_cap=float(cfg["data"]["rul_cap"]))
