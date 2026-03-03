"""
model.py
--------
Multi-scale 1D CNN → Stacked LSTM → Attention → Dual Head

Architecture:
    Static features (op_cluster + fault_mode)
            ↓
    Linear → tanh → h₀, c₀  (hidden state init)
            ↓
    Sequential input → Multi-scale 1D CNN (kernels 3,5,7)
            ↓
    LSTM Layer 1 (h₀, c₀) + MC Dropout
            ↓
    LSTM Layer 2 + MC Dropout
            ↓
    Attention Layer (weighted sum over time steps)
            ↓
       ┌────┴────┐
  RUL Head   Class Head
  (MSE)      (Focal CE)
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
    Each branch: Conv1d → BatchNorm → GELU → MaxPool
    Outputs are concatenated along channel dimension.

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
                # Conv1d expects [batch, channels, seq_len]
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=out_channels,
                    kernel_size=k,
                    padding=k // 2   # same padding — preserves seq_len
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.branches.append(branch)

        self.out_dim = out_channels * len(kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        x_t = x.permute(0, 2, 1)   # → [batch, input_dim, seq_len]

        branch_outs = []
        for branch in self.branches:
            out = branch(x_t)       # → [batch, out_channels, seq_len]
            branch_outs.append(out)

        # Concatenate along channel dimension
        out = torch.cat(branch_outs, dim=1)   # → [batch, out_channels*3, seq_len]
        out = out.permute(0, 2, 1)            # → [batch, seq_len, out_channels*3]

        return out


# ---------------------------------------------------------------------------
# Attention Layer
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Additive attention over LSTM output sequence.
    Learns to weight recent/informative time steps more.

    Input:  [batch, seq_len, hidden_dim]
    Output: [batch, hidden_dim]  (context vector)
            [batch, seq_len]     (attention weights for interpretability)
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        lstm_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_out: [batch, seq_len, hidden_dim]

        # Compute attention scores
        scores = torch.tanh(self.W(lstm_out))   # [batch, seq_len, attn_dim]
        scores = self.v(scores).squeeze(-1)      # [batch, seq_len]

        # Normalise to probabilities
        weights = F.softmax(scores, dim=-1)      # [batch, seq_len]

        # Weighted sum of LSTM outputs
        context = torch.bmm(
            weights.unsqueeze(1),   # [batch, 1, seq_len]
            lstm_out                # [batch, seq_len, hidden_dim]
        ).squeeze(1)                # [batch, hidden_dim]

        return context, weights


# ---------------------------------------------------------------------------
# Static Feature Encoder (for hidden state initialisation)
# ---------------------------------------------------------------------------

class StaticEncoder(nn.Module):
    """
    Encodes static features into LSTM initial hidden and cell states.

    Input:  [batch, static_dim]
    Output: h₀, c₀  each [num_layers, batch, hidden_dim]
    """

    def __init__(
        self,
        static_dim:  int,
        hidden_dim:  int,
        num_layers:  int
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # One linear layer projects static → hidden_dim
        # We produce both h and c
        self.encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.Tanh()
        )

        # Separate projections for h and c per layer
        self.h_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.c_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        static: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # static: [batch, static_dim]
        base = self.encoder(static)   # [batch, hidden_dim]

        h_list = [proj(base) for proj in self.h_proj]
        c_list = [proj(base) for proj in self.c_proj]

        # Stack to [num_layers, batch, hidden_dim]
        h0 = torch.stack(h_list, dim=0)
        c0 = torch.stack(c_list, dim=0)

        return h0, c0


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Combined Weighted CrossEntropy + Focal Loss.

    FL(p_t) = -w_c * (1 - p_t)^gamma * log(p_t)

    Args:
        class_weights:  Tensor of per-class weights [n_classes]
        gamma:          Focusing parameter (default 2.0)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0
    ):
        super().__init__()
        self.gamma        = gamma
        self.class_weights = class_weights

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # logits:  [batch, n_classes]
        # targets: [batch]

        # Standard CE loss (with class weights)
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            reduction="none"
        )   # [batch]

        # p_t = probability of true class
        probs = F.softmax(logits, dim=-1)                    # [batch, n_classes]
        p_t   = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [batch]

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        loss = (focal_weight * ce_loss).mean()

        return loss


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class CMAPSS_CNN_LSTM(nn.Module):
    """
    Multi-scale 1D CNN → Stacked LSTM → Attention → Dual Head

    Args:
        input_dim:    Number of sequential features (sensors + op_settings)
        static_dim:   Number of static features (one-hot cluster + fault mode)
        hidden_dim:   LSTM hidden dimension
        num_layers:   Number of stacked LSTM layers
        dropout:      Dropout rate (used for MC Dropout at inference too)
        cnn_kernels:  List of CNN kernel sizes
        cnn_channels: CNN output channels per kernel
        attention_dim: Attention projection dimension
        num_classes:  Number of health classes
    """

    def __init__(
        self,
        input_dim:     int = 17,
        static_dim:    int = 8,
        hidden_dim:    int = 128,
        num_layers:    int = 2,
        dropout:       float = 0.3,
        cnn_kernels:   list = [3, 5, 7],
        cnn_channels:  int = 64,
        attention_dim: int = 64,
        num_classes:   int = 4
    ):
        super().__init__()

        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.dropout_rate = dropout

        # ── Static encoder → LSTM h₀, c₀ ─────────────────────────────────
        self.static_encoder = StaticEncoder(
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        # ── Multi-scale CNN ───────────────────────────────────────────────
        self.cnn = MultiScaleCNN(
            input_dim=input_dim,
            out_channels=cnn_channels,
            kernels=cnn_kernels,
            dropout=dropout * 0.5   # lighter dropout in CNN
        )
        cnn_out_dim = cnn_channels * len(cnn_kernels)

        # ── Stacked LSTM ──────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # MC Dropout layer (kept active during inference)
        self.mc_dropout = nn.Dropout(dropout)

        # ── Attention ─────────────────────────────────────────────────────
        self.attention = TemporalAttention(
            hidden_dim=hidden_dim,
            attention_dim=attention_dim
        )

        # ── Dual Head ─────────────────────────────────────────────────────
        # Concat context + static features again before heads
        head_input_dim = hidden_dim + static_dim

        # RUL regression head
        self.rul_head = nn.Sequential(
            nn.Linear(head_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Health class head
        self.class_head = nn.Sequential(
            nn.Linear(head_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # Weight initialisation
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
                    # Set forget gate bias to 1 (helps LSTM remember)
                    n = param.size(0)
                    param.data[n//4 : n//2].fill_(1.0)
            elif isinstance(self, nn.Linear):
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        x:      torch.Tensor,
        static: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x:       [batch, seq_len, input_dim]  sequential sensor input
            static:  [batch, static_dim]           static features
            hidden:  Optional (h, c) for recursive inference

        Returns:
            rul_pred:    [batch, 1]        predicted RUL
            class_logits:[batch,n_classes] class logits
            attn_weights:[batch, seq_len]  attention weights
            (h_n, c_n):  LSTM final hidden state (for recursive inference)
        """
        batch_size = x.size(0)

        # ── Static → initial hidden state ─────────────────────────────────
        if hidden is None:
            h0, c0 = self.static_encoder(static)
        else:
            h0, c0 = hidden

        # ── Multi-scale CNN ───────────────────────────────────────────────
        cnn_out = self.cnn(x)   # [batch, seq_len, cnn_out_dim]

        # ── LSTM ──────────────────────────────────────────────────────────
        lstm_out, (h_n, c_n) = self.lstm(cnn_out, (h0, c0))
        # lstm_out: [batch, seq_len, hidden_dim]

        # MC Dropout on LSTM output (active during both train and inference)
        lstm_out = self.mc_dropout(lstm_out)

        # ── Attention ─────────────────────────────────────────────────────
        context, attn_weights = self.attention(lstm_out)
        # context: [batch, hidden_dim]

        # ── Concat static features for final prediction ───────────────────
        head_input = torch.cat([context, static], dim=-1)
        # head_input: [batch, hidden_dim + static_dim]

        # ── Dual Head ─────────────────────────────────────────────────────
        rul_pred     = self.rul_head(head_input)         # [batch, 1]
        class_logits = self.class_head(head_input)       # [batch, n_classes]

        return rul_pred, class_logits, attn_weights, (h_n, c_n)

    def enable_dropout(self):
        """Enable dropout for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class CMAPSSLoss(nn.Module):
    """
    Combined loss: alpha * MSE_RUL + (1 - alpha) * FocalCE_class

    Args:
        class_weights:  Per-class weights for focal loss
        alpha:          Weight for RUL MSE loss (0.0 to 1.0)
        gamma:          Focal loss focusing parameter
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        gamma: float = 2.0
    ):
        super().__init__()
        self.alpha      = alpha
        self.mse_loss   = nn.MSELoss()
        self.focal_loss = FocalLoss(class_weights=class_weights, gamma=gamma)

    def forward(
        self,
        rul_pred:    torch.Tensor,
        rul_target:  torch.Tensor,
        class_logits: torch.Tensor,
        class_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss: alpha * mse + (1-alpha) * focal
            mse:        RUL MSE component
            focal:      classification focal loss component
        """
        mse   = self.mse_loss(rul_pred.squeeze(-1), rul_target)
        focal = self.focal_loss(class_logits, class_target)

        total = self.alpha * mse + (1 - self.alpha) * focal

        return total, mse, focal


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(config_path: str = "configs/config.yaml") -> CMAPSS_CNN_LSTM:
    cfg   = load_config(config_path)
    mcfg  = cfg["model"]
    tcfg  = cfg["training"]

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


def build_loss(
    class_weights: Optional[torch.Tensor] = None,
    config_path: str = "configs/config.yaml"
) -> CMAPSSLoss:
    cfg  = load_config(config_path)
    tcfg = cfg["training"]
    mcfg = cfg["model"]

    return CMAPSSLoss(
        class_weights=class_weights,
        alpha=tcfg["alpha"],
        gamma=tcfg["focal_gamma"]
    )


# ---------------------------------------------------------------------------
# Entry point for architecture inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_model()

    print("=" * 60)
    print("CMAPSS CNN-LSTM Architecture")
    print("=" * 60)
    print(model)
    print("=" * 60)
    print(f"Total trainable parameters: {model.count_parameters():,}")
    print("=" * 60)

    # Test forward pass
    batch_size = 8
    seq_len    = 30
    input_dim  = 17
    static_dim = 8

    x      = torch.randn(batch_size, seq_len, input_dim)
    static = torch.randn(batch_size, static_dim)

    rul_pred, class_logits, attn_weights, (h_n, c_n) = model(x, static)

    print(f"\nForward pass shapes:")
    print(f"  Input x:        {x.shape}")
    print(f"  Input static:   {static.shape}")
    print(f"  RUL pred:       {rul_pred.shape}")
    print(f"  Class logits:   {class_logits.shape}")
    print(f"  Attn weights:   {attn_weights.shape}")
    print(f"  h_n:            {h_n.shape}")
    print(f"  c_n:            {c_n.shape}")
    print("\n[✓] Forward pass successful.")
