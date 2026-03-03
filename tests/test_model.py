"""
tests/test_model.py
-------------------
Tests for model architecture, forward pass, loss, and MC Dropout.

Run with:
    pytest tests/test_model.py -v
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import (
    CMAPSS_CNN_LSTM,
    MultiScaleCNN,
    TemporalAttention,
    StaticEncoder,
    FocalLoss,
    CMAPSSLoss,
    build_model,
    build_loss
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH      = 8
SEQ_LEN    = 30
INPUT_DIM  = 17
STATIC_DIM = 8
NUM_CLASSES = 4
HIDDEN_DIM = 64   # smaller for fast tests


@pytest.fixture
def model():
    return CMAPSS_CNN_LSTM(
        input_dim=INPUT_DIM,
        static_dim=STATIC_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        dropout=0.3,
        cnn_kernels=[3, 5, 7],
        cnn_channels=32,
        attention_dim=32,
        num_classes=NUM_CLASSES
    )


@pytest.fixture
def dummy_batch():
    x      = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
    static = torch.randn(BATCH, STATIC_DIM)
    y_rul  = torch.rand(BATCH) * 125
    y_cls  = torch.randint(0, NUM_CLASSES, (BATCH,))
    return x, static, y_rul, y_cls


# ---------------------------------------------------------------------------
# MultiScaleCNN
# ---------------------------------------------------------------------------

class TestMultiScaleCNN:

    def test_output_shape(self):
        cnn = MultiScaleCNN(input_dim=INPUT_DIM, out_channels=32,
                            kernels=[3, 5, 7])
        x   = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = cnn(x)
        # output channels = 32 * 3 kernels = 96
        assert out.shape == (BATCH, SEQ_LEN, 96), \
            f"Expected ({BATCH}, {SEQ_LEN}, 96), got {out.shape}"

    def test_sequence_length_preserved(self):
        """Same padding should preserve sequence length."""
        cnn = MultiScaleCNN(input_dim=INPUT_DIM, out_channels=16,
                            kernels=[3, 5, 7])
        for seq_len in [10, 30, 50, 100]:
            x   = torch.randn(BATCH, seq_len, INPUT_DIM)
            out = cnn(x)
            assert out.shape[1] == seq_len, \
                f"Sequence length changed: {seq_len} → {out.shape[1]}"

    def test_no_nan_output(self):
        cnn = MultiScaleCNN(input_dim=INPUT_DIM, out_channels=32)
        x   = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        out = cnn(x)
        assert not torch.isnan(out).any(), "NaN in CNN output"


# ---------------------------------------------------------------------------
# TemporalAttention
# ---------------------------------------------------------------------------

class TestTemporalAttention:

    def test_output_shapes(self):
        attn    = TemporalAttention(hidden_dim=HIDDEN_DIM, attention_dim=32)
        lstm_out = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        context, weights = attn(lstm_out)

        assert context.shape == (BATCH, HIDDEN_DIM), \
            f"Context shape wrong: {context.shape}"
        assert weights.shape == (BATCH, SEQ_LEN), \
            f"Weights shape wrong: {weights.shape}"

    def test_attention_weights_sum_to_one(self):
        attn     = TemporalAttention(hidden_dim=HIDDEN_DIM)
        lstm_out = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        _, weights = attn(lstm_out)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5), \
            f"Attention weights don't sum to 1: {sums}"


# ---------------------------------------------------------------------------
# StaticEncoder
# ---------------------------------------------------------------------------

class TestStaticEncoder:

    def test_output_shapes(self):
        encoder = StaticEncoder(
            static_dim=STATIC_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=2
        )
        static = torch.randn(BATCH, STATIC_DIM)
        h0, c0 = encoder(static)

        assert h0.shape == (2, BATCH, HIDDEN_DIM), \
            f"h0 shape wrong: {h0.shape}"
        assert c0.shape == (2, BATCH, HIDDEN_DIM), \
            f"c0 shape wrong: {c0.shape}"

    def test_different_statics_give_different_hidden(self):
        encoder  = StaticEncoder(STATIC_DIM, HIDDEN_DIM, num_layers=2)
        static_a = torch.zeros(1, STATIC_DIM)
        static_b = torch.ones(1, STATIC_DIM)
        h0_a, _ = encoder(static_a)
        h0_b, _ = encoder(static_b)
        assert not torch.allclose(h0_a, h0_b), \
            "Different static inputs should give different hidden states"


# ---------------------------------------------------------------------------
# Full Model Forward Pass
# ---------------------------------------------------------------------------

class TestModelForwardPass:

    def test_output_shapes(self, model, dummy_batch):
        x, static, _, _ = dummy_batch
        rul_pred, class_logits, attn_weights, (h_n, c_n) = model(x, static)

        assert rul_pred.shape    == (BATCH, 1),           f"RUL shape: {rul_pred.shape}"
        assert class_logits.shape == (BATCH, NUM_CLASSES), f"Logits shape: {class_logits.shape}"
        assert attn_weights.shape == (BATCH, SEQ_LEN),    f"Attn shape: {attn_weights.shape}"
        assert h_n.shape == (2, BATCH, HIDDEN_DIM),       f"h_n shape: {h_n.shape}"
        assert c_n.shape == (2, BATCH, HIDDEN_DIM),       f"c_n shape: {c_n.shape}"

    def test_no_nan_in_outputs(self, model, dummy_batch):
        x, static, _, _ = dummy_batch
        rul_pred, class_logits, attn_weights, _ = model(x, static)

        assert not torch.isnan(rul_pred).any(),     "NaN in RUL predictions"
        assert not torch.isnan(class_logits).any(), "NaN in class logits"
        assert not torch.isnan(attn_weights).any(), "NaN in attention weights"

    def test_class_probs_sum_to_one(self, model, dummy_batch):
        x, static, _, _ = dummy_batch
        _, class_logits, _, _ = model(x, static)
        probs = torch.softmax(class_logits, dim=-1)
        sums  = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5), \
            "Class probabilities don't sum to 1"

    def test_hidden_state_carry_forward(self, model, dummy_batch):
        """Hidden state from one window should flow into the next."""
        x, static, _, _ = dummy_batch
        _, _, _, (h_n, c_n) = model(x, static, hidden=None)

        # Second forward pass with carried hidden state
        x2 = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        rul2, _, _, _ = model(x2, static, hidden=(h_n, c_n))

        # Should not crash and should produce valid output
        assert rul2.shape == (BATCH, 1)
        assert not torch.isnan(rul2).any()

    def test_single_sample_inference(self, model):
        """Model must work with batch_size=1 for real-time inference."""
        x      = torch.randn(1, SEQ_LEN, INPUT_DIM)
        static = torch.randn(1, STATIC_DIM)
        rul, logits, attn, _ = model(x, static)

        assert rul.shape    == (1, 1)
        assert logits.shape == (1, NUM_CLASSES)


# ---------------------------------------------------------------------------
# MC Dropout
# ---------------------------------------------------------------------------

class TestMCDropout:

    def test_mc_dropout_gives_variance(self, model, dummy_batch):
        """
        With dropout enabled at inference, repeated forward passes
        should give different results (non-zero variance).
        """
        x, static, _, _ = dummy_batch
        x      = x[0:1]      # single sample
        static = static[0:1]

        model.enable_dropout()

        rul_samples = []
        with torch.no_grad():
            for _ in range(20):
                rul, _, _, _ = model(x, static)
                rul_samples.append(rul.item())

        std = np.std(rul_samples)
        assert std > 0.0, \
            f"MC Dropout should give non-zero variance, got std={std}"

    def test_mc_dropout_mean_reasonable(self, model, dummy_batch):
        """MC Dropout mean should be finite."""
        x, static, _, _ = dummy_batch
        model.enable_dropout()

        rul_samples = []
        with torch.no_grad():
            for _ in range(10):
                rul, _, _, _ = model(x, static)
                rul_samples.append(rul.detach().numpy())

        mean = np.mean(rul_samples)
        assert np.isfinite(mean), f"MC Dropout mean is not finite: {mean}"


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class TestLossFunctions:

    def test_focal_loss_forward(self):
        focal = FocalLoss(gamma=2.0)
        logits  = torch.randn(BATCH, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = focal(logits, targets)

        assert loss.shape == torch.Size([]), "Loss should be scalar"
        assert loss.item() >= 0,            "Loss should be non-negative"
        assert not torch.isnan(loss),       "Loss should not be NaN"

    def test_focal_loss_with_class_weights(self):
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        focal   = FocalLoss(class_weights=weights, gamma=2.0)
        logits  = torch.randn(BATCH, NUM_CLASSES)
        targets = torch.randint(0, NUM_CLASSES, (BATCH,))
        loss = focal(logits, targets)

        assert not torch.isnan(loss), "Weighted focal loss should not be NaN"

    def test_combined_loss_components(self):
        criterion = CMAPSSLoss(alpha=0.5, gamma=2.0)
        rul_pred    = torch.randn(BATCH, 1)
        rul_target  = torch.rand(BATCH) * 125
        class_logits = torch.randn(BATCH, NUM_CLASSES)
        class_target = torch.randint(0, NUM_CLASSES, (BATCH,))

        total, mse, focal = criterion(
            rul_pred, rul_target, class_logits, class_target
        )

        assert not torch.isnan(total), "Total loss NaN"
        assert not torch.isnan(mse),   "MSE loss NaN"
        assert not torch.isnan(focal), "Focal loss NaN"
        assert mse.item()   >= 0,      "MSE should be non-negative"
        assert focal.item() >= 0,      "Focal loss should be non-negative"

    def test_combined_loss_alpha_weighting(self):
        """alpha=1.0 should give total=MSE, alpha=0.0 should give total=Focal."""
        rul_pred     = torch.randn(BATCH, 1)
        rul_target   = torch.rand(BATCH) * 125
        class_logits = torch.randn(BATCH, NUM_CLASSES)
        class_target = torch.randint(0, NUM_CLASSES, (BATCH,))

        loss_mse_only   = CMAPSSLoss(alpha=1.0)
        loss_focal_only = CMAPSSLoss(alpha=0.0)

        total_mse,   mse,   focal   = loss_mse_only(
            rul_pred, rul_target, class_logits, class_target
        )
        total_focal, mse_f, focal_f = loss_focal_only(
            rul_pred, rul_target, class_logits, class_target
        )

        assert torch.allclose(total_mse,   mse,   atol=1e-5), \
            "alpha=1.0 should give total=MSE"
        assert torch.allclose(total_focal, focal_f, atol=1e-5), \
            "alpha=0.0 should give total=Focal"

    def test_loss_decreases_with_perfect_prediction(self):
        """Loss should be lower when predictions are closer to targets."""
        criterion    = CMAPSSLoss(alpha=0.5)
        rul_target   = torch.ones(BATCH) * 50.0
        class_target = torch.zeros(BATCH, dtype=torch.long)

        # Perfect RUL prediction
        rul_perfect  = torch.ones(BATCH, 1) * 50.0
        logits_perfect = torch.zeros(BATCH, NUM_CLASSES)
        logits_perfect[:, 0] = 10.0   # high confidence class 0

        # Bad RUL prediction
        rul_bad  = torch.ones(BATCH, 1) * 125.0
        logits_bad = torch.zeros(BATCH, NUM_CLASSES)
        logits_bad[:, 3] = 10.0   # wrong class

        total_perfect, _, _ = criterion(
            rul_perfect, rul_target, logits_perfect, class_target
        )
        total_bad, _, _ = criterion(
            rul_bad, rul_target, logits_bad, class_target
        )

        assert total_perfect.item() < total_bad.item(), \
            "Perfect predictions should give lower loss than bad predictions"


# ---------------------------------------------------------------------------
# Parameter count sanity
# ---------------------------------------------------------------------------

class TestModelParameters:

    def test_parameter_count_reasonable(self, model):
        """Model shouldn't be too small or unreasonably large."""
        n_params = model.count_parameters()
        assert n_params > 10_000,     f"Model too small: {n_params:,} params"
        assert n_params < 10_000_000, f"Model too large: {n_params:,} params"

    def test_all_parameters_have_gradients(self, model, dummy_batch):
        """After a forward+backward pass, all params should have gradients."""
        x, static, y_rul, y_class = dummy_batch
        criterion = CMAPSSLoss(alpha=0.5)

        rul_pred, class_logits, _, _ = model(x, static)
        total, _, _ = criterion(rul_pred, y_rul, class_logits, y_class)
        total.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"Parameter {name} has no gradient after backward()"
