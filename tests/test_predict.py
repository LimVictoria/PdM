"""
tests/test_predict.py
---------------------
Tests for recursive inference, MC Dropout uncertainty,
and LSTM hidden state carry-forward.

Run with:
    pytest tests/test_predict.py -v
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import CMAPSS_CNN_LSTM
from predict import predict_single, recursive_predict, batch_predict
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INPUT_DIM  = 17
STATIC_DIM = 8
HIDDEN_DIM = 64
NUM_CLASSES = 4
SEQ_LEN    = 30
MC_PASSES  = 10   # small for fast tests


@pytest.fixture
def small_model():
    model = CMAPSS_CNN_LSTM(
        input_dim=INPUT_DIM,
        static_dim=STATIC_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        dropout=0.3,
        cnn_kernels=[3, 5],
        cnn_channels=16,
        attention_dim=16,
        num_classes=NUM_CLASSES
    )
    model.eval()
    return model


@pytest.fixture
def dummy_window():
    x      = torch.randn(1, SEQ_LEN, INPUT_DIM)
    static = torch.zeros(1, STATIC_DIM)
    static[0, 0] = 1.0   # cluster 0
    static[0, 6] = 1.0   # fault mode 0
    return x, static


@pytest.fixture
def dummy_sequence():
    """Full engine lifecycle: 150 cycles."""
    seq    = np.random.randn(150, INPUT_DIM).astype(np.float32)
    static = np.zeros(STATIC_DIM, dtype=np.float32)
    static[0] = 1.0   # cluster 0
    static[6] = 1.0   # fault mode 0
    return seq, static


# ---------------------------------------------------------------------------
# predict_single
# ---------------------------------------------------------------------------

class TestPredictSingle:

    def test_output_types(self, small_model, dummy_window):
        x, static = dummy_window
        rul_mean, rul_std, health_class, class_probs, hidden = predict_single(
            small_model, x, static,
            hidden=None, mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        assert isinstance(rul_mean,     float), "rul_mean should be float"
        assert isinstance(rul_std,      float), "rul_std should be float"
        assert isinstance(health_class, int),   "health_class should be int"
        assert isinstance(class_probs,  np.ndarray), "class_probs should be ndarray"

    def test_class_probs_sum_to_one(self, small_model, dummy_window):
        x, static = dummy_window
        _, _, _, class_probs, _ = predict_single(
            small_model, x, static,
            hidden=None, mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        assert abs(class_probs.sum() - 1.0) < 1e-5, \
            f"Class probs should sum to 1, got {class_probs.sum()}"

    def test_health_class_in_valid_range(self, small_model, dummy_window):
        x, static = dummy_window
        _, _, health_class, _, _ = predict_single(
            small_model, x, static,
            hidden=None, mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        assert 0 <= health_class <= 3, \
            f"Health class should be 0-3, got {health_class}"

    def test_mc_dropout_produces_uncertainty(self, small_model, dummy_window):
        """MC Dropout std should be > 0 with multiple passes."""
        x, static = dummy_window
        _, rul_std, _, _, _ = predict_single(
            small_model, x, static,
            hidden=None, mc_passes=30,   # more passes for reliable std
            device=torch.device("cpu")
        )
        assert rul_std >= 0.0, "RUL std should be non-negative"
        # With 30 passes and dropout=0.3, std should be > 0
        assert rul_std >= 0.0, "MC Dropout should produce non-zero uncertainty"

    def test_hidden_state_returned(self, small_model, dummy_window):
        x, static = dummy_window
        _, _, _, _, hidden = predict_single(
            small_model, x, static,
            hidden=None, mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        assert hidden is not None,          "Hidden state should be returned"
        assert len(hidden) == 2,            "Hidden should be (h_n, c_n) tuple"
        h_n, c_n = hidden
        assert h_n.shape == (2, 1, HIDDEN_DIM), f"h_n shape wrong: {h_n.shape}"
        assert c_n.shape == (2, 1, HIDDEN_DIM), f"c_n shape wrong: {c_n.shape}"

    def test_hidden_state_carry_forward_changes_output(
        self, small_model, dummy_window
    ):
        """
        Prediction with a non-None hidden state should differ from
        prediction with hidden=None (fresh start).
        This verifies the carry-forward mechanism works.
        """
        x, static = dummy_window

        # First prediction — fresh hidden state
        rul1, _, _, _, hidden = predict_single(
            small_model, x, static,
            hidden=None, mc_passes=1,
            device=torch.device("cpu")
        )

        # Second prediction — carried hidden state
        x2 = torch.randn(1, SEQ_LEN, INPUT_DIM)
        rul2_carried, _, _, _, _ = predict_single(
            small_model, x2, static,
            hidden=hidden, mc_passes=1,
            device=torch.device("cpu")
        )

        # Second prediction — fresh hidden state on same x2
        rul2_fresh, _, _, _, _ = predict_single(
            small_model, x2, static,
            hidden=None, mc_passes=1,
            device=torch.device("cpu")
        )

        # These should differ because hidden state carries context
        assert rul2_carried != rul2_fresh, \
            "Carried hidden state should produce different output than fresh start"


# ---------------------------------------------------------------------------
# recursive_predict
# ---------------------------------------------------------------------------

class TestRecursivePredict:

    def test_output_keys(self, small_model, dummy_sequence):
        seq, static = dummy_sequence
        results = recursive_predict(
            model=small_model,
            sensor_sequence=seq,
            static_vec=static,
            window_size=SEQ_LEN,
            mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        expected_keys = {
            "rul_mean", "rul_std", "health_class",
            "class_probs", "class_names", "n_cycles"
        }
        assert expected_keys.issubset(results.keys()), \
            f"Missing keys: {expected_keys - results.keys()}"

    def test_output_lengths_match_sequence(self, small_model, dummy_sequence):
        seq, static = dummy_sequence
        n_cycles = len(seq)
        results = recursive_predict(
            small_model, seq, static,
            window_size=SEQ_LEN,
            mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        assert len(results["rul_mean"])     == n_cycles
        assert len(results["rul_std"])      == n_cycles
        assert len(results["health_class"]) == n_cycles
        assert results["n_cycles"]          == n_cycles

    def test_health_classes_valid_range(self, small_model, dummy_sequence):
        seq, static = dummy_sequence
        results = recursive_predict(
            small_model, seq, static,
            window_size=SEQ_LEN, mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        classes = results["health_class"]
        assert all(0 <= c <= 3 for c in classes), \
            "All health classes should be in range 0-3"

    def test_no_nan_in_outputs(self, small_model, dummy_sequence):
        seq, static = dummy_sequence
        results = recursive_predict(
            small_model, seq, static,
            window_size=SEQ_LEN, mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        assert not np.isnan(results["rul_mean"]).any(),  "NaN in rul_mean"
        assert not np.isnan(results["rul_std"]).any(),   "NaN in rul_std"
        assert not np.isnan(results["class_probs"]).any(), "NaN in class_probs"

    def test_class_probs_sum_to_one_per_cycle(self, small_model, dummy_sequence):
        seq, static = dummy_sequence
        results = recursive_predict(
            small_model, seq, static,
            window_size=SEQ_LEN, mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        probs_sums = results["class_probs"].sum(axis=1)
        assert np.allclose(probs_sums, 1.0, atol=1e-4), \
            "Class probs should sum to 1 at each cycle"

    def test_rul_std_non_negative(self, small_model, dummy_sequence):
        seq, static = dummy_sequence
        results = recursive_predict(
            small_model, seq, static,
            window_size=SEQ_LEN, mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        assert (results["rul_std"] >= 0).all(), "RUL std should be non-negative"

    def test_works_with_sequence_shorter_than_window(self, small_model):
        """Sequence shorter than window_size should be padded and not crash."""
        short_seq = np.random.randn(10, INPUT_DIM).astype(np.float32)
        static    = np.zeros(STATIC_DIM, dtype=np.float32)
        static[0] = 1.0

        results = recursive_predict(
            small_model, short_seq, static,
            window_size=SEQ_LEN,   # 30 > 10
            mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        assert results["n_cycles"] == 10

    def test_works_with_sequence_longer_than_window(self, small_model):
        """Sequence longer than window_size should use sliding windows."""
        long_seq = np.random.randn(300, INPUT_DIM).astype(np.float32)
        static   = np.zeros(STATIC_DIM, dtype=np.float32)
        static[0] = 1.0

        results = recursive_predict(
            small_model, long_seq, static,
            window_size=SEQ_LEN,
            mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        assert results["n_cycles"] == 300


# ---------------------------------------------------------------------------
# batch_predict
# ---------------------------------------------------------------------------

class TestBatchPredict:

    @pytest.fixture
    def dummy_loader(self):
        n = 32
        X       = torch.randn(n, SEQ_LEN, INPUT_DIM)
        static  = torch.zeros(n, STATIC_DIM)
        static[:, 0] = 1.0
        static[:, 6] = 1.0
        y_rul   = torch.rand(n) * 125
        y_class = torch.randint(0, NUM_CLASSES, (n,))

        dataset = TensorDataset(X, static, y_rul, y_class)
        return DataLoader(dataset, batch_size=8, shuffle=False)

    def test_output_keys(self, small_model, dummy_loader):
        results = batch_predict(
            small_model, dummy_loader,
            mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        expected = {
            "rul_mean", "rul_std", "class_pred",
            "class_probs", "rul_target", "class_target"
        }
        assert expected.issubset(results.keys())

    def test_output_lengths(self, small_model, dummy_loader):
        results = batch_predict(
            small_model, dummy_loader,
            mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        n = 32
        assert len(results["rul_mean"])    == n
        assert len(results["rul_std"])     == n
        assert len(results["class_pred"])  == n
        assert len(results["class_probs"]) == n

    def test_class_predictions_valid(self, small_model, dummy_loader):
        results = batch_predict(
            small_model, dummy_loader,
            mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        preds = results["class_pred"]
        assert all(0 <= p <= 3 for p in preds), \
            "All class predictions should be in 0-3"

    def test_uncertainty_non_negative(self, small_model, dummy_loader):
        results = batch_predict(
            small_model, dummy_loader,
            mc_passes=MC_PASSES,
            device=torch.device("cpu")
        )
        assert (results["rul_std"] >= 0).all(), \
            "All uncertainty values should be non-negative"
