"""
tests/test_dataset.py
---------------------
Tests for preprocessing pipeline — RUL computation, class labels,
clustering, normalisation, windowing, and data leakage prevention.

Run with:
    pytest tests/test_dataset.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import (
    compute_rul,
    compute_class_labels,
    compute_class_weights,
    get_flat_sensors,
    drop_flat_sensors,
    fit_op_clusters,
    assign_op_clusters,
    fit_cluster_scalers,
    apply_cluster_scalers,
    split_by_engine,
    build_windows,
    COLUMNS,
    ALL_SENSOR_COLS,
    OP_COLS
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_df():
    """
    Minimal synthetic CMAPSS-like DataFrame.
    3 engines, 50 cycles each.
    """
    rows = []
    for engine_id in range(1, 4):
        for cycle in range(1, 51):
            row = {
                "engine_id": engine_id,
                "cycle":     cycle,
                "op_setting_1": np.random.uniform(0, 42),
                "op_setting_2": np.random.uniform(0, 1),
                "op_setting_3": np.random.choice([100.0]),
                "subset":      "FD001",
                "fault_mode":  0,
            }
            for s in ALL_SENSOR_COLS:
                row[s] = np.random.normal(500, 20)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def multi_condition_df():
    """
    DataFrame with 6 distinct operating conditions (like FD002).
    """
    op_conditions = [
        (0.0,   0.0025, 100.0),
        (10.0,  0.25,   100.0),
        (20.0,  0.70,   100.0),
        (25.0,  0.62,   60.0),
        (35.0,  0.84,   100.0),
        (42.0,  0.84,   100.0),
    ]
    rows = []
    for engine_id in range(1, 7):
        op = op_conditions[(engine_id - 1) % 6]
        for cycle in range(1, 51):
            row = {
                "engine_id":   engine_id,
                "cycle":       cycle,
                "op_setting_1": op[0] + np.random.normal(0, 0.01),
                "op_setting_2": op[1] + np.random.normal(0, 0.001),
                "op_setting_3": op[2],
                "subset":      "FD002",
                "fault_mode":  0,
            }
            for s in ALL_SENSOR_COLS:
                row[s] = np.random.normal(500, 20)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RUL Computation
# ---------------------------------------------------------------------------

class TestComputeRUL:

    def test_rul_at_last_cycle_is_zero(self, dummy_df):
        df = compute_rul(dummy_df, rul_cap=125)
        for engine_id, group in df.groupby("engine_id"):
            last_rul = group.sort_values("cycle")["rul"].iloc[-1]
            assert last_rul == 0, \
                f"Engine {engine_id}: last cycle RUL should be 0, got {last_rul}"

    def test_rul_decreases_over_time(self, dummy_df):
        df = compute_rul(dummy_df, rul_cap=125)
        for engine_id, group in df.groupby("engine_id"):
            rul_vals = group.sort_values("cycle")["rul"].values
            # After the cap plateau, RUL should be monotonically decreasing
            # Find where cap ends (first value below cap)
            below_cap = np.where(rul_vals < 125)[0]
            if len(below_cap) > 0:
                start = below_cap[0]
                segment = rul_vals[start:]
                diffs = np.diff(segment)
                assert (diffs <= 0).all(), \
                    f"Engine {engine_id}: RUL not monotonically decreasing"

    def test_rul_cap_applied(self, dummy_df):
        rul_cap = 125
        df = compute_rul(dummy_df, rul_cap=rul_cap)
        assert (df["rul"] <= rul_cap).all(), \
            f"RUL values exceed cap of {rul_cap}"

    def test_rul_non_negative(self, dummy_df):
        df = compute_rul(dummy_df, rul_cap=125)
        assert (df["rul"] >= 0).all(), "RUL should never be negative"

    def test_custom_cap(self, dummy_df):
        df = compute_rul(dummy_df, rul_cap=60)
        assert (df["rul"] <= 60).all()


# ---------------------------------------------------------------------------
# Class Labels
# ---------------------------------------------------------------------------

class TestClassLabels:

    def test_class_values_in_valid_range(self, dummy_df):
        df = compute_rul(dummy_df)
        df = compute_class_labels(df)
        assert set(df["health_class"].unique()).issubset({0, 1, 2, 3}), \
            "Class labels should only be 0, 1, 2, 3"

    def test_class_boundaries(self):
        """Test exact boundary conditions."""
        df = pd.DataFrame({
            "engine_id": [1] * 6,
            "cycle":     [1, 2, 3, 4, 5, 6],
            "op_setting_1": [0] * 6,
            "op_setting_2": [0] * 6,
            "op_setting_3": [0] * 6,
            "subset":    ["FD001"] * 6,
            "fault_mode": [0] * 6,
        })
        df["rul"] = [126, 125, 76, 75, 26, 25]

        df = compute_class_labels(df)
        expected = [0, 1, 1, 2, 2, 3]
        assert list(df["health_class"]) == expected, \
            f"Class boundaries wrong: {list(df['health_class'])} != {expected}"

    def test_all_four_classes_present_in_sufficient_data(self, dummy_df):
        """With enough data and various RUL values, all 4 classes should appear."""
        # Create engine with long life to get all classes
        rows = []
        engine_id = 999
        for cycle in range(1, 201):
            row = {"engine_id": engine_id, "cycle": cycle,
                   "op_setting_1": 0, "op_setting_2": 0,
                   "op_setting_3": 100, "subset": "FD001", "fault_mode": 0}
            for s in ALL_SENSOR_COLS:
                row[s] = 500.0
            rows.append(row)
        long_df = pd.DataFrame(rows)
        long_df = compute_rul(long_df, rul_cap=125)
        long_df = compute_class_labels(long_df)
        assert len(long_df["health_class"].unique()) == 4, \
            "Expected all 4 health classes for a long engine life"


# ---------------------------------------------------------------------------
# Flat Sensor Detection
# ---------------------------------------------------------------------------

class TestFlatSensors:

    def test_constant_sensor_detected(self):
        """A sensor with zero variance should be detected as flat."""
        rows = []
        for i in range(100):
            row = {c: np.random.normal() for c in ALL_SENSOR_COLS}
            row["s1"] = 5.0   # constant → flat
            rows.append(row)
        df = pd.DataFrame(rows)
        flat = get_flat_sensors(df, threshold=0.001)
        assert "s1" in flat, "Constant sensor s1 should be detected as flat"

    def test_varying_sensor_not_dropped(self):
        """A sensor with clear variance should not be dropped."""
        rows = []
        for i in range(100):
            row = {c: np.random.normal(500, 20) for c in ALL_SENSOR_COLS}
            rows.append(row)
        df = pd.DataFrame(rows)
        flat = get_flat_sensors(df, threshold=0.001)
        assert "s2" not in flat, "Varying sensor s2 should not be flat"

    def test_drop_sensors_removes_columns(self, dummy_df):
        flat = ["s1", "s5", "s6"]
        df = drop_flat_sensors(dummy_df, flat)
        for s in flat:
            assert s not in df.columns, f"{s} should be dropped"
        assert "s2" in df.columns, "s2 should remain"


# ---------------------------------------------------------------------------
# Operating Condition Clustering
# ---------------------------------------------------------------------------

class TestClustering:

    def test_cluster_count(self, multi_condition_df):
        kmeans = fit_op_clusters(multi_condition_df, n_clusters=6)
        assert len(kmeans.cluster_centers_) == 6

    def test_cluster_assignment_valid_range(self, multi_condition_df):
        kmeans = fit_op_clusters(multi_condition_df, n_clusters=6)
        df = assign_op_clusters(multi_condition_df, kmeans)
        assert df["op_cluster"].between(0, 5).all(), \
            "Cluster IDs should be in range 0-5"

    def test_engine_cluster_is_single_value_per_engine(self, multi_condition_df):
        kmeans = fit_op_clusters(multi_condition_df, n_clusters=6)
        df = assign_op_clusters(multi_condition_df, kmeans)
        for engine_id, group in df.groupby("engine_id"):
            n_unique = group["engine_cluster"].nunique()
            assert n_unique == 1, \
                f"Engine {engine_id} should have 1 cluster label, got {n_unique}"

    def test_test_data_uses_train_kmeans(self, multi_condition_df):
        """
        Critical: Test data must use kmeans fitted on training data.
        This test verifies the same kmeans object is used.
        """
        train_df = multi_condition_df[
            multi_condition_df["engine_id"].isin([1, 2, 3, 4])
        ].copy()
        test_df = multi_condition_df[
            multi_condition_df["engine_id"].isin([5, 6])
        ].copy()

        kmeans = fit_op_clusters(train_df, n_clusters=6)

        # Apply same kmeans to test — should not raise
        test_df = assign_op_clusters(test_df, kmeans)
        assert "op_cluster" in test_df.columns


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class TestNormalisation:

    def test_cluster_scaler_fitted_on_train_only(self, multi_condition_df):
        """Scalers should be fitted on training data, applied to test."""
        kmeans = fit_op_clusters(multi_condition_df, n_clusters=6)
        df = assign_op_clusters(multi_condition_df, kmeans)

        feature_cols = [s for s in ALL_SENSOR_COLS
                        if s not in ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]]

        train_df = df[df["engine_id"].isin([1, 2, 3, 4])].copy()
        test_df  = df[df["engine_id"].isin([5, 6])].copy()

        scalers = fit_cluster_scalers(train_df, feature_cols, n_clusters=6)

        # Apply to test — should not raise
        test_norm = apply_cluster_scalers(test_df, feature_cols, scalers)
        assert not test_norm[feature_cols].isna().any().any(), \
            "Normalised test data should not contain NaN"

    def test_normalised_mean_near_zero(self, multi_condition_df):
        """After cluster-wise normalisation, each cluster should have ~0 mean."""
        kmeans = fit_op_clusters(multi_condition_df, n_clusters=6)
        df = assign_op_clusters(multi_condition_df, kmeans)
        feature_cols = ["s2", "s3", "s4"]

        scalers = fit_cluster_scalers(df, feature_cols, n_clusters=6)
        df_norm = apply_cluster_scalers(df, feature_cols, scalers)

        for cluster_id in range(6):
            mask = df_norm["op_cluster"] == cluster_id
            if mask.sum() > 5:
                cluster_mean = df_norm.loc[mask, "s2"].mean()
                assert abs(cluster_mean) < 0.1, \
                    f"Cluster {cluster_id} mean after normalisation: {cluster_mean:.4f}"


# ---------------------------------------------------------------------------
# Train/Val Split
# ---------------------------------------------------------------------------

class TestTrainValSplit:

    def test_no_engine_in_both_splits(self, dummy_df):
        """Critical: no engine_id should appear in both train and val."""
        df = compute_rul(dummy_df)
        df = compute_class_labels(df)
        train_df, val_df = split_by_engine(df, val_split=0.3)

        train_engines = set(train_df["engine_id"].unique())
        val_engines   = set(val_df["engine_id"].unique())
        overlap = train_engines & val_engines

        assert len(overlap) == 0, \
            f"Data leakage: engines {overlap} appear in both train and val"

    def test_split_ratio_approximate(self, dummy_df):
        df = compute_rul(dummy_df)
        df = compute_class_labels(df)
        train_df, val_df = split_by_engine(df, val_split=0.3)

        total    = dummy_df["engine_id"].nunique()
        n_val    = val_df["engine_id"].nunique()
        val_frac = n_val / total

        # Allow 20% tolerance due to small dataset
        assert 0.1 <= val_frac <= 0.5, \
            f"Val fraction {val_frac:.2f} outside expected range [0.1, 0.5]"

    def test_all_data_accounted_for(self, dummy_df):
        """Train + val should cover all engines."""
        df = compute_rul(dummy_df)
        df = compute_class_labels(df)
        train_df, val_df = split_by_engine(df, val_split=0.3)

        all_engines   = set(df["engine_id"].unique())
        train_engines = set(train_df["engine_id"].unique())
        val_engines   = set(val_df["engine_id"].unique())

        assert train_engines | val_engines == all_engines, \
            "Some engines lost during split"


# ---------------------------------------------------------------------------
# Sliding Window Construction
# ---------------------------------------------------------------------------

class TestWindowConstruction:

    @pytest.fixture
    def processed_df(self, dummy_df):
        flat = get_flat_sensors(dummy_df)
        df   = drop_flat_sensors(dummy_df, flat)
        df   = compute_rul(df, rul_cap=125)
        df   = compute_class_labels(df)
        kmeans = fit_op_clusters(df, n_clusters=3)
        df   = assign_op_clusters(df, kmeans)
        sensor_cols = [c for c in df.columns
                       if c.startswith("s") and c in ALL_SENSOR_COLS]
        feature_cols = sensor_cols + OP_COLS
        scalers = fit_cluster_scalers(df, feature_cols, n_clusters=3)
        df = apply_cluster_scalers(df, feature_cols, scalers)
        return df, sensor_cols

    def test_window_shape(self, processed_df):
        df, sensor_cols = processed_df
        X, static, y_rul, y_class = build_windows(df, sensor_cols, window_size=10)
        assert X.shape[1] == 10, f"Window size should be 10, got {X.shape[1]}"
        assert X.shape[2] == len(sensor_cols) + 3, \
            f"Feature dim wrong: {X.shape[2]}"

    def test_static_shape(self, processed_df):
        df, sensor_cols = processed_df
        X, static, y_rul, y_class = build_windows(df, sensor_cols, window_size=10)
        # static = 3 cluster one-hot + 2 fault one-hot = 5
        assert static.shape[1] == 5, \
            f"Static dim should be 5 (3 clusters + 2 fault modes), got {static.shape[1]}"

    def test_rul_values_valid(self, processed_df):
        df, sensor_cols = processed_df
        _, _, y_rul, _ = build_windows(df, sensor_cols, window_size=10)
        assert (y_rul >= 0).all(),   "RUL should be non-negative"
        assert (y_rul <= 125).all(), "RUL should not exceed cap"

    def test_class_values_valid(self, processed_df):
        df, sensor_cols = processed_df
        _, _, _, y_class = build_windows(df, sensor_cols, window_size=10)
        assert set(y_class).issubset({0, 1, 2, 3}), \
            "Class labels should be in {0, 1, 2, 3}"

    def test_padding_for_short_sequences(self, processed_df):
        """Engines with fewer cycles than window_size should be left-padded."""
        df, sensor_cols = processed_df
        # window_size=100 is larger than engine life of 50 cycles
        X, static, y_rul, y_class = build_windows(
            df, sensor_cols, window_size=100
        )
        assert X.shape[1] == 100, \
            "Windows should always be exactly window_size even with padding"

    def test_no_nan_in_windows(self, processed_df):
        df, sensor_cols = processed_df
        X, static, y_rul, y_class = build_windows(df, sensor_cols, window_size=10)
        assert not np.isnan(X).any(),      "NaN in X windows"
        assert not np.isnan(static).any(), "NaN in static features"
        assert not np.isnan(y_rul).any(),  "NaN in RUL targets"

    def test_number_of_windows_correct(self, processed_df):
        """
        With stride=1 and window_size=10:
        Each engine of length 50 produces 50 windows (with left-padding for first 9).
        """
        df, sensor_cols = processed_df
        n_engines = df["engine_id"].nunique()
        X, _, _, _ = build_windows(df, sensor_cols, window_size=10, stride=1)
        # Each engine has 50 cycles → 50 windows
        expected = n_engines * 50
        assert len(X) == expected, \
            f"Expected {expected} windows, got {len(X)}"
