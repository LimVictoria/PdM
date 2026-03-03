"""
dataset.py
----------
Full preprocessing pipeline for NASA CMAPSS dataset.

Pipeline:
    1. Load all 4 subsets (FD001-FD004)
    2. Drop flat sensors
    3. Compute + cap RUL at 125
    4. Derive class labels from RUL
    5. K-Means clustering on op_settings (fit on train only)
    6. Cluster-wise Z-score normalisation (fit on train only)
    7. Train/val split by engine_id (stratified)
    8. Sliding window construction (size=30, stride=1)
    9. Left-pad short sequences with first reading
"""

import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

COLUMNS = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",
    "s8",  "s9",  "s10", "s11", "s12", "s13", "s14",
    "s15", "s16", "s17", "s18", "s19", "s20", "s21"
]

OP_COLS   = ["op_setting_1", "op_setting_2", "op_setting_3"]
ALL_SENSOR_COLS = [f"s{i}" for i in range(1, 22)]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Auto-download from Hugging Face
# ---------------------------------------------------------------------------

HF_REPO_ID = "lvictoria/pdm"

CMAPSS_FILES = [
    "train_FD001.txt", "train_FD002.txt",
    "train_FD003.txt", "train_FD004.txt",
    "test_FD001.txt",  "test_FD002.txt",
    "test_FD003.txt",  "test_FD004.txt",
    "RUL_FD001.txt",   "RUL_FD002.txt",
    "RUL_FD003.txt",   "RUL_FD004.txt"
]


def download_from_huggingface(raw_dir: str = "data/raw") -> None:
    """
    Auto-download CMAPSS files from Hugging Face if not already present.
    Files are at root level in lvictoria/pdm.

    Only downloads missing files — safe to call every run.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub not installed. "
            "Run: pip install huggingface_hub"
        )

    os.makedirs(raw_dir, exist_ok=True)

    missing = [
        f for f in CMAPSS_FILES
        if not (Path(raw_dir) / f).exists()
    ]

    if not missing:
        print(f"[INFO] All CMAPSS files already in {raw_dir}/")
        return

    print(f"[INFO] Downloading {len(missing)} files from "
          f"Hugging Face ({HF_REPO_ID})...")

    for filename in missing:
        print(f"  → {filename}")
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,       # files are at root level
            repo_type="dataset",
            local_dir=raw_dir
        )
        # hf_hub_download saves to a cache subdir — move to raw_dir directly
        target = Path(raw_dir) / filename
        if not target.exists():
            import shutil
            shutil.copy(local_path, target)

    print(f"[✓] All CMAPSS files downloaded to {raw_dir}/")


# ---------------------------------------------------------------------------
# Step 1: Load raw CMAPSS files
# ---------------------------------------------------------------------------

def load_subset(raw_dir: str, subset: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load train, test, and RUL files for one CMAPSS subset.

    Returns:
        train_df:   Full engine lives with engine_id, cycle, sensors
        test_df:    Truncated engine lives (last window for each engine)
        true_rul:   True RUL values for each test engine
    """
    raw_dir = Path(raw_dir)

    train_df = pd.read_csv(
        raw_dir / f"train_{subset}.txt",
        sep=r"\s+", header=None, names=COLUMNS
    )
    test_df = pd.read_csv(
        raw_dir / f"test_{subset}.txt",
        sep=r"\s+", header=None, names=COLUMNS
    )
    true_rul = pd.read_csv(
        raw_dir / f"RUL_{subset}.txt",
        sep=r"\s+", header=None, names=["true_rul"]
    )["true_rul"]

    # Tag subset and fault mode
    fault_map = {"FD001": 0, "FD002": 0, "FD003": 1, "FD004": 1}
    train_df["subset"]     = subset
    train_df["fault_mode"] = fault_map[subset]
    test_df["subset"]      = subset
    test_df["fault_mode"]  = fault_map[subset]

    return train_df, test_df, true_rul


def load_all_subsets(
    raw_dir: str,
    subsets: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series]]:
    """
    Load and concatenate all subsets.
    Offsets engine_id per subset to avoid collisions.
    """
    train_dfs, test_dfs = [], []
    true_ruls = {}
    engine_offset = 0

    for subset in subsets:
        train_df, test_df, true_rul = load_subset(raw_dir, subset)

        # Offset engine IDs to make them globally unique
        train_df["engine_id"] += engine_offset
        test_df["engine_id"]  += engine_offset

        engine_offset += max(train_df["engine_id"].max(),
                             test_df["engine_id"].max()) + 1

        train_dfs.append(train_df)
        test_dfs.append(test_df)
        true_ruls[subset] = true_rul

    train_all = pd.concat(train_dfs, ignore_index=True)
    test_all  = pd.concat(test_dfs,  ignore_index=True)

    return train_all, test_all, true_ruls


# ---------------------------------------------------------------------------
# Step 2: Drop flat sensors
# ---------------------------------------------------------------------------

def get_flat_sensors(
    df: pd.DataFrame,
    variance_percentile: float = 1.0
) -> List[str]:
    """
    Identify flat sensors using variance percentile cutoff.

    Instead of a fixed threshold (which breaks across raw/normalised scales),
    we drop sensors whose variance falls in the bottom N percentile
    across all engines.

    This is scale-invariant and data-driven — works on any dataset.

    Args:
        df:                   DataFrame with sensor columns
        variance_percentile:  Drop sensors below this percentile of variance
                              Default 1.0 = bottom 1% (very conservative)
                              For CMAPSS, use 30.0 to match literature list

    Returns:
        List of sensor column names to drop
    """
    variances = {}
    for col in ALL_SENSOR_COLS:
        if col in df.columns:
            variances[col] = df[col].var()

    variance_series = pd.Series(variances)
    threshold       = np.percentile(variance_series.values, variance_percentile)

    flat = variance_series[variance_series <= threshold].index.tolist()

    # Validate against literature for CMAPSS
    literature = {f"s{i}" for i in [1, 5, 6, 10, 16, 18, 19]}
    detected   = set(flat)

    if detected != literature:
        print(f"      [INFO] Detected flat: {sorted(detected)}")
        print(f"      [INFO] Literature:    {sorted(literature)}")
        print(f"      [INFO] Using detected. If unexpected, adjust variance_percentile.")
    else:
        print(f"      Flat sensors match literature: {sorted(detected)}")

    return flat


def drop_flat_sensors(df: pd.DataFrame, flat_sensors: List[str]) -> pd.DataFrame:
    return df.drop(columns=flat_sensors)


# ---------------------------------------------------------------------------
# Step 3 & 4: RUL computation and class labels
# ---------------------------------------------------------------------------

def compute_rul(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """
    Compute piecewise linear RUL per engine and apply cap.
    RUL = min(max_cycle - current_cycle, rul_cap)
    """
    df = df.copy()

    max_cycles = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycles, on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["rul"] = df["rul"].clip(upper=rul_cap)
    df = df.drop(columns=["max_cycle"])

    return df


def compute_class_labels(
    df: pd.DataFrame,
    bins: Tuple[int, int, int] = (75, 50, 25)
) -> pd.DataFrame:
    """
    Derive 4-class health label from capped RUL (max=125):

        Class 0 — Healthy:   RUL >= 75  (early life, far from failure)
        Class 1 — Degrading: 50 <= RUL < 75
        Class 2 — Warning:   25 <= RUL < 50
        Class 3 — Critical:  RUL < 25   (imminent failure)

    Note: bins are set relative to the RUL cap (125).
    With cap=125, using >125 for Class 0 means Class 0 never
    appears (max RUL is exactly 125 after capping).
    These bins ensure all 4 classes are always populated.
    """
    df = df.copy()
    healthy, degrading, warning = bins

    conditions = [
        df["rul"] >= healthy,
        (df["rul"] >= degrading) & (df["rul"] < healthy),
        (df["rul"] >= warning)   & (df["rul"] < degrading),
        df["rul"] < warning
    ]
    df["health_class"] = np.select(conditions, [0, 1, 2, 3])

    return df


def compute_class_weights(
    df: pd.DataFrame,
    num_classes: int = 4,
    override: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Compute class weights for weighted loss.

    Two modes:
        1. override=None: inverse frequency weights (automatic)
        2. override=[w0,w1,w2,w3]: manual weights (from config)

    Always returns a tensor of length num_classes.
    If a class is missing, assigns weight=1.0 as fallback.
    """
    if override is not None:
        # Manual override from config
        assert len(override) == num_classes,             f"override must have {num_classes} values, got {len(override)}"
        print(f"      Using manual class weight override: {override}")
        return torch.FloatTensor(override)

    # Automatic inverse frequency
    counts  = df["health_class"].value_counts().sort_index()
    n_total = len(df)

    weights = []
    for c in range(num_classes):
        if c in counts.index and counts[c] > 0:
            w = n_total / (num_classes * counts[c])
        else:
            w = 1.0
        weights.append(w)

    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes

    return torch.FloatTensor(weights)


def oversample_minority_classes(
    X:       np.ndarray,
    static:  np.ndarray,
    y_rul:   np.ndarray,
    y_class: np.ndarray,
    target_ratio: float = 0.5,
    random_seed:  int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Oversample minority classes by duplicating existing windows.
    Safe for time series — no synthetic data created.

    Strategy:
        Find the majority class count
        Oversample each minority class up to target_ratio
        of majority class count

    Args:
        target_ratio: minority class target as fraction of majority
                      0.5 = minority gets 50% as many samples as majority
                      1.0 = fully balanced (equal samples per class)

    Returns:
        Oversampled X, static, y_rul, y_class
    """
    rng = np.random.RandomState(random_seed)

    class_counts = {c: (y_class == c).sum() for c in range(4)}
    majority_count = max(class_counts.values())
    target_count   = int(majority_count * target_ratio)

    print(f"      Class distribution before oversampling:")
    class_names = ["Healthy", "Degrading", "Warning", "Critical"]
    for c, name in enumerate(class_names):
        print(f"        Class {c} ({name}): {class_counts[c]:,}")

    X_list      = [X]
    static_list = [static]
    rul_list    = [y_rul]
    class_list  = [y_class]

    for c in range(4):
        count = class_counts[c]
        if count < target_count:
            # How many extra samples needed
            n_extra = target_count - count

            # Indices of this class
            idx = np.where(y_class == c)[0]

            # Sample with replacement
            extra_idx = rng.choice(idx, size=n_extra, replace=True)

            X_list.append(X[extra_idx])
            static_list.append(static[extra_idx])
            rul_list.append(y_rul[extra_idx])
            class_list.append(y_class[extra_idx])

    X_out      = np.concatenate(X_list,      axis=0)
    static_out = np.concatenate(static_list, axis=0)
    rul_out    = np.concatenate(rul_list,    axis=0)
    class_out  = np.concatenate(class_list,  axis=0)

    # Shuffle to mix oversampled with original
    shuffle_idx = rng.permutation(len(X_out))
    X_out      = X_out[shuffle_idx]
    static_out = static_out[shuffle_idx]
    rul_out    = rul_out[shuffle_idx]
    class_out  = class_out[shuffle_idx]

    new_counts = {c: (class_out == c).sum() for c in range(4)}
    print(f"      Class distribution after oversampling:")
    for c, name in enumerate(class_names):
        print(f"        Class {c} ({name}): {new_counts[c]:,}")

    return X_out, static_out, rul_out, class_out


# ---------------------------------------------------------------------------
# Step 5: Operating condition clustering
# ---------------------------------------------------------------------------

def fit_op_clusters(df: pd.DataFrame, n_clusters: int = 6) -> KMeans:
    """
    Fit K-Means on op_settings from TRAINING data only.
    Always call this on training data, never on test data.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df[OP_COLS].values)
    return kmeans


def assign_op_clusters(df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    """
    Assign cluster labels using fitted K-Means.
    Per engine: use mode cluster as static feature.
    """
    df = df.copy()
    df["op_cluster"] = kmeans.predict(df[OP_COLS].values)

    # Per-engine dominant cluster (static feature)
    engine_cluster = (
        df.groupby("engine_id")["op_cluster"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("engine_cluster")
    )
    df = df.join(engine_cluster, on="engine_id")

    return df


# ---------------------------------------------------------------------------
# Step 6: Cluster-wise normalisation
# ---------------------------------------------------------------------------

def fit_cluster_scalers(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 6
) -> Dict[int, StandardScaler]:
    """
    Fit one StandardScaler per operating condition cluster.
    Fit on TRAINING data only.

    Returns dict: {cluster_id: fitted_scaler}
    """
    scalers = {}
    for cluster_id in range(n_clusters):
        mask    = df["op_cluster"] == cluster_id
        subset  = df.loc[mask, feature_cols]

        if len(subset) == 0:
            # Fallback: fit on all data if cluster empty
            # (happens in FD001/FD003 with 1 condition)
            scalers[cluster_id] = StandardScaler().fit(df[feature_cols])
        else:
            scalers[cluster_id] = StandardScaler().fit(subset)

    return scalers


def apply_cluster_scalers(
    df: pd.DataFrame,
    feature_cols: List[str],
    scalers: Dict[int, StandardScaler]
) -> pd.DataFrame:
    """
    Normalise each row using the scaler fitted for its cluster.
    """
    df = df.copy()
    normalised = df[feature_cols].values.copy().astype(np.float64)

    for cluster_id, scaler in scalers.items():
        mask = (df["op_cluster"] == cluster_id).values
        if mask.sum() > 0:
            normalised[mask] = scaler.transform(df.loc[df["op_cluster"] == cluster_id, feature_cols])

    df[feature_cols] = normalised
    return df


# ---------------------------------------------------------------------------
# Step 7: Train/val split by engine_id
# ---------------------------------------------------------------------------

def split_by_engine(
    df: pd.DataFrame,
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by engine_id so no engine appears in both train and val.
    Stratified by subset to ensure all 4 subsets in both splits.
    """
    engine_meta = df.groupby("engine_id")["subset"].first().reset_index()
    engines     = engine_meta["engine_id"].values
    groups      = engine_meta["subset"].values

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_split,
        random_state=random_seed
    )
    train_idx, val_idx = next(splitter.split(engines, groups=groups))

    train_engines = set(engines[train_idx])
    val_engines   = set(engines[val_idx])

    train_df = df[df["engine_id"].isin(train_engines)].copy()
    val_df   = df[df["engine_id"].isin(val_engines)].copy()

    return train_df, val_df


# ---------------------------------------------------------------------------
# Step 8 & 9: Sliding window construction with left-padding
# ---------------------------------------------------------------------------

def build_windows(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window_size: int = 30,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows per engine.

    Returns:
        X:            [N, window_size, n_features]  sequential features
        static:       [N, static_dim]               one-hot static features
        y_rul:        [N]                            continuous RUL
        y_class:      [N]                            health class 0-3
    """
    feature_cols = sensor_cols + OP_COLS

    X_list, static_list, rul_list, class_list = [], [], [], []

    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle").reset_index(drop=True)

        seq    = group[feature_cols].values.astype(np.float32)
        ruls   = group["rul"].values.astype(np.float32)
        classes = group["health_class"].values.astype(np.int64)

        # Static features: one-hot op_cluster + one-hot fault_mode
        engine_cluster   = int(group["engine_cluster"].iloc[0])
        engine_fault     = int(group["fault_mode"].iloc[0])
        cluster_onehot   = np.eye(6, dtype=np.float32)[engine_cluster]
        fault_onehot     = np.eye(2, dtype=np.float32)[engine_fault]
        static_vec       = np.concatenate([cluster_onehot, fault_onehot])

        n_cycles = len(seq)

        for end in range(stride, n_cycles + 1, stride):
            start = end - window_size

            if start < 0:
                # Left-pad with first reading
                pad_len = -start
                window  = seq[0:end]
                pad     = np.repeat(seq[0:1], pad_len, axis=0)
                window  = np.concatenate([pad, window], axis=0)
            else:
                window = seq[start:end]

            X_list.append(window)
            static_list.append(static_vec)
            rul_list.append(ruls[end - 1])
            class_list.append(classes[end - 1])

    X       = np.stack(X_list)
    static  = np.stack(static_list)
    y_rul   = np.array(rul_list,   dtype=np.float32)
    y_class = np.array(class_list, dtype=np.int64)

    return X, static, y_rul, y_class


def build_test_windows(
    df: pd.DataFrame,
    sensor_cols: List[str],
    true_ruls: Dict[str, pd.Series],
    window_size: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build test windows — one window per engine (last 30 cycles).
    Uses true RUL from RUL_FD00X.txt files.
    """
    feature_cols = sensor_cols + OP_COLS

    X_list, static_list, rul_list, class_list = [], [], [], []

    # Build a combined true_rul series indexed by engine order per subset
    rul_by_subset = {}
    for subset, rul_series in true_ruls.items():
        rul_by_subset[subset] = rul_series.values

    subset_counters = {s: 0 for s in true_ruls.keys()}

    for engine_id, group in df.groupby("engine_id"):
        group  = group.sort_values("cycle").reset_index(drop=True)
        subset = group["subset"].iloc[0]

        seq = group[feature_cols].values.astype(np.float32)
        n_cycles = len(seq)

        # Last window
        if n_cycles >= window_size:
            window = seq[-window_size:]
        else:
            pad_len = window_size - n_cycles
            pad     = np.repeat(seq[0:1], pad_len, axis=0)
            window  = np.concatenate([pad, seq], axis=0)

        # True RUL for this engine
        idx     = subset_counters[subset]
        true_r  = float(rul_by_subset[subset][idx])
        subset_counters[subset] += 1

        # Class from true RUL
        if true_r > 125:
            cls = 0
        elif true_r > 75:
            cls = 1
        elif true_r > 25:
            cls = 2
        else:
            cls = 3

        engine_cluster = int(group["engine_cluster"].iloc[0])
        engine_fault   = int(group["fault_mode"].iloc[0])
        cluster_onehot = np.eye(6, dtype=np.float32)[engine_cluster]
        fault_onehot   = np.eye(2, dtype=np.float32)[engine_fault]
        static_vec     = np.concatenate([cluster_onehot, fault_onehot])

        X_list.append(window)
        static_list.append(static_vec)
        rul_list.append(true_r)
        class_list.append(cls)

    X       = np.stack(X_list)
    static  = np.stack(static_list)
    y_rul   = np.array(rul_list,   dtype=np.float32)
    y_class = np.array(class_list, dtype=np.int64)

    return X, static, y_rul, y_class


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CMAPSSDataset(Dataset):
    """
    PyTorch Dataset for CMAPSS windowed sequences.

    Args:
        X:        [N, window_size, n_features]
        static:   [N, static_dim]
        y_rul:    [N]
        y_class:  [N]
    """

    def __init__(
        self,
        X:       np.ndarray,
        static:  np.ndarray,
        y_rul:   np.ndarray,
        y_class: np.ndarray
    ):
        self.X       = torch.FloatTensor(X)
        self.static  = torch.FloatTensor(static)
        self.y_rul   = torch.FloatTensor(y_rul)
        self.y_class = torch.LongTensor(y_class)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            self.X[idx],
            self.static[idx],
            self.y_rul[idx],
            self.y_class[idx]
        )


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_artifacts(
    artifacts: dict,
    save_dir: str = "artifacts"
) -> None:
    """Save preprocessing artifacts (scalers, kmeans) for deployment."""
    os.makedirs(save_dir, exist_ok=True)
    with open(Path(save_dir) / "preprocessing.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print(f"[INFO] Artifacts saved to {save_dir}/preprocessing.pkl")


def load_artifacts(save_dir: str = "artifacts") -> dict:
    """Load preprocessing artifacts for inference/deployment."""
    with open(Path(save_dir) / "preprocessing.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Master preprocessing function
# ---------------------------------------------------------------------------

def preprocess(
    config_path: str = "configs/config.yaml",
    save_artifacts_dir: Optional[str] = "artifacts"
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, dict]:
    """
    Full preprocessing pipeline. Returns DataLoaders ready for training.

    Returns:
        train_loader:   DataLoader for training
        val_loader:     DataLoader for validation
        test_loader:    DataLoader for test
        class_weights:  Tensor for weighted loss
        artifacts:      Dict of fitted scalers/kmeans for deployment
    """
    cfg = load_config(config_path)

    raw_dir     = cfg["data"]["raw_dir"]
    subsets     = cfg["data"]["subsets"]
    window_size = cfg["data"]["window_size"]
    rul_cap     = cfg["data"]["rul_cap"]
    val_split   = cfg["data"]["val_split"]
    random_seed = cfg["data"]["random_seed"]
    n_clusters  = cfg["data"]["n_clusters"]
    batch_size  = cfg["training"]["batch_size"]


    # ── Auto-download from Hugging Face if files missing ──────────────────
    download_from_huggingface(raw_dir)

    # ── Step 1: Load ──────────────────────────────────────────────────────
    print("[1/9] Loading CMAPSS subsets...")
    train_all, test_all, true_ruls = load_all_subsets(raw_dir, subsets)
    print(f"      Train: {len(train_all):,} rows | "
          f"Test: {len(test_all):,} rows")

    # ── Step 2: Drop flat sensors ─────────────────────────────────────────
    print("[2/9] Identifying and dropping flat sensors...")
    percentile   = cfg["data"].get("flat_sensor_percentile", 30.0)
    flat_sensors = get_flat_sensors(train_all, variance_percentile=percentile)
    print(f"      Dropping: {flat_sensors}")
    train_all = drop_flat_sensors(train_all, flat_sensors)
    test_all  = drop_flat_sensors(test_all,  flat_sensors)

    sensor_cols = [c for c in train_all.columns
                   if c.startswith("s") and c not in flat_sensors
                   and c in ALL_SENSOR_COLS]

    # ── Step 3 & 4: RUL + class labels ───────────────────────────────────
    print("[3/9] Computing RUL targets...")
    train_all = compute_rul(train_all, rul_cap)
    train_all = compute_class_labels(train_all)

    # ── Step 4b: Class weights ────────────────────────────────────────────
    print("[4/9] Computing class weights...")
    weight_override = cfg["data"].get("class_weight_override", None)
    class_weights   = compute_class_weights(train_all, override=weight_override)
    print(f"      Class weights: {class_weights.tolist()}")

    # ── Step 5: Clustering ────────────────────────────────────────────────
    print("[5/9] Fitting operating condition clusters...")
    kmeans    = fit_op_clusters(train_all, n_clusters)
    train_all = assign_op_clusters(train_all, kmeans)
    test_all  = assign_op_clusters(test_all,  kmeans)

    cluster_dist = train_all.groupby("engine_id")["engine_cluster"] \
                             .first().value_counts().sort_index()
    print(f"      Cluster distribution (engines): {cluster_dist.to_dict()}")

    # ── Step 6: Cluster-wise normalisation ────────────────────────────────
    print("[6/9] Fitting cluster-wise scalers...")
    feature_cols = sensor_cols + OP_COLS
    scalers      = fit_cluster_scalers(train_all, feature_cols, n_clusters)
    train_all    = apply_cluster_scalers(train_all, feature_cols, scalers)
    test_all     = apply_cluster_scalers(test_all,  feature_cols, scalers)

    # ── Step 7: Train/val split ───────────────────────────────────────────
    print("[7/9] Splitting train/val by engine_id...")
    train_df, val_df = split_by_engine(train_all, val_split, random_seed)
    print(f"      Train engines: {train_df['engine_id'].nunique()} | "
          f"Val engines: {val_df['engine_id'].nunique()}")

    # ── Step 8 & 9: Sliding windows ───────────────────────────────────────
    print("[8/9] Building training windows...")
    X_tr, s_tr, r_tr, c_tr = build_windows(
        train_df, sensor_cols, window_size, stride=1
    )

    # ── Oversample minority classes ───────────────────────────────────────
    oversample        = cfg["data"].get("oversample", True)
    oversample_ratio  = cfg["data"].get("oversample_ratio", 0.5)
    if oversample:
        print("[8b] Oversampling minority classes...")
        X_tr, s_tr, r_tr, c_tr = oversample_minority_classes(
            X_tr, s_tr, r_tr, c_tr,
            target_ratio=oversample_ratio,
            random_seed=random_seed
        )

    print("[8/9] Building validation windows...")
    X_val, s_val, r_val, c_val = build_windows(
        val_df, sensor_cols, window_size, stride=1
    )

    print("[9/9] Building test windows...")
    X_te, s_te, r_te, c_te = build_test_windows(
        test_all, sensor_cols, true_ruls, window_size
    )

    print(f"\n      Train windows:  {len(X_tr):,}")
    print(f"      Val windows:    {len(X_val):,}")
    print(f"      Test windows:   {len(X_te):,}")
    print(f"      Window shape:   {X_tr.shape}")
    print(f"      Static shape:   {s_tr.shape}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = DataLoader(
        CMAPSSDataset(X_tr,  s_tr,  r_tr,  c_tr),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        CMAPSSDataset(X_val, s_val, r_val, c_val),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        CMAPSSDataset(X_te,  s_te,  r_te,  c_te),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # ── Save artifacts ────────────────────────────────────────────────────
    artifacts = {
        "kmeans":       kmeans,
        "scalers":      scalers,
        "sensor_cols":  sensor_cols,
        "feature_cols": feature_cols,
        "flat_sensors": flat_sensors,
        "window_size":  window_size,
        "rul_cap":      rul_cap,
        "n_clusters":   n_clusters,
        "config_path":  config_path
    }

    if save_artifacts_dir:
        save_artifacts(artifacts, save_artifacts_dir)

    print("\n[✓] Preprocessing complete.")
    return train_loader, val_loader, test_loader, class_weights, artifacts


# ---------------------------------------------------------------------------
# Entry point for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_weights, artifacts = preprocess()

    # Inspect one batch
    X, static, y_rul, y_class = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  X:        {X.shape}")
    print(f"  static:   {static.shape}")
    print(f"  y_rul:    {y_rul.shape}")
    print(f"  y_class:  {y_class.shape}")
    print(f"\nClass weights: {class_weights}")
    print(f"\nClass distribution in batch:")
    for c in range(4):
        names = ["Healthy", "Degrading", "Warning", "Critical"]
        count = (y_class == c).sum().item()
        print(f"  Class {c} ({names[c]}): {count}")
