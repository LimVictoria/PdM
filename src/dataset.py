"""
dataset.py
----------
Full preprocessing pipeline for NASA CMAPSS dataset.
Pure RUL regression — no classification.

Pipeline:
    1. Load all 4 subsets (FD001-FD004)
    2. Select sensors by RUL correlation (drop low-correlation sensors)
    3. Compute + cap RUL at 125
    4. K-Means clustering on op_settings (fit on train only)
    5. Cluster-wise Z-score normalisation (fit on train only)
    6. Rolling statistics features (mean + std)
    7. Train/val split by engine_id
    8. Sliding window construction (size=50, stride=10)
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

OP_COLS         = ["op_setting_1", "op_setting_2", "op_setting_3"]
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

HF_REPO_ID   = "lvictoria/pdm"
CMAPSS_FILES = [
    "train_FD001.txt", "train_FD002.txt",
    "train_FD003.txt", "train_FD004.txt",
    "test_FD001.txt",  "test_FD002.txt",
    "test_FD003.txt",  "test_FD004.txt",
    "RUL_FD001.txt",   "RUL_FD002.txt",
    "RUL_FD003.txt",   "RUL_FD004.txt"
]


def download_from_huggingface(raw_dir: str = "data/raw") -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Run: pip install huggingface_hub")

    os.makedirs(raw_dir, exist_ok=True)
    missing = [f for f in CMAPSS_FILES if not (Path(raw_dir) / f).exists()]

    if not missing:
        print(f"[INFO] All CMAPSS files already in {raw_dir}/")
        return

    print(f"[INFO] Downloading {len(missing)} files from Hugging Face ({HF_REPO_ID})...")
    for filename in missing:
        print(f"  -> {filename}")
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=filename,
            repo_type="dataset", local_dir=raw_dir
        )
        target = Path(raw_dir) / filename
        if not target.exists():
            import shutil
            shutil.copy(local_path, target)

    print(f"[OK] All CMAPSS files downloaded to {raw_dir}/")


# ---------------------------------------------------------------------------
# Step 1: Load raw CMAPSS files
# ---------------------------------------------------------------------------

def load_subset(raw_dir: str, subset: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    raw_dir  = Path(raw_dir)
    train_df = pd.read_csv(raw_dir / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test_df  = pd.read_csv(raw_dir / f"test_{subset}.txt",  sep=r"\s+", header=None, names=COLUMNS)
    true_rul = pd.read_csv(raw_dir / f"RUL_{subset}.txt",   sep=r"\s+", header=None, names=["true_rul"])["true_rul"]

    fault_map = {"FD001": 0, "FD002": 0, "FD003": 1, "FD004": 1}
    for df in [train_df, test_df]:
        df["subset"]     = subset
        df["fault_mode"] = fault_map[subset]

    return train_df, test_df, true_rul


def load_all_subsets(
    raw_dir: str,
    subsets: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series]]:
    train_dfs, test_dfs, true_ruls = [], [], {}
    engine_offset = 0

    for subset in subsets:
        train_df, test_df, true_rul = load_subset(raw_dir, subset)
        train_df["engine_id"] += engine_offset
        test_df["engine_id"]  += engine_offset
        engine_offset += max(train_df["engine_id"].max(), test_df["engine_id"].max()) + 1
        train_dfs.append(train_df)
        test_dfs.append(test_df)
        true_ruls[subset] = true_rul

    return (
        pd.concat(train_dfs, ignore_index=True),
        pd.concat(test_dfs,  ignore_index=True),
        true_ruls
    )


# ---------------------------------------------------------------------------
# Step 2: Sensor selection by RUL correlation
# ---------------------------------------------------------------------------

def get_zero_variance_sensors(df: pd.DataFrame) -> List[str]:
    return [col for col in ALL_SENSOR_COLS if col in df.columns and df[col].var() == 0]


def select_sensors_by_rul_correlation(
    train_all: pd.DataFrame,
    subsets: List[str],
    min_correlation: float = 0.1
) -> List[str]:
    """
    Keep only sensors whose average absolute correlation with RUL
    meets the threshold across ALL subsets.

    Better than variance-based selection because it directly measures
    whether a sensor is predictive of remaining useful life.
    """
    sensor_cols = [c for c in train_all.columns if c in ALL_SENSOR_COLS]
    subset_corrs: Dict[str, Dict[str, float]] = {}

    for subset in subsets:
        subset_df = train_all[train_all["subset"] == subset]
        corrs = {}
        for col in sensor_cols:
            if subset_df[col].std() > 0:
                corrs[col] = abs(subset_df[col].corr(subset_df["rul"]))
            else:
                corrs[col] = 0.0
        subset_corrs[subset] = corrs

    avg_corrs = {}
    for col in sensor_cols:
        scores = [subset_corrs[s].get(col, 0.0) for s in subsets]
        avg_corrs[col] = float(np.mean(scores))

    sorted_corrs = sorted(avg_corrs.items(), key=lambda x: x[1], reverse=True)
    print(f"      Sensor correlations with RUL:")
    for col, corr in sorted_corrs:
        flag = "KEEP" if corr >= min_correlation else "DROP"
        print(f"        {col:6s}: {corr:.4f}  [{flag}]")

    selected = [col for col, corr in avg_corrs.items() if corr >= min_correlation]
    dropped  = [col for col, corr in avg_corrs.items() if corr < min_correlation]
    print(f"      Keeping {len(selected)} sensors: {sorted(selected)}")
    print(f"      Dropping {len(dropped)} sensors: {sorted(dropped)}")
    return selected


# ---------------------------------------------------------------------------
# Step 3: RUL computation
# ---------------------------------------------------------------------------

def compute_rul(df: pd.DataFrame, rul_cap: int = 125) -> pd.DataFrame:
    """Compute piecewise-linear RUL per engine and cap at rul_cap."""
    df         = df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df         = df.join(max_cycles, on="engine_id")
    df["rul"]  = (df["max_cycle"] - df["cycle"]).clip(upper=rul_cap)
    return df.drop(columns=["max_cycle"])


# ---------------------------------------------------------------------------
# Step 4: Operating condition clustering
# ---------------------------------------------------------------------------

def fit_op_clusters(df: pd.DataFrame, n_clusters: int = 2) -> KMeans:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df[OP_COLS].values)
    return kmeans


def assign_op_clusters(df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    df               = df.copy()
    df["op_cluster"] = kmeans.predict(df[OP_COLS].values)
    engine_cluster   = (
        df.groupby("engine_id")["op_cluster"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("engine_cluster")
    )
    return df.join(engine_cluster, on="engine_id")


# ---------------------------------------------------------------------------
# Step 5: Cluster-wise normalisation
# ---------------------------------------------------------------------------

def fit_cluster_scalers(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 2
) -> Dict[int, StandardScaler]:
    scalers = {}
    for cid in range(n_clusters):
        mask      = df["op_cluster"] == cid
        subset_df = df.loc[mask, feature_cols]
        scalers[cid] = StandardScaler().fit(
            subset_df if len(subset_df) > 0 else df[feature_cols]
        )
    return scalers


def apply_cluster_scalers(
    df: pd.DataFrame,
    feature_cols: List[str],
    scalers: Dict[int, StandardScaler]
) -> pd.DataFrame:
    df         = df.copy()
    normalised = df[feature_cols].values.copy().astype(np.float64)
    for cid, scaler in scalers.items():
        mask = (df["op_cluster"] == cid).values
        if mask.sum() > 0:
            normalised[mask] = scaler.transform(
                df.loc[df["op_cluster"] == cid, feature_cols]
            )
    df[feature_cols] = normalised
    return df


# ---------------------------------------------------------------------------
# Step 6: Rolling statistics
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    windows: List[int] = [5, 20]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add per-engine rolling mean and std for each sensor.
    Captures degradation rate, not just current value.
    Called AFTER normalisation so all values on same scale.
    """
    df       = df.copy().sort_values(["engine_id", "cycle"])
    new_cols = []

    for window in windows:
        for col in sensor_cols:
            mean_col = f"{col}_rmean{window}"
            std_col  = f"{col}_rstd{window}"
            grouped  = df.groupby("engine_id")[col]
            df[mean_col] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[std_col] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
            new_cols.extend([mean_col, std_col])

    print(f"      Added {len(new_cols)} rolling features "
          f"({len(windows)} windows x {len(sensor_cols)} sensors x 2 stats)")
    return df, new_cols


# ---------------------------------------------------------------------------
# Step 7: Train/val split by engine_id
# ---------------------------------------------------------------------------

def split_by_engine(
    df: pd.DataFrame,
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    engine_meta        = df.groupby("engine_id")["subset"].first().reset_index()
    engines, groups    = engine_meta["engine_id"].values, engine_meta["subset"].values
    splitter           = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=random_seed)
    train_idx, val_idx = next(splitter.split(engines, groups=groups))
    train_engines      = set(engines[train_idx])
    val_engines        = set(engines[val_idx])
    return (
        df[df["engine_id"].isin(train_engines)].copy(),
        df[df["engine_id"].isin(val_engines)].copy()
    )


# ---------------------------------------------------------------------------
# Step 8: Static feature vector
# ---------------------------------------------------------------------------

def _make_static_vec(group: pd.DataFrame) -> np.ndarray:
    """9-dim static vector: [3 cluster one-hot] + [2 fault one-hot] + [4 subset one-hot]"""
    subset_map     = {"FD001": 0, "FD002": 1, "FD003": 2, "FD004": 3}
    engine_cluster = int(group["engine_cluster"].iloc[0])
    engine_fault   = int(group["fault_mode"].iloc[0])
    subset_id      = subset_map.get(group["subset"].iloc[0], 0)
    return np.concatenate([
        np.eye(3, dtype=np.float32)[engine_cluster],
        np.eye(2, dtype=np.float32)[engine_fault],
        np.eye(4, dtype=np.float32)[subset_id]
    ])


# ---------------------------------------------------------------------------
# Step 8: Sliding window construction
# ---------------------------------------------------------------------------

def build_windows(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window_size: int = 50,
    stride: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows per engine.

    Returns:
        X:      [N, window_size, n_features]
        static: [N, static_dim]
        y_rul:  [N]
    """
    feature_cols = sensor_cols + OP_COLS
    X_list, static_list, rul_list = [], [], []

    for engine_id, group in df.groupby("engine_id"):
        group      = group.sort_values("cycle").reset_index(drop=True)
        seq        = group[feature_cols].values.astype(np.float32)
        ruls       = group["rul"].values.astype(np.float32)
        static_vec = _make_static_vec(group)
        n_cycles   = len(seq)

        for end in range(stride, n_cycles + 1, stride):
            start = end - window_size
            if start < 0:
                pad    = np.repeat(seq[0:1], -start, axis=0)
                window = np.concatenate([pad, seq[0:end]], axis=0)
            else:
                window = seq[start:end]

            X_list.append(window)
            static_list.append(static_vec)
            rul_list.append(ruls[end - 1])

    return (
        np.stack(X_list),
        np.stack(static_list),
        np.array(rul_list, dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# Step 9: Test window construction (full trajectory)
# ---------------------------------------------------------------------------

def reconstruct_test_rul(
    df: pd.DataFrame,
    true_ruls: Dict[str, pd.Series],
    rul_cap: int = 125
) -> pd.DataFrame:
    """
    Reconstruct full RUL trajectory for every test engine.

    true_rul at last cycle comes from the NASA RUL file.
    We work backwards:
        RUL at cycle t = true_rul + (max_cycle - t), clipped at rul_cap

    Not data leakage — true_rul is provided NASA ground truth,
    not derived from sensor readings.
    """
    df              = df.copy().reset_index(drop=True)
    rul_by_subset   = {s: r.values for s, r in true_ruls.items()}
    subset_counters = {s: 0 for s in true_ruls}
    rul_col         = np.zeros(len(df), dtype=np.float32)

    for engine_id, group in df.groupby("engine_id"):
        subset    = group["subset"].iloc[0]
        idx       = subset_counters[subset]
        true_r    = float(rul_by_subset[subset][idx])
        subset_counters[subset] += 1
        max_cycle   = group["cycle"].max()
        engine_ruls = (true_r + (max_cycle - group["cycle"])).clip(upper=rul_cap)
        rul_col[group.index] = engine_ruls.values

    df["rul"] = rul_col
    return df


def build_test_windows(
    df: pd.DataFrame,
    sensor_cols: List[str],
    true_ruls: Dict[str, pd.Series],
    window_size: int = 50,
    stride: int = 10,
    rul_cap: int = 125
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build test windows from full RUL trajectory reconstruction.
    Gives ~10x more test windows than single-window approach.
    """
    df           = reconstruct_test_rul(df, true_ruls, rul_cap)
    feature_cols = sensor_cols + OP_COLS
    X_list, static_list, rul_list = [], [], []

    for engine_id, group in df.groupby("engine_id"):
        group      = group.sort_values("cycle").reset_index(drop=True)
        seq        = group[feature_cols].values.astype(np.float32)
        ruls       = group["rul"].values.astype(np.float32)
        static_vec = _make_static_vec(group)
        n_cycles   = len(seq)

        for end in range(stride, n_cycles + 1, stride):
            start = end - window_size
            if start < 0:
                pad    = np.repeat(seq[0:1], -start, axis=0)
                window = np.concatenate([pad, seq[0:end]], axis=0)
            else:
                window = seq[start:end]

            X_list.append(window)
            static_list.append(static_vec)
            rul_list.append(ruls[end - 1])

    X      = np.stack(X_list)
    static = np.stack(static_list)
    y_rul  = np.array(rul_list, dtype=np.float32)

    print(f"      Test windows: {len(X):,}  "
          f"(engines: {df['engine_id'].nunique()})")
    return X, static, y_rul


# ---------------------------------------------------------------------------
# PyTorch Dataset — RUL only
# ---------------------------------------------------------------------------

class CMAPSSDataset(Dataset):
    def __init__(self, X: np.ndarray, static: np.ndarray, y_rul: np.ndarray):
        self.X      = torch.FloatTensor(X)
        self.static = torch.FloatTensor(static)
        self.y_rul  = torch.FloatTensor(y_rul)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.static[idx], self.y_rul[idx]


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_artifacts(artifacts: dict, save_dir: str = "artifacts") -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(Path(save_dir) / "preprocessing.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print(f"[INFO] Artifacts saved to {save_dir}/preprocessing.pkl")


def load_artifacts(save_dir: str = "artifacts") -> dict:
    with open(Path(save_dir) / "preprocessing.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Master preprocessing function
# ---------------------------------------------------------------------------

def preprocess(
    config_path: str = "configs/config.yaml",
    save_artifacts_dir: Optional[str] = "artifacts",
    config_override: Optional[dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Full preprocessing pipeline. Returns train/val/test DataLoaders."""
    cfg = load_config(config_path)
    if config_override:
        for section, values in config_override.items():
            if isinstance(values, dict) and section in cfg:
                cfg[section].update(values)
            else:
                cfg[section] = values

    raw_dir     = cfg["data"]["raw_dir"]
    subsets     = cfg["data"]["subsets"]
    window_size = cfg["data"]["window_size"]
    stride      = cfg["data"].get("stride", 10)
    rul_cap     = cfg["data"]["rul_cap"]
    val_split   = cfg["data"]["val_split"]
    random_seed = cfg["data"]["random_seed"]
    n_clusters  = cfg["data"]["n_clusters"]
    batch_size  = cfg["training"]["batch_size"]

    download_from_huggingface(raw_dir)

    # 1. Load
    print("[1/8] Loading CMAPSS subsets...")
    train_all, test_all, true_ruls = load_all_subsets(raw_dir, subsets)
    print(f"      Train: {len(train_all):,} rows | Test: {len(test_all):,} rows")

    # 2. Select sensors by RUL correlation
    print("[2/8] Selecting sensors by RUL correlation...")
    zero_var = get_zero_variance_sensors(train_all)
    if zero_var:
        train_all = train_all.drop(columns=zero_var)
        test_all  = test_all.drop(columns=zero_var)
        print(f"      Dropped zero-variance sensors: {zero_var}")

    # Compute RUL temporarily for correlation analysis
    train_for_sel = compute_rul(train_all, rul_cap)
    min_corr      = cfg["data"].get("min_rul_correlation", 0.1)
    sensor_cols   = select_sensors_by_rul_correlation(
        train_for_sel, subsets, min_correlation=min_corr
    )

    # 3. Compute RUL
    print("[3/8] Computing RUL targets...")
    train_all = compute_rul(train_all, rul_cap)

    # 4. Clustering
    print("[4/8] Fitting operating condition clusters...")
    kmeans    = fit_op_clusters(train_all, n_clusters)
    train_all = assign_op_clusters(train_all, kmeans)
    test_all  = assign_op_clusters(test_all,  kmeans)
    cluster_dist = train_all.groupby("engine_id")["engine_cluster"].first().value_counts().sort_index()
    print(f"      Cluster distribution: {cluster_dist.to_dict()}")

    # 5. Cluster-wise normalisation
    print("[5/8] Fitting cluster-wise scalers...")
    feature_cols = sensor_cols + OP_COLS
    scalers      = fit_cluster_scalers(train_all, feature_cols, n_clusters)
    train_all    = apply_cluster_scalers(train_all, feature_cols, scalers)
    test_all     = apply_cluster_scalers(test_all,  feature_cols, scalers)

    # 6. Rolling features
    rolling_windows = cfg["data"].get("rolling_windows", [5, 20])
    if rolling_windows:
        print("[6/8] Adding rolling statistics features...")
        train_all, rolling_cols = add_rolling_features(train_all, sensor_cols, rolling_windows)
        test_all,  _            = add_rolling_features(test_all,  sensor_cols, rolling_windows)
        sensor_cols = sensor_cols + rolling_cols
        print(f"      Total sequential features: {len(sensor_cols + OP_COLS)}")

    # 7. Train/val split
    print("[7/8] Splitting train/val by engine_id...")
    train_df, val_df = split_by_engine(train_all, val_split, random_seed)
    print(f"      Train engines: {train_df['engine_id'].nunique()} | "
          f"Val engines: {val_df['engine_id'].nunique()}")

    # 8. Sliding windows
    print("[8/8] Building windows...")
    X_tr,  s_tr,  r_tr  = build_windows(train_df, sensor_cols, window_size, stride)
    X_val, s_val, r_val = build_windows(val_df,   sensor_cols, window_size, stride)
    X_te,  s_te,  r_te  = build_test_windows(
        test_all, sensor_cols, true_ruls, window_size, stride, rul_cap
    )

    print(f"\n      Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_te):,}")
    print(f"      Window shape: {X_tr.shape}  Static shape: {s_tr.shape}")

    # DataLoaders
    train_loader = DataLoader(
        CMAPSSDataset(X_tr,  s_tr,  r_tr),
        batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        CMAPSSDataset(X_val, s_val, r_val),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        CMAPSSDataset(X_te,  s_te,  r_te),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    artifacts = {
        "kmeans":          kmeans,
        "scalers":         scalers,
        "sensor_cols":     sensor_cols,
        "feature_cols":    feature_cols,
        "window_size":     window_size,
        "rul_cap":         rul_cap,
        "n_clusters":      n_clusters,
        "rolling_windows": rolling_windows,
    }

    if save_artifacts_dir:
        save_artifacts(artifacts, save_artifacts_dir)

    print("\n[OK] Preprocessing complete.")
    return train_loader, val_loader, test_loader, artifacts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_loader, val_loader, test_loader, artifacts = preprocess()
    X, static, y_rul = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  X:       {X.shape}")
    print(f"  static:  {static.shape}")
    print(f"  y_rul:   {y_rul.shape}")
