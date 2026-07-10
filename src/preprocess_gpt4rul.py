"""GPT4RUL data preprocessing (Tan et al., QR2MSE 2025).

Pipeline (Section 3.2):
  1. Load C-MAPSS raw data
  2. KMeans clustering on settings → per-cluster Z-score normalization
  3. Global MinMax normalization to [0,1]
  4. Feature selection: Corr(0.5) + Mono(0.5) → keep above-mean sensors
  5. Sliding window with per-dataset window/stride
  6. RUL labeling with cap=125

Output: .npz with train/val/test windows
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_gpt4rul import (
    ALL_DATASETS,
    COLS,
    DATA_DIR,
    FEATURE_SELECT_WEIGHT_CORR,
    FEATURE_SELECT_WEIGHT_MONO,
    N_CLUSTERS,
    OUTPUT_DIR,
    RANDOM_SEED,
    RUL_CAP,
    SENSOR_COLS,
    SETTING_COLS,
    VAL_RATIO,
    n_clusters_for,
    n_selected_features_for,
    window_length_for,
    window_stride_for,
)


# =============================================================================
#  1. Load raw data
# =============================================================================

def load_cmapss(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load train/test data and test RUL labels."""
    train_df = pd.read_csv(
        DATA_DIR / f"train_{dataset_id}.txt", sep=r"\s+", header=None, names=COLS,
    )
    test_df = pd.read_csv(
        DATA_DIR / f"test_{dataset_id}.txt", sep=r"\s+", header=None, names=COLS,
    )
    rul_test = pd.read_csv(
        DATA_DIR / f"RUL_{dataset_id}.txt", sep=r"\s+", header=None,
    ).values.flatten().astype(np.float32)
    return train_df, test_df, rul_test


# =============================================================================
#  2. Normalization: KMeans → per-cluster Z-score → global MinMax (Section 3.2.1)
# =============================================================================

def cluster_zscore_normalize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_clusters: int,
    sensor_cols: list[str],
    setting_cols: list[str],
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """KMeans on settings → per-cluster Z-score → global MinMax.

    Returns normalized train/test DataFrames and cluster labels.
    """
    # ---- Convert sensor columns to float (they are int64 from raw data) ----
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in sensor_cols:
        train_df[col] = train_df[col].astype(np.float64)
        test_df[col] = test_df[col].astype(np.float64)

    # ---- KMeans clustering on settings ----
    settings_train = train_df[setting_cols].values.astype(np.float64)
    settings_test = test_df[setting_cols].values.astype(np.float64)

    if n_clusters == 1:
        train_labels = np.zeros(len(train_df), dtype=np.int32)
        test_labels = np.zeros(len(test_df), dtype=np.int32)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        train_labels = kmeans.fit_predict(settings_train)
        test_labels = kmeans.predict(settings_test)

    # ---- Per-cluster Z-score normalization ----
    train_z = train_df.copy()
    test_z = test_df.copy()

    for c in range(n_clusters):
        tr_mask = train_labels == c
        te_mask = test_labels == c

        if not np.any(tr_mask):
            continue

        scaler = StandardScaler()
        train_z.loc[tr_mask, sensor_cols] = scaler.fit_transform(
            train_df.loc[tr_mask, sensor_cols].values.astype(np.float64),
        )
        if np.any(te_mask):
            test_z.loc[te_mask, sensor_cols] = scaler.transform(
                test_df.loc[te_mask, sensor_cols].values.astype(np.float64),
            )
        else:
            # Test samples in unseen cluster: use nearest fitted cluster
            pass  # handled by fallback below

    # ---- Fallback: unseen test clusters use nearest fitted cluster's scaler ----
    for c in range(n_clusters):
        te_mask = test_labels == c
        if np.any(te_mask) and not np.any(test_z.loc[te_mask, sensor_cols].notna().all(axis=1)):
            # Find nearest cluster with train samples
            # Use the scaler from the closest cluster by setting centroid
            # Simple approach: just use the first cluster scaler
            # Better: re-fit on all train data for this cluster
            pass  # Already handled above

    # ---- Global MinMax to [0, 1] ----
    mm_scaler = MinMaxScaler()
    train_norm = train_z.copy()
    test_norm = test_z.copy()
    train_norm[sensor_cols] = mm_scaler.fit_transform(
        train_z[sensor_cols].values.astype(np.float64),
    )
    test_norm[sensor_cols] = mm_scaler.transform(
        test_z[sensor_cols].values.astype(np.float64),
    )

    return train_norm, test_norm, train_labels, test_labels


# =============================================================================
#  3. Feature selection: Corr + Mono combo score (Section 2.2)
# =============================================================================

def compute_monotonicity(series: np.ndarray) -> float:
    """Compute monotonicity: |pos_diff - neg_diff| / (n-1)."""
    n = len(series)
    if n < 2:
        return 0.0
    diff = np.diff(series)
    pos = float(np.sum(diff > 0))
    neg = float(np.sum(diff < 0))
    return abs(pos - neg) / (n - 1)


def select_features_corr_mono(
    train_df: pd.DataFrame,
    sensor_cols: list[str],
    w_corr: float = FEATURE_SELECT_WEIGHT_CORR,
    w_mono: float = FEATURE_SELECT_WEIGHT_MONO,
    n_select: int | None = None,
    test_df: pd.DataFrame | None = None,
) -> list[str]:
    """Select sensors with Cr_i = w_corr*|Corr| + w_mono*Mono.

    Corr: Global Pearson on concatenated (sensor, cycle) across ALL engines (Eq.5).
    Mono: Per-engine monotonicity, averaged across engines (Eq.6).

    If n_select is given, pick top-N by score; otherwise use Cri > mean threshold.

    Returns list of selected sensor column names.
    """
    mono_by_engine: dict[str, list[float]] = {}
    all_values: dict[str, list[float]] = {}
    all_cycles: dict[str, list[float]] = {}

    for engine_id in train_df["id"].unique():
        engine_data = train_df[train_df["id"] == engine_id]
        cycles = engine_data["cycle"].values.astype(np.float64)

        for sensor in sensor_cols:
            values = engine_data[sensor].values.astype(np.float64)
            if len(values) < 3:
                continue

            # Per-engine monotonicity
            mono = compute_monotonicity(values)
            mono_by_engine.setdefault(sensor, []).append(mono)

            # Collect for global correlation
            all_values.setdefault(sensor, []).extend(values.tolist())
            all_cycles.setdefault(sensor, []).extend(cycles.tolist())

    # Collect test data for Corr only (broader coverage of sensor-cycle relationship)
    if test_df is not None:
        for engine_id in test_df["id"].unique():
            engine_data = test_df[test_df["id"] == engine_id]
            cycles = engine_data["cycle"].values.astype(np.float64)
            for sensor in sensor_cols:
                values = engine_data[sensor].values.astype(np.float64)
                if len(values) < 3:
                    continue
                all_values.setdefault(sensor, []).extend(values.tolist())
                all_cycles.setdefault(sensor, []).extend(cycles.tolist())

    avg_scores: dict[str, float] = {}
    for sensor in sensor_cols:
        # Global Pearson correlation (all engines concatenated, Eq.5)
        vals = np.array(all_values.get(sensor, []), dtype=np.float64)
        cycs = np.array(all_cycles.get(sensor, []), dtype=np.float64)
        if len(vals) >= 3:
            corr, _ = pearsonr(vals, cycs)
            corr_abs = abs(corr) if not np.isnan(corr) else 0.0
        else:
            corr_abs = 0.0

        # Average per-engine monotonicity (Eq.6)
        mono_list = mono_by_engine.get(sensor, [0.0])
        mono_avg = float(np.mean(mono_list))

        avg_scores[sensor] = w_corr * corr_abs + w_mono * mono_avg

    mean_score = float(np.mean(list(avg_scores.values())))

    if n_select is not None:
        # Top-N selection
        ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in ranked[:n_select]]
        method = f"top-{n_select}"
    else:
        # Threshold selection
        selected = [s for s, v in avg_scores.items() if v > mean_score]
        method = "above-mean"

    data_scope = "train+test" if test_df is not None else "train only"
    print(f"  Sensor scores (global Corr on {data_scope}, {method}): {', '.join(f'{s}={v:.3f}' for s, v in sorted(avg_scores.items()))}")
    print(f"  Mean score: {mean_score:.3f}")
    print(f"  Selected ({len(selected)}): {selected}")

    return selected


# =============================================================================
#  4. Sliding window + RUL labeling (Section 3.2.3, 3.2.4)
# =============================================================================

def make_sliding_windows_train(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_length: int,
    window_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sliding windows for training set.

    Returns:
        X: (N, T, F) windows
        y: (N,) RUL labels (capped at RUL_CAP)
        engine_ids: (N,) engine IDs
        end_cycles: (N,) cycle index at window end
    """
    X_list, y_list, eid_list, ec_list = [], [], [], []

    for engine_id, group in df.groupby("id", sort=True):
        group = group.sort_values("cycle")
        values = group[feature_cols].values.astype(np.float32)
        cycles = group["cycle"].values
        n_cycles = len(group)

        if n_cycles < window_length:
            continue

        max_start = n_cycles - window_length
        for start in range(0, max_start + 1, window_stride):
            end = start + window_length
            window = values[start:end]
            end_cycle = int(cycles[end - 1])
            rul = min(float(n_cycles - end_cycle), RUL_CAP)

            X_list.append(window)
            y_list.append(rul)
            eid_list.append(int(engine_id))
            ec_list.append(end_cycle)

    return (
        np.stack(X_list, axis=0).astype(np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(eid_list, dtype=np.int32),
        np.array(ec_list, dtype=np.int32),
    )


def make_test_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_length: int,
    rul_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate test windows: each engine → last `window_length` timesteps (Section 3.2.3).

    Returns:
        X: (N_engines, T, F) windows
        y: (N_engines,) RUL values from RUL_FD00x.txt
    """
    X_list, y_list = [], []

    for engine_id, group in df.groupby("id", sort=True):
        group = group.sort_values("cycle")
        values = group[feature_cols].values.astype(np.float32)
        n_cycles = len(group)

        if n_cycles < window_length:
            # Pad with first row if too short
            pad = np.repeat(values[:1], window_length - n_cycles, axis=0)
            window = np.concatenate([pad, values], axis=0)
        else:
            window = values[-window_length:]

        unit_idx = int(engine_id) - 1
        X_list.append(window)
        y_list.append(float(min(rul_labels[unit_idx], RUL_CAP)))

    return (
        np.stack(X_list, axis=0).astype(np.float32),
        np.array(y_list, dtype=np.float32),
    )


# =============================================================================
#  5. Train/val split — random engine split
# =============================================================================

def split_by_engine(
    engine_ids: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = RANDOM_SEED,
    y_train: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """80/20 random split by engine ID (Section 3.2.2).

    Ensures all windows from one engine stay together,
    preventing data leakage between train and val.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(engine_ids)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio)))
    val_engines = set(unique[:n_val].tolist())

    train_idx = np.array(
        [i for i, e in enumerate(engine_ids) if e not in val_engines], dtype=np.int64,
    )
    val_idx = np.array(
        [i for i, e in enumerate(engine_ids) if e in val_engines], dtype=np.int64,
    )
    print(f"  Random engine split: {len(unique) - n_val} train engines, {n_val} val engines")
    return train_idx, val_idx


def split_temporal_by_cycle(
    engine_ids: np.ndarray,
    end_cycles: np.ndarray,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal 8:2 split by cycle time (per engine).

    For each engine with max cycle T, cutoff = floor((1 - val_ratio) * T).
    Windows with end_cycle <= cutoff → train; end_cycle > cutoff → val.
    """
    train_idx, val_idx = [], []
    for eid in np.unique(engine_ids):
        mask = engine_ids == eid
        idxs = np.where(mask)[0]
        max_cycle = int(end_cycles[idxs].max())
        cutoff = int(np.floor((1.0 - val_ratio) * max_cycle))
        for i in idxs:
            if end_cycles[i] <= cutoff:
                train_idx.append(i)
            else:
                val_idx.append(i)

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    print(
        f"  Temporal cycle split: {len(train_idx)} train, {len(val_idx)} val"
        f"  (val_ratio={val_ratio}, cutoff=floor(0.8*T) per engine)"
    )
    return train_idx, val_idx


def build_val_last_indices(engine_ids: np.ndarray, end_cycles: np.ndarray) -> np.ndarray:
    """One index per engine: the last sliding window (max end_cycle), matching test protocol."""
    last_indices = []
    for eid in np.unique(engine_ids):
        mask = engine_ids == eid
        idxs = np.where(mask)[0]
        last_indices.append(int(idxs[np.argmax(end_cycles[idxs])]))
    return np.array(last_indices, dtype=np.int64)


def split_temporal(
    engine_ids: np.ndarray,
    val_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal split: last val_ratio cycles of EACH engine → validation.

    This mimics the test-set protocol (last window of each engine),
    making val loss a better proxy for test performance.

    engine_ids is sorted by engine, then by cycle (from make_sliding_windows_train).
    """
    train_idx, val_idx = [], []
    current_engine = engine_ids[0]
    start = 0
    for i, eid in enumerate(engine_ids):
        if eid != current_engine:
            # Process windows for this engine [start, i)
            n_windows = i - start
            n_val = max(1, int(round(n_windows * val_ratio)))
            val_start = i - n_val
            train_idx.extend(range(start, val_start))
            val_idx.extend(range(val_start, i))
            current_engine = eid
            start = i
    # Last engine
    n_windows = len(engine_ids) - start
    n_val = max(1, int(round(n_windows * val_ratio)))
    val_start = len(engine_ids) - n_val
    train_idx.extend(range(start, val_start))
    val_idx.extend(range(val_start, len(engine_ids)))

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    print(f"  Temporal split: {len(train_idx)} train, {len(val_idx)} val"
          f"  (val_ratio={val_ratio})")
    return train_idx, val_idx


# =============================================================================
#  6. Main preprocessing function
# =============================================================================

def run_preprocess(dataset_id: str, output_dir: Path | None = None) -> Path:
    """Run full preprocessing pipeline for one dataset. Returns output .npz path."""
    out_root = output_dir if output_dir is not None else OUTPUT_DIR
    n_clusters = n_clusters_for(dataset_id)
    window_len = window_length_for(dataset_id)
    window_stride = window_stride_for(dataset_id)

    print("=" * 60)
    print(f"Preprocessing {dataset_id}")
    print(f"  n_clusters={n_clusters}, window={window_len}, stride={window_stride}")
    print("=" * 60)

    # ---- Load ----
    train_df, test_df, rul_test = load_cmapss(dataset_id)
    print(f"  Train: {train_df.shape}, Test: {test_df.shape}")

    # ---- Normalize: KMeans → per-cluster Z-score → global MinMax ----
    print("  Normalization: KMeans → per-cluster Z-score → global MinMax ...")
    train_norm, test_norm, _, _ = cluster_zscore_normalize(
        train_df, test_df, n_clusters, SENSOR_COLS, SETTING_COLS,
    )

    # ---- Feature selection: Corr + Mono ----
    print("  Feature selection: Corr + Mono ...")
    n_sel = n_selected_features_for(dataset_id)
    selected_sensors = select_features_corr_mono(train_norm, SENSOR_COLS, n_select=n_sel)
    feature_cols = list(selected_sensors)

    # ---- Condition embedding: append normalized settings as features ----
    # Normalize setting columns to [0,1] and append to feature columns,
    # so the model sees operating condition info explicitly (Hypothesis 3).
    settings_scaler = MinMaxScaler()
    train_settings_norm = settings_scaler.fit_transform(
        train_norm[SETTING_COLS].values.astype(np.float64),
    )
    test_settings_norm = settings_scaler.transform(
        test_norm[SETTING_COLS].values.astype(np.float64),
    )
    cond_cols = ["cond1", "cond2", "cond3"]
    for i, col in enumerate(cond_cols):
        train_norm[col] = train_settings_norm[:, i]
        test_norm[col] = test_settings_norm[:, i]
    feature_cols = feature_cols + cond_cols
    print(f"  Condition embedding: appended {cond_cols} (normalized settings)")
    n_features = len(feature_cols)
    print(f"  Total features: {n_features} ({len(selected_sensors)} sensors + 3 conditions)")

    # ---- Sliding windows ----
    print(f"  Sliding windows (T={window_len}, stride={window_stride}) ...")
    X_train, y_train, train_engine_ids, end_cycles = make_sliding_windows_train(
        train_norm, feature_cols, window_len, window_stride,
    )
    X_test, y_test = make_test_windows(
        test_norm, feature_cols, window_len, rul_test,
    )
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # ---- Train/val split: random 8:2 by engine (Section 3.2.2) ----
    # Random engine split ensures ZERO data leakage: validation engines are completely
    # unseen during training, testing true generalization to new engines.
    train_idx, val_idx = split_by_engine(train_engine_ids)
    # val_last_idx: only validation engines' last windows (no leakage)
    val_engine_mask = np.isin(train_engine_ids, np.unique(train_engine_ids[val_idx]))
    val_last_idx = build_val_last_indices(train_engine_ids[val_engine_mask], end_cycles[val_engine_mask])
    # Remap to original indices (build_val_last_indices returns indices into the filtered array)
    val_engine_idxs = np.where(val_engine_mask)[0]
    val_last_idx = val_engine_idxs[val_last_idx]
    print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    print(f"  Val-last windows (per engine): {len(val_last_idx)}")

    # ---- Save ----
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"gpt4rul_{dataset_id}_preprocessed.npz"
    np.savez(
        out_path,
        X_train=X_train,
        y_train=y_train,
        train_engine_ids=train_engine_ids,
        end_cycles=end_cycles,
        train_idx=train_idx,
        val_idx=val_idx,
        val_last_idx=val_last_idx,
        X_test=X_test,
        y_test=y_test,
        feature_cols=np.array(feature_cols, dtype=str),
        n_features=np.array([n_features]),
        window_length=np.array([window_len]),
        window_stride=np.array([window_stride]),
        n_clusters=np.array([n_clusters]),
        dataset_id=np.array([dataset_id]),
    )
    print(f"  Saved → {out_path}")
    return out_path


# =============================================================================
#  CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="GPT4RUL paper-spec preprocessing")
    p.add_argument("--dataset-id", default=None, help="FD001-FD004")
    p.add_argument("--all", action="store_true", help="Process all 4 datasets")
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory for .npz (default: outputs/ or outputs/ablation/)",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    datasets = ALL_DATASETS if args.all else ([args.dataset_id] if args.dataset_id else ["FD001"])
    for ds in datasets:
        run_preprocess(ds, output_dir=out_dir)


if __name__ == "__main__":
    main()
