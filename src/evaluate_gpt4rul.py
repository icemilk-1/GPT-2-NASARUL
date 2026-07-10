"""Standalone evaluation script for GPT4RUL.

Load a trained checkpoint, run inference on the test set, compute metrics
(RMSE, MAE, asymmetric scoring function), and generate visualization plots.

Usage:
    python src/evaluate_gpt4rul.py --dataset-id FD001
    python src/evaluate_gpt4rul.py --dataset-id FD002 --checkpoint outputs/checkpoints/best_gpt4rul_FD002.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_gpt4rul import (
    CHECKPOINT_DIR,
    DROPOUT,
    GPT2_HIDDEN_DIM,
    GPT2_MODEL_NAME,
    GPT2_N_LAYERS,
    OUTPUT_DIR,
    PATCH_SIZE,
    PCF_HIDDEN_DIM,
    PCF_STYLE,
    RANDOM_SEED,
    RESULTS_DIR,
    RUL_CAP,
    batch_size_for,
    patch_stride_for,
    pcf_mixing_factor_for,
    pcf_n_blocks_for,
    window_length_for,
)
from src.model_gpt4rul import GPT4RUL
from src.utils import plot_rul_prediction, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 10.0) - 1.0, np.exp(d / 13.0) - 1.0)
    return float(np.sum(scores))


def evaluate(
    dataset_id: str,
    checkpoint_path: Path | str | None = None,
    device: str | None = None,
    save_plot: bool = True,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    set_seed(RANDOM_SEED)

    preprocessed_path = OUTPUT_DIR / f"gpt4rul_{dataset_id}_preprocessed.npz"
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {preprocessed_path}. Run preprocess first."
        )

    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / f"best_gpt4rul_{dataset_id}.pt"
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data = np.load(preprocessed_path, allow_pickle=True)
    n_features = int(data["n_features"])
    window_length = int(data["window_length"])
    X_test = torch.from_numpy(data["X_test"].astype(np.float32))
    y_test = torch.from_numpy(data["y_test"].astype(np.float32))

    logger.info(
        "Dataset: %s | n_features=%d | window=%d | test samples=%d",
        dataset_id, n_features, window_length, len(X_test),
    )

    model = GPT4RUL(
        n_features=n_features,
        window_length=window_length,
        patch_size=PATCH_SIZE,
        patch_stride=patch_stride_for(dataset_id),
        pcf_hidden_dim=PCF_HIDDEN_DIM,
        n_pcf_blocks=pcf_n_blocks_for(dataset_id),
        pcf_mixing_factor=pcf_mixing_factor_for(dataset_id),
        pcf_dropout=DROPOUT,
        pcf_style=PCF_STYLE,
        gpt2_model_name=GPT2_MODEL_NAME,
        gpt2_n_layers=GPT2_N_LAYERS,
        gpt2_hidden_dim=GPT2_HIDDEN_DIM,
        freeze_gpt2=True,
        pooling="flatten",
        use_gpt2_residual=True,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    best_val = checkpoint.get("val_rmse", checkpoint.get("val_loss", checkpoint.get("best_val_metric", float("nan"))))
    logger.info(
        "Model loaded from %s (epoch=%s, val=%.4f)",
        checkpoint_path, checkpoint.get("epoch", "?"), best_val,
    )

    bs = batch_size_for(dataset_id)
    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=bs, shuffle=False)

    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.append(out.cpu().numpy())
            trues.append(y.numpy())

    y_pred = np.clip(np.concatenate(preds), 0, RUL_CAP)
    y_true = np.concatenate(trues)

    rmse = compute_rmse(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    score = compute_score(y_true, y_pred)
    logger.info("Results — RMSE=%.4f | MAE=%.4f | Score=%.2f", rmse, mae, score)

    if save_plot:
        figures_dir = ROOT / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_path = figures_dir / f"prediction_{dataset_id}.png"
        plot_rul_prediction(y_true, y_pred, dataset_id=dataset_id, save_path=plot_path)

    result = {
        "dataset_id": dataset_id,
        "checkpoint": str(checkpoint_path),
        "test_rmse": rmse,
        "test_mae": mae,
        "test_score": score,
        "n_test": len(y_true),
    }

    out_path = RESULTS_DIR / f"eval_{dataset_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Evaluation results saved → %s", out_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="GPT4RUL Evaluation")
    parser.add_argument("--dataset-id", required=True, choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    evaluate(args.dataset_id, args.checkpoint, args.device, save_plot=not args.no_plot)


if __name__ == "__main__":
    main()
