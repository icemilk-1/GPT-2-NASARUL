"""Utility functions for GPT4RUL.

- Reproducibility: set_seed
- Checkpoint I/O: save_checkpoint, load_checkpoint
- Visualization: plot_rul_prediction, plot_training_history

Author: Qingcheng Tan
Date: 2026-07
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
#  Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d (deterministic mode).", seed)


# =============================================================================
#  Checkpoint I/O
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_metric: float,
    filepath: Path | str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model checkpoint with optimizer state and metadata.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        epoch: Current epoch number (0-indexed).
        best_val_metric: Best validation metric so far.
        filepath: Output path for the checkpoint.
        extra: Optional extra metadata to save.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_metric": best_val_metric,
    }
    if extra:
        checkpoint["extra"] = extra
    torch.save(checkpoint, filepath)
    logger.info("Checkpoint saved → %s (epoch=%d, best_val=%.4f)", filepath, epoch, best_val_metric)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    filepath: Path | str,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load model checkpoint and restore optimizer state.

    Args:
        model: The PyTorch model to load weights into.
        optimizer: Optional optimizer to restore state.
        filepath: Path to the checkpoint file.
        device: Target device for loading.

    Returns:
        The full checkpoint dictionary.
    """
    filepath = Path(filepath)
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(
        "Checkpoint loaded ← %s (epoch=%d, best_val=%.4f)",
        filepath, checkpoint["epoch"], checkpoint["best_val_metric"],
    )
    return checkpoint


# =============================================================================
#  Visualization
# =============================================================================

def plot_rul_prediction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_id: str = "",
    save_path: Path | str | None = None,
) -> None:
    """Scatter plot: predicted RUL vs. true RUL with identity line.

    Args:
        y_true: Ground-truth RUL values.
        y_pred: Predicted RUL values.
        dataset_id: Dataset identifier for the plot title.
        save_path: If provided, save figure to this path instead of showing.
    """
    import matplotlib.pyplot as plt

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, label=f"n={len(y_true)}")
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"{dataset_id}: Predicted vs True RUL\nRMSE={rmse:.2f}, MAE={mae:.2f}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved → %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_training_history(
    history: list[dict],
    dataset_id: str = "",
    save_path: Path | str | None = None,
) -> None:
    """Plot training & validation loss and RMSE over epochs.

    Args:
        history: List of per-epoch dicts with keys train_loss, val_loss, val_rmse, lr.
        dataset_id: Dataset identifier for the plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_rmse = [h.get("val_rmse", float("nan")) for h in history]
    lrs = [h.get("lr", float("nan")) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss
    axes[0, 0].plot(epochs, train_loss, "b-", label="Train Loss", linewidth=1)
    axes[0, 0].plot(epochs, val_loss, "r-", label="Val Loss", linewidth=1)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].set_title(f"{dataset_id}: Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # RMSE
    axes[0, 1].plot(epochs, val_rmse, "g-", label="Val RMSE", linewidth=1)
    best_epoch = epochs[np.nanargmin(val_rmse)] if val_rmse else 0
    best_rmse = np.nanmin(val_rmse) if val_rmse else float("nan")
    axes[0, 1].axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5,
                       label=f"Best: epoch={best_epoch}, RMSE={best_rmse:.2f}")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].set_title(f"{dataset_id}: Val RMSE")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 0].step(epochs, lrs, "m-", where="post", linewidth=1)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title(f"{dataset_id}: LR Schedule")
    axes[1, 0].grid(True, alpha=0.3)

    # Train vs Val loss scatter
    axes[1, 1].scatter(train_loss, val_loss, c=epochs, cmap="viridis", s=15, alpha=0.7)
    axes[1, 1].set_xlabel("Train Loss")
    axes[1, 1].set_ylabel("Val Loss")
    axes[1, 1].set_title(f"{dataset_id}: Train vs Val Loss")
    cbar = fig.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label("Epoch")

    fig.suptitle(f"Training History — {dataset_id}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved → %s", save_path)
    else:
        plt.show()
    plt.close(fig)
