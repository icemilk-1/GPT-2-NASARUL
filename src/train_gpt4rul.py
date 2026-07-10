"""GPT4RUL training script — strict paper reproduction (Tan et al., QR2MSE 2025)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_gpt4rul import (
    ALL_DATASETS,
    ABLATION_OUTPUT_DIR,
    ABLATION_RESULTS_DIR,
    CHECKPOINT_DIR,
    DROPOUT,
    EARLY_STOP_METRIC,
    EARLY_STOP_PATIENCE,
    GPT2_FREEZE,
    GPT2_HIDDEN_DIM,
    GPT2_MODEL_NAME,
    GPT2_N_LAYERS,
    GRAD_CLIP,
    LEARNING_RATE,
    LR_DECAY_EPOCHS,
    LR_DECAY_FACTOR,
    MAX_EPOCHS,
    OUTPUT_DIR,
    PATCH_SIZE,
    PCF_HIDDEN_DIM,
    PCF_STYLE,
    RANDOM_SEED,
    RESULTS_DIR,
    RUL_CAP,
    VAL_MODE,
    WEIGHT_DECAY,
    batch_size_for,
    patch_stride_for,
    pcf_mixing_factor_for,
    pcf_n_blocks_for,
    window_length_for,
)
from src.model_gpt4rul import GPT4RUL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    output_dir: Path = OUTPUT_DIR
    results_dir: Path = RESULTS_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
    val_mode: str = VAL_MODE
    early_stop_metric: str = EARLY_STOP_METRIC
    pcf_style: str = PCF_STYLE
    grad_clip: float = GRAD_CLIP
    tag_suffix: str = ""
    seed: int = RANDOM_SEED


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def score(pred: np.ndarray, true: np.ndarray) -> float:
    d = pred - true
    scores = np.where(d < 0, np.exp(-d / 10.0) - 1.0, np.exp(d / 13.0) - 1.0)
    return float(np.sum(scores))


def clip_predictions(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 0.0, RUL_CAP)


def load_preprocessed(dataset_id: str, output_dir: Path) -> dict:
    npz_path = output_dir / f"gpt4rul_{dataset_id}_preprocessed.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing {npz_path}. Run: python src/preprocess_gpt4rul.py --dataset-id {dataset_id}"
        )
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def make_dataloaders(
    dataset_id: str,
    batch_size: int,
    cfg: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    bundle = load_preprocessed(dataset_id, cfg.output_dir)

    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    train_idx = bundle["train_idx"]
    val_idx = bundle["val_idx"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]
    feature_cols = bundle["feature_cols"]

    if cfg.val_mode == "last_window" and "val_last_idx" in bundle:
        eval_idx = bundle["val_last_idx"]
    else:
        eval_idx = val_idx

    n_features = len(feature_cols)

    train_ds = TensorDataset(
        torch.from_numpy(X_train[train_idx].astype(np.float32)),
        torch.from_numpy(y_train[train_idx].astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_train[eval_idx].astype(np.float32)),
        torch.from_numpy(y_train[eval_idx].astype(np.float32)),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test.astype(np.float32)),
        torch.from_numpy(y_test.astype(np.float32)),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    meta = {
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "n_features": n_features,
        "val_mode": cfg.val_mode,
        "early_stop_metric": cfg.early_stop_metric,
    }
    return train_loader, val_loader, test_loader, meta


@torch.no_grad()
def evaluate(model: GPT4RUL, loader: DataLoader, device: torch.device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        preds.append(out.cpu().numpy())
        trues.append(y.numpy())
    pred = clip_predictions(np.concatenate(preds))
    true = np.concatenate(trues)
    return rmse(pred, true), score(pred, true), pred, true


def set_seed(seed: int = RANDOM_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(
    dataset_id: str = "FD001",
    device_str: str | None = None,
    cfg: TrainConfig | None = None,
) -> dict:
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    batch_size = batch_size_for(dataset_id)
    window_len = window_length_for(dataset_id)
    patch_stride = patch_stride_for(dataset_id)
    n_pcf = pcf_n_blocks_for(dataset_id)
    mixing_factor = pcf_mixing_factor_for(dataset_id)
    suffix = f"_{cfg.tag_suffix}" if cfg.tag_suffix else ""
    tag = f"gpt4rul_{dataset_id}{suffix}"

    logger.info(
        "[%s] batch=%d, val_mode=%s, early_stop=%s, pcf=%s, grad_clip=%s",
        tag, batch_size, cfg.val_mode, cfg.early_stop_metric, cfg.pcf_style, cfg.grad_clip,
    )

    train_loader, val_loader, test_loader, meta = make_dataloaders(dataset_id, batch_size, cfg)
    logger.info("[%s] train=%d, val=%d, test=%d, features=%d",
                 tag, meta["n_train"], meta["n_val"], meta["n_test"], meta["n_features"])

    model = GPT4RUL(
        n_features=meta["n_features"],
        window_length=window_len,
        patch_size=PATCH_SIZE,
        patch_stride=patch_stride,
        pcf_hidden_dim=PCF_HIDDEN_DIM,
        n_pcf_blocks=n_pcf,
        pcf_mixing_factor=mixing_factor,
        pcf_dropout=DROPOUT,
        pcf_style=cfg.pcf_style,
        gpt2_model_name=GPT2_MODEL_NAME,
        gpt2_n_layers=GPT2_N_LAYERS,
        gpt2_hidden_dim=GPT2_HIDDEN_DIM,
        freeze_gpt2=GPT2_FREEZE,
        pooling="flatten",
        use_gpt2_residual=True,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_EPOCHS, gamma=LR_DECAY_FACTOR)
    criterion = nn.MSELoss()

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.checkpoint_dir / f"best_{tag}.pt"

    best_metric = float("inf")
    best_epoch = 0
    bad = 0
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(losses))
        current_lr = scheduler.get_last_lr()[0]

        val_preds, val_trues = [], []
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                val_preds.append(out.cpu().numpy())
                val_trues.append(y.numpy())
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_loss = float(np.mean((val_preds - val_trues) ** 2))
        val_rmse = rmse(clip_predictions(val_preds), val_trues)
        monitor = val_rmse if cfg.early_stop_metric == "val_rmse" else val_loss

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_rmse": val_rmse, "lr": current_lr,
        })
        logger.info("[%s] epoch %03d | loss=%.4f | val_loss=%.4f | val_rmse=%.4f | lr=%.6f",
                     tag, epoch, train_loss, val_loss, val_rmse, current_lr)

        if monitor < best_metric - 1e-6:
            best_metric = monitor
            best_epoch = epoch
            bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": model.config_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
            }, ckpt_path)
            logger.info("[%s] ✓ Best saved (%s=%.4f)", tag, cfg.early_stop_metric, monitor)
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                logger.info("[%s] Early stop at epoch %d", tag, epoch)
                break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_rmse, test_score, _, _ = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    result = {
        "dataset_id": dataset_id,
        "tag": tag,
        "test_rmse": test_rmse,
        "test_score": test_score,
        "best_epoch": best_epoch,
        "best_val_metric": best_metric,
        "early_stop_metric": cfg.early_stop_metric,
        "val_mode": cfg.val_mode,
        "pcf_style": cfg.pcf_style,
        "train_time_sec": elapsed,
        "n_features": meta["n_features"],
        "n_train": meta["n_train"],
        "n_val": meta["n_val"],
        "n_test": meta["n_test"],
    }

    out_json = cfg.results_dir / f"test_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({**result, "history": history}, f, indent=2)
    logger.info("[%s] TEST rmse=%.4f  score=%.2f  time=%.0fs → %s",
                 tag, test_rmse, test_score, elapsed, out_json)
    return result


def main():
    p = argparse.ArgumentParser(description="GPT4RUL — paper reproduction training")
    p.add_argument("--dataset-id", default="FD001")
    p.add_argument("--all", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--tag-suffix", default="")
    p.add_argument("--val-mode", default=VAL_MODE, choices=["all", "last_window"])
    p.add_argument("--early-stop-metric", default=EARLY_STOP_METRIC, choices=["val_loss", "val_rmse"])
    p.add_argument("--pcf-style", default=PCF_STYLE, choices=["sequential", "parallel"])
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--ablation", action="store_true", help="Use outputs/ablation and results/ablation")
    args = p.parse_args()

    if args.ablation or args.output_dir or args.results_dir:
        out_dir = Path(args.output_dir) if args.output_dir else ABLATION_OUTPUT_DIR
        res_dir = Path(args.results_dir) if args.results_dir else ABLATION_RESULTS_DIR
        ckpt_dir = out_dir / "checkpoints"
    else:
        out_dir = OUTPUT_DIR
        res_dir = RESULTS_DIR
        ckpt_dir = CHECKPOINT_DIR

    cfg = TrainConfig(
        output_dir=out_dir,
        results_dir=res_dir,
        checkpoint_dir=ckpt_dir,
        val_mode=args.val_mode,
        early_stop_metric=args.early_stop_metric,
        pcf_style=args.pcf_style,
        grad_clip=args.grad_clip,
        tag_suffix=args.tag_suffix,
        seed=args.seed,
    )

    datasets = ALL_DATASETS if args.all else [args.dataset_id]
    results = {}
    for ds in datasets:
        results[ds] = train_one(ds, args.device, cfg)

    if args.all or len(datasets) > 1:
        csv_name = f"gpt4rul_summary{('_' + args.tag_suffix) if args.tag_suffix else ''}.csv"
        csv_path = cfg.results_dir / csv_name
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("dataset,test_rmse,test_score,best_epoch,train_time_sec,val_mode,early_stop_metric\n")
            for ds in datasets:
                r = results[ds]
                f.write(
                    f"{ds},{r['test_rmse']:.4f},{r['test_score']:.2f},"
                    f"{r['best_epoch']},{r['train_time_sec']:.0f},"
                    f"{r['val_mode']},{r['early_stop_metric']}\n"
                )
        logger.info("Summary → %s", csv_path)


if __name__ == "__main__":
    main()
