"""Training script for Hybrid Prompt RUL models.

Modes:
  hybrid  — Route B: PCF → Soft Prompts → Frozen GPT-2
  text    — Route A: Sensor text → Tokenizer → Soft Prompts → Frozen GPT-2
  no_gpt2 — Ablation: PCF → Flatten → Linear (no GPT-2)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
    DROPOUT,
    EARLY_STOP_PATIENCE,
    GPT2_HIDDEN_DIM,
    GPT2_MODEL_NAME,
    GPT2_N_LAYERS,
    LEARNING_RATE,
    LR_DECAY_EPOCHS,
    LR_DECAY_FACTOR,
    MAX_EPOCHS,
    OUTPUT_DIR,
    PATCH_SIZE,
    PCF_HIDDEN_DIM,
    RANDOM_SEED,
    RESULTS_DIR,
    RUL_CAP,
    WEIGHT_DECAY,
    batch_size_for,
    pcf_mixing_factor_for,
    pcf_n_blocks_for,
    patch_stride_for,
    window_length_for,
)
from src.model_hybrid_prompt import HybridPromptRUL
from src.model_text_prompt import TextPromptRUL
from src.text_encoder import windows_to_texts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
#  Metrics & utilities
# =============================================================================

def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def score(pred: np.ndarray, true: np.ndarray) -> float:
    d = pred - true
    scores = np.where(d < 0, np.exp(-d / 10.0) - 1.0, np.exp(d / 13.0) - 1.0)
    return float(np.sum(scores))


def clip_predictions(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 0.0, RUL_CAP)


def set_seed(seed: int = RANDOM_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
#  Data loading (reuses existing .npz)
# =============================================================================

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
    output_dir: Path,
    val_mode: str = "all",
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    bundle = load_preprocessed(dataset_id, output_dir)
    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    train_idx = bundle["train_idx"]
    val_idx = bundle["val_idx"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]
    feature_cols = bundle["feature_cols"]

    if val_mode == "last_window" and "val_last_idx" in bundle:
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
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "n_features": n_features, "val_mode": val_mode,
    }
    return train_loader, val_loader, test_loader, meta


# =============================================================================
#  No-GPT2 baseline model (ablation)
# =============================================================================

class NoGPT2Baseline(nn.Module):
    """PCF → Flatten → Linear → RUL (no GPT-2, no soft prompts)."""

    def __init__(
        self,
        n_features: int,
        window_length: int,
        patch_size: int = 8,
        patch_stride: int = 4,
        pcf_hidden_dim: int = 128,
        n_pcf_blocks: int = 2,
        pcf_mixing_factor: float = 2.0,
        pcf_dropout: float = 0.2,
    ):
        super().__init__()
        self.num_patches = (window_length - patch_size) // patch_stride + 1
        self.patch_flat_dim = patch_size * n_features

        self.input_proj = nn.Linear(self.patch_flat_dim, pcf_hidden_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, pcf_hidden_dim) * 0.02,
        )

        from src.pcf_block import PCFBlock
        self.pcf_blocks = nn.ModuleList([
            PCFBlock(dim=pcf_hidden_dim, num_patches=self.num_patches,
                     mixing_factor=pcf_mixing_factor, dropout=pcf_dropout)
            for _ in range(n_pcf_blocks)
        ])

        self.output_norm = nn.LayerNorm(pcf_hidden_dim * self.num_patches)
        self.output_head = nn.Linear(pcf_hidden_dim * self.num_patches, 1)

        total = sum(p.numel() for p in self.parameters())
        logger.info("NoGPT2Baseline: %d params (all trainable)", total)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_t = x.transpose(1, 2)
        patches = x_t.unfold(dimension=2, size=self.patch_size, step=self.patch_stride)
        patches = patches.permute(0, 2, 1, 3)
        patches = patches.reshape(B, self.num_patches, self.patch_flat_dim)
        return patches.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches = self._patchify(x)
        h = self.input_proj(patches)
        h = h + self.pos_encoding[:, :self.num_patches, :]
        for block in self.pcf_blocks:
            h = block(h)
        h = h.reshape(B, -1)
        h = self.output_norm(h)
        return self.output_head(h).squeeze(-1)


# =============================================================================
#  Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, text_mode: bool = False):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        if text_mode:
            input_ids, attn_mask, y = [t.to(device) for t in batch]
            out = model(input_ids, attn_mask)
        else:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
        preds.append(out.cpu().numpy())
        trues.append(y.cpu().numpy())
    pred = clip_predictions(np.concatenate(preds))
    true = np.concatenate(trues)
    return rmse(pred, true), score(pred, true), pred, true


# =============================================================================
#  Training
# =============================================================================

def train_one(
    dataset_id: str = "FD001",
    mode: str = "hybrid",
    n_soft_prompts: int = 4,
    device_str: str | None = None,
    seed: int = RANDOM_SEED,
    output_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict:
    set_seed(seed)
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s | Mode: %s | Soft prompts: %d", device, mode, n_soft_prompts)

    batch_size = batch_size_for(dataset_id)
    window_len = window_length_for(dataset_id)
    patch_stride = patch_stride_for(dataset_id)
    n_pcf = pcf_n_blocks_for(dataset_id)
    mixing_factor = pcf_mixing_factor_for(dataset_id)

    out_dir = output_dir or OUTPUT_DIR
    res_dir = results_dir or RESULTS_DIR
    tag = f"hybrid_{dataset_id}_{mode}_M{n_soft_prompts}"

    logger.info("[%s] batch=%d window=%d pcf_blocks=%d", tag, batch_size, window_len, n_pcf)

    train_loader, val_loader, test_loader, meta = make_dataloaders(
        dataset_id, batch_size, out_dir,
    )
    logger.info("[%s] train=%d val=%d test=%d features=%d",
                 tag, meta["n_train"], meta["n_val"], meta["n_test"], meta["n_features"])

    # ---- Build model ----
    if mode == "no_gpt2":
        model = NoGPT2Baseline(
            n_features=meta["n_features"],
            window_length=window_len,
            patch_size=PATCH_SIZE,
            patch_stride=patch_stride,
            pcf_hidden_dim=PCF_HIDDEN_DIM,
            n_pcf_blocks=n_pcf,
            pcf_mixing_factor=mixing_factor,
            pcf_dropout=DROPOUT,
        ).to(device)
    elif mode == "hybrid":
        model = HybridPromptRUL(
            n_features=meta["n_features"],
            window_length=window_len,
            patch_size=PATCH_SIZE,
            patch_stride=patch_stride,
            pcf_hidden_dim=PCF_HIDDEN_DIM,
            n_pcf_blocks=n_pcf,
            pcf_mixing_factor=mixing_factor,
            pcf_dropout=DROPOUT,
            n_soft_prompts=n_soft_prompts,
            gpt2_model_name=GPT2_MODEL_NAME,
            gpt2_n_layers=GPT2_N_LAYERS,
            gpt2_hidden_dim=GPT2_HIDDEN_DIM,
            freeze_gpt2=True,
            pooling="flatten",
        ).to(device)
    elif mode == "text":
        # ---- Route A: Text prompt — load data + tokenize ----
        model = TextPromptRUL(
            gpt2_model_name=GPT2_MODEL_NAME,
            gpt2_n_layers=GPT2_N_LAYERS,
            gpt2_hidden_dim=GPT2_HIDDEN_DIM,
            n_soft_prompts=n_soft_prompts,
            freeze_gpt2=True,
            max_length=512,
        ).to(device)
        tokenizer = model.tokenizer
        # Load raw preprocessed data for text encoding
        bundle = load_preprocessed(dataset_id, out_dir)
        feature_cols = [str(c) for c in bundle["feature_cols"]]
        X_train_arr = bundle["X_train"]
        y_train_arr = bundle["y_train"]
        train_idx = bundle["train_idx"]
        val_idx = bundle["val_idx"]
        eval_idx = val_idx  # text mode always uses "all" windows
        logger.info("Tokenizing train texts (%d samples)...", len(train_idx))
        train_texts = windows_to_texts(X_train_arr[train_idx], feature_cols)
        train_tokens = tokenizer(train_texts, padding=True, truncation=True,
                                  max_length=512, return_tensors="pt")
        logger.info("Tokenizing val texts (%d samples)...", len(eval_idx))
        val_texts = windows_to_texts(X_train_arr[eval_idx], feature_cols)
        val_tokens = tokenizer(val_texts, padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
        logger.info("Tokenizing test texts (%d samples)...", len(bundle["X_test"]))
        test_texts = windows_to_texts(bundle["X_test"], feature_cols)
        test_tokens = tokenizer(test_texts, padding=True, truncation=True,
                                 max_length=512, return_tensors="pt")
        n_features = len(feature_cols)
        meta = {"n_train": len(train_idx), "n_val": len(eval_idx),
                 "n_test": len(bundle["X_test"]), "n_features": n_features}
        train_ds = TensorDataset(
            train_tokens["input_ids"], train_tokens["attention_mask"],
            torch.from_numpy(y_train_arr[train_idx].astype(np.float32)),
        )
        val_ds = TensorDataset(
            val_tokens["input_ids"], val_tokens["attention_mask"],
            torch.from_numpy(y_train_arr[eval_idx].astype(np.float32)),
        )
        test_ds = TensorDataset(
            test_tokens["input_ids"], test_tokens["attention_mask"],
            torch.from_numpy(bundle["y_test"].astype(np.float32)),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info("[%s] Trainable params: %d", tag, n_trainable)

    is_text_mode = (mode == "text")

    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_EPOCHS, gamma=LR_DECAY_FACTOR)
    criterion = nn.SmoothL1Loss()  # more robust to outliers than MSE

    res_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_{tag}.pt"

    best_metric = float("inf")
    best_epoch = 0
    bad = 0
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            if is_text_mode:
                input_ids, attn_mask, y = [t.to(device) for t in batch]
                optimizer.zero_grad(set_to_none=True)
                pred = model(input_ids, attn_mask)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(losses))
        current_lr = scheduler.get_last_lr()[0]

        val_preds, val_trues = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if is_text_mode:
                    input_ids, attn_mask, y = [t.to(device) for t in batch]
                    out = model(input_ids, attn_mask)
                else:
                    x, y = batch[0].to(device), batch[1].to(device)
                    out = model(x)
                val_preds.append(out.cpu().numpy())
                val_trues.append(y.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_loss = float(np.mean((val_preds - val_trues) ** 2))
        val_rmse_val = rmse(clip_predictions(val_preds), val_trues)
        monitor = val_rmse_val

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_rmse": val_rmse_val, "lr": current_lr,
        })
        logger.info("[%s] epoch %03d | loss=%.4f | val_loss=%.4f | val_rmse=%.4f | lr=%.6f",
                     tag, epoch, train_loss, val_loss, val_rmse_val, current_lr)

        if monitor < best_metric - 1e-6:
            best_metric = monitor
            best_epoch = epoch
            bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": model.config_dict() if hasattr(model, "config_dict") else {},
                "epoch": epoch,
                "val_loss": val_loss,
                "val_rmse": val_rmse_val,
            }, ckpt_path)
            logger.info("[%s] ✓ Best saved (val_rmse=%.4f)", tag, monitor)
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                logger.info("[%s] Early stop at epoch %d", tag, epoch)
                break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_rmse_val, test_score_val, _, _ = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    result = {
        "dataset_id": dataset_id,
        "mode": mode,
        "n_soft_prompts": n_soft_prompts,
        "tag": tag,
        "test_rmse": test_rmse_val,
        "test_score": test_score_val,
        "best_epoch": best_epoch,
        "best_val_rmse": best_metric,
        "train_time_sec": elapsed,
        "n_features": meta["n_features"],
        "n_train": meta["n_train"],
        "n_val": meta["n_val"],
        "n_test": meta["n_test"],
        "n_trainable_params": n_trainable,
    }

    out_json = res_dir / f"test_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({**result, "history": history}, f, indent=2)
    logger.info("[%s] TEST rmse=%.4f  score=%.2f  time=%.0fs → %s",
                 tag, test_rmse_val, test_score_val, elapsed, out_json)
    return result


# =============================================================================
#  CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Hybrid Prompt RUL training")
    p.add_argument("--dataset-id", default="FD001")
    p.add_argument("--all", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--mode", default="hybrid", choices=["hybrid", "text", "no_gpt2"],
                   help="hybrid=RouteB | text=RouteA | no_gpt2=PCF only")
    p.add_argument("--n-soft-prompts", type=int, default=4)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--results-dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    res_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    datasets = ALL_DATASETS if args.all else [args.dataset_id]
    results = {}
    for ds in datasets:
        results[ds] = train_one(
            ds, args.mode, args.n_soft_prompts, args.device, args.seed,
            out_dir, res_dir,
        )

    if len(datasets) > 1:
        csv_path = res_dir / f"hybrid_summary_{args.mode}.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("dataset,mode,test_rmse,test_score,best_epoch,n_soft_prompts\n")
            for ds in datasets:
                r = results[ds]
                f.write(f"{ds},{r['mode']},{r['test_rmse']:.4f},{r['test_score']:.2f},"
                        f"{r['best_epoch']},{r['n_soft_prompts']}\n")
        logger.info("Summary → %s", csv_path)


if __name__ == "__main__":
    main()
