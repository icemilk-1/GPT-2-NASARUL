# Reproducibility: Paper vs. Code

This document provides a line-by-line comparison between Tan et al. (QR2MSE 2025) and the GPT4RUL codebase.

---

## 1. Data Preprocessing (12 Items)

| # | Aspect | Paper (Section 3.2) | Code | Status | Notes |
|---|--------|---------------------|------|--------|-------|
| 1.1 | Clustering method | KMeans on 3 operating settings | `KMeans(n_clusters=k)` on setting1/2/3 | ✅ Consistent | FD001/003 k=1, FD002/004 k=6 |
| 1.2 | Per-cluster normalization | Z-score within each cluster | `StandardScaler` per cluster | ✅ Consistent | |
| 1.3 | Global normalization | MinMax to [0,1] after Z-score | `MinMaxScaler(feature_range=(0,1))` global | ✅ Consistent | Order: Z-score → MinMax |
| 1.4 | Feature selection — Correlation | Pearson corr(sensor, time), weights vary per sensor | Global Pearson corr across all train data | ⚠️ Possibly different | Paper computes per-engine and averages; code uses global Pearson |
| 1.5 | Feature selection — Monotonicity | Monotonicity score per engine, averaged | Per-engine monotonicity, then mean | ✅ Consistent | |
| 1.6 | Feature selection — Combined score | α·Corr + (1−α)·Mono, α=0.5 | `FEATURE_SELECT_WEIGHT_CORR=0.5, FEATURE_SELECT_WEIGHT_MONO=0.5` | ✅ Consistent | |
| 1.7 | Feature selection — Threshold | Cri > mean(Cri) | Same | ✅ Consistent | FD003/004 additionally use `N_SELECTED_FEATURES` top-N override |
| 1.8 | Sliding window stride | stride=1 | `WINDOW_STRIDE=1` for all datasets | ✅ Consistent | |
| 1.9 | Window lengths | FD001=30, FD002=64, FD003=40, FD004=64 | Same | ✅ Consistent | Paper Table 2 |
| 1.10 | RUL label — cap | Capped at 125 | `RUL_CAP=125.0` | ✅ Consistent | |
| 1.11 | RUL label — piecewise | Linear degradation from cap to 0 | Piecewise linear decay | ✅ Consistent | |
| 1.12 | Train/val split | 80/20 temporal by cycle | Random Engine Split (80/20 by engine ID) | ⚠️ Different | Paper uses temporal split within each engine; code uses engine-level random split to prevent leakage |

---

## 2. Model Architecture (15 Items)

| # | Aspect | Paper (Section 2.3) | Code | Status | Notes |
|---|--------|---------------------|------|--------|-------|
| 2.1 | Patch size (K) | K=8 | `PATCH_SIZE=8` | ✅ Consistent | |
| 2.2 | Patch stride (S) | FD001=4, others=8 | `PATCH_STRIDE: FD001=4, FD002/3/4=8` | ✅ Consistent | Paper Table 2 |
| 2.3 | Input projection | Linear(K*F → 128) | `Linear(patch_flat_dim, pcf_hidden_dim)` | ✅ Consistent | pcf_hidden_dim=128 |
| 2.4 | Position encoding | Learnable | `nn.Parameter(randn(1, num_patches, 128) * 0.02)` | ✅ Consistent | |
| 2.5 | PCF Block — Patch-Mixing | MLP across patch dimension | `patch_mlp: Linear(P→P*mf)→GELU→Dropout→Linear(P*mf→P)` | ✅ Consistent | MLP-Mixer style |
| 2.6 | PCF Block — Channel-Mixing | MLP across channel dimension | `channel_mlp: Linear(D→D*mf)→GELU→Dropout→Linear(D*mf→D)` | ✅ Consistent | MLP-Mixer style |
| 2.7 | PCF Block — Normalization | Pre-LN | Dual independent `LayerNorm` before each mixing | ✅ Consistent | `patch_norm` and `channel_norm` |
| 2.8 | PCF Block — Residual | Residual addition after each mixing | `x = x + patch_out`, `x = x + channel_out` | ✅ Consistent | |
| 2.9 | PCF Block — Activation | GELU | `nn.GELU()` | ✅ Consistent | MLP-Mixer [14] |
| 2.10 | PCF Block — count per dataset | FD001=2, FD002=2, FD003=1, FD004=2 | Same | ✅ Consistent | Paper Table 2 |
| 2.11 | Mixing factor per dataset | FD001=2, FD002=2, FD003=1, FD004=1 | Same | ✅ Consistent | Paper Table 2 |
| 2.12 | Linear Mapping | 128 → 768 with LayerNorm | `LayerNorm(128)→Linear(128,768)→Dropout` | ✅ Consistent | |
| 2.13 | GPT-2 model | GPT-2 (openai-community/gpt2) | `GPT2Model.from_pretrained("openai-community/gpt2")` | ✅ Consistent | |
| 2.14 | GPT-2 layers used | First 3 layers only | `gpt2.h = gpt2.h[:3]`, `config.n_layer=3` | ✅ Consistent | Paper Table 2 |
| 2.15 | GPT-2 frozen | Fully frozen | `p.requires_grad_(False)` for all GPT-2 params | ✅ Consistent | Paper Section 2.3.2 |
| 2.16 | Residual around GPT-2 | Add + LayerNorm | `h + hidden → LayerNorm` | ✅ Consistent | Controlled by `use_gpt2_residual` flag |
| 2.17 | Pooling | Flatten | `h.reshape(B, -1)` (flatten mode) | ✅ Consistent | Also supports `mean` and `last` for ablation |
| 2.18 | Output head | LayerNorm → Linear(1) | `LayerNorm→Linear(dim,1)→squeeze` | ✅ Consistent | |

---

## 3. Training Configuration (10 Items)

| # | Aspect | Paper (Table 2) | Code | Status | Notes |
|---|--------|-----------------|------|--------|-------|
| 3.1 | Optimizer | Adam | `torch.optim.Adam` | ✅ Consistent | |
| 3.2 | Learning rate | 0.005 | `LEARNING_RATE=0.005` | ✅ Consistent | |
| 3.3 | LR schedule | StepLR, step=10, γ=0.1 | `StepLR(step_size=10, gamma=0.1)` | ✅ Consistent | |
| 3.4 | Weight decay | 0.01 | `WEIGHT_DECAY=0.01` | ✅ Consistent | L2 regularization |
| 3.5 | Dropout | 0.2 | `DROPOUT=0.2` | ✅ Consistent | |
| 3.6 | Batch size per dataset | FD001=64, FD002=128, FD003=64, FD004=128 | Same | ✅ Consistent | Paper Table 2 |
| 3.7 | Max epochs | 100 | `MAX_EPOCHS=100` | ✅ Consistent | |
| 3.8 | Early stop patience | 10 | `EARLY_STOP_PATIENCE=10` | ✅ Consistent | Paper Table 2 |
| 3.9 | Early stop metric | val_loss (inferred) | `EARLY_STOP_METRIC=val_rmse` | ⚠️ Possibly different | Paper likely uses val_loss; code defaults to val_rmse (configurable) |
| 3.10 | Validation mode | Per-engine last window (inferred) | `VAL_MODE=all` (all windows from val engines) | ⚠️ Different | Code found `all` windows gives more stable validation signal |
| 3.11 | Gradient clipping | Not mentioned | `GRAD_CLIP=0` (disabled by default) | ⚠️ Possibly different | Ablation showed grad_clip=1.0 helps in some configs |
| 3.12 | Loss function | MSE | `nn.MSELoss()` | ✅ Consistent | |

---

## 4. Evaluation (3 Items)

| # | Aspect | Paper | Code | Status | Notes |
|---|--------|-------|------|--------|-------|
| 4.1 | RMSE | ✓ | `sqrt(mean((pred - true)^2))` | ✅ Consistent | |
| 4.2 | Scoring function | C-MAPSS asymmetric scoring | Late predictions penalized more heavily | ✅ Consistent | Standard PHM08 metric |
| 4.3 | Test RUL clipping | Capped at 125 | Same clipping applied to test predictions | ✅ Consistent | Removing clipping causes FD002/004 to collapse |

---

## 5. Extensions Beyond Paper

These are improvements added during reproduction that go beyond the original paper:

| # | Extension | Description | Impact |
|---|-----------|-------------|--------|
| E.1 | Condition Embedding | Encode operating condition params and inject into model | ✅ Improved all 4 datasets |
| E.2 | Soft Prompts (Hybrid Prompt) | M learnable 768-dim vectors prepended to GPT-2 input | ✅ FD001: 14.87→13.80, FD002: 12.87→12.55 |
| E.3 | No-GPT2 ablation baseline | PCF-only model to isolate LLM contribution | ✅ Revealed LLM value depends on task complexity |
| E.4 | Random Engine Split | Engine-level split instead of temporal split | ✅ Prevents cross-engine data leakage |
| E.5 | Global Pearson correlation | Compute Corr globally instead of per-engine average | ✅ Improved FD003/004 |
| E.6 | PCF Parallel variant | Dual-path parallel MLP + additive fusion | 🔬 Under investigation |

---

## Summary

| Category | Total | ✅ Consistent | ⚠️ Different / Uncertain |
|----------|-------|-------------|------------------------|
| Preprocessing | 12 | 9 | 3 |
| Model Architecture | 18 | 18 | 0 |
| Training | 12 | 8 | 4 |
| Evaluation | 3 | 3 | 0 |
| **Total** | **45** | **38** | **7** |

The 7 differences are intentional improvements discovered during reproduction (see [Reproduction Diagnostics](../README.md#reproduction-diagnostics-12-experiments) in README).
