# GPT4RUL — Pre-trained LLM-based RUL Prediction of Aircraft Engines

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)

Reproduction and improvement of: **Tan et al., "Pre-trained LLM-based Remaining Useful Life Prediction of Aircraft Engines"** (QR2MSE 2025).

> 将 NLP 的"预训练-微调"范式迁移到 PHM 领域：冻结 GPT-2 作为时序特征提取器，用轻量级 PCF Adapter + 可训练软提示桥接传感器数据与 LLM 语义空间。在 NASA C-MAPSS 数据集上复现并**在 FD002 上超越论文结果**。

---

## Results Overview

### 消融实验：GPT-2 到底有没有用？

| Dataset | Conditions | Faults | No-GPT2 (PCF only) | GPT4RUL+CE | **Hybrid+SP (M=4)** | Paper |
|---------|-----------|--------|-------------------|-----------|-------------------|-------|
| FD001 | 1 | 1 | 14.16 | 14.87 | **13.80** | ~11.2 |
| FD002 | 6 | 1 | 13.61 | 12.87 | **12.55** ✅ | ~12.6 |
| FD003 | 1 | 2 | **12.99** | 13.69 | 13.59 | ~13.2 |
| FD004 | 6 | 2 | 14.52 | **13.38** | 13.76 | ~12.9 |

> **CE** = Condition Embedding, **SP** = Soft Prompts. Bold = best per dataset. ✅ = surpasses paper.

### Key Findings

1. **LLM 的价值取决于任务复杂度**：简单场景（单工况 FD003）纯 PCF 已足够（12.99）；多工况大数据场景 GPT-2 + 软提示才发挥优势（FD002: 12.55）
2. **软提示是激活 LLM 能力的关键**：不加软提示时 GPT-2 甚至拖后腿（FD001: 14.16 → 14.87 → 13.80），软提示让 LLM 从"噪声源"变成"增强器"
3. **FD002 首次且唯一超越论文**（12.55 < ~12.6）
4. **没有万能架构**：4 个数据集的胜出模型各不相同，需要根据任务复杂度自适应选择

---

## Architecture

```
Sensor Window (B, T, F)
  ↓ Patching — unfold(K=8, stride per dataset)             → (B, P, K*F)
  ↓ Linear Proj (K*F → 128) + Learnable Position Enc       → (B, P, 128)
  ↓ PCF Block × N (MLP-Mixer style, Pre-LN)                → (B, P, 128)
  │   ├─ Patch-Mixing MLP:  transpose → MLP across P → + residual
  │   └─ Channel-Mixing MLP: MLP across D → + residual
  ↓ Linear Mapping — LayerNorm → Linear(128 → 768)         → (B, P, 768)
  ↓ Frozen GPT-2 (first 3 layers, inputs_embeds)            → (B, P, 768)
  ↓ Add & LayerNorm (residual around GPT-2)                 → (B, P, 768)
  ↓ Flatten & LayerNorm → Linear → RUL                      → (B,)
```

**Trainable** (~95K params): PCF blocks, input embedding, linear mapping, output head, soft prompts (optional).
**Frozen**: GPT-2 (3 layers, ~38M params).

### Hybrid Prompt Variant (Soft Prompt Injection)

```
Sensor Features (B, P, 768)
  ↓ [Soft Prompts (B, M, 768); Sensor Features (B, P, 768)]
  ↓ → Extended Sequence (B, M+P, 768)
  ↓ Frozen GPT-2 → strip prompt tokens → keep data tokens
  ↓ Flatten → Linear(1) → RUL
```

Soft prompts are M learnable 768-dim vectors prepended to the sequence. They act as "task instructions" that guide GPT-2's attention toward degradation-relevant patterns.

### PCF Block (Patch-Channel Fusion)

MLP-Mixer inspired, two sequential sub-steps with dual independent Pre-LN:
1. LayerNorm → Patch-Mixing MLP (mix across time patches) → Residual Add
2. LayerNorm → Channel-Mixing MLP (mix across feature channels) → Residual Add

Alternative: `--pcf-style parallel` (dual-path parallel MLP + additive fusion).

---

## Preprocessing Pipeline

| Step | Method |
|------|--------|
| Clustering | KMeans on 3 operating settings: FD002/004 k=6, FD001/003 k=1 |
| Normalization | Per-cluster Z-score → Global MinMax [0,1] |
| Feature selection | Corr(0.5) + Monotonicity(0.5) combined score, keep above-mean sensors |
| Windowing | Per-dataset sliding window (stride=1) |
| RUL label | Piecewise linear decay, capped at 125 |
| Split | Random Engine Split (80/20 by engine ID) — **no data leakage** |

## Per-Dataset Hyperparameters

| Param | FD001 | FD002 | FD003 | FD004 |
|-------|-------|-------|-------|-------|
| Window | 30 | 64 | 40 | 64 |
| Patch stride | 4 | 8 | 8 | 8 |
| PCF blocks | 2 | 2 | 1 | 2 |
| Mixing factor | 2 | 2 | 1 | 1 |
| Batch size | 64 | 128 | 64 | 128 |
| Epochs | 100 | 100 | 100 | 100 |
| Early stop patience | 10 | 10 | 10 | 10 |

Global: `lr=0.005`, `weight_decay=0.01`, `dropout=0.2`, `StepLR(step=10, γ=0.1)`.

---

## One-Click Reproduction（一键复现）

> 完整指南见 [docs/REPRODUCE.md](docs/REPRODUCE.md)

**Windows（推荐）**

```powershell
git clone https://github.com/YOUR_USERNAME/GPT4RUL-Reproduction.git
cd GPT4RUL-Reproduction

# 1) 按 data/README.md 下载 C-MAPSS 到 data/CMaps/
# 2) 环境安装 + GPT-2 下载 + 数据校验
.\scripts\setup.ps1

# 3) 完整复现 FD001-FD004（CPU 可能需要数小时）
.\scripts\reproduce.ps1

# 单数据集： .\scripts\reproduce.ps1 -DatasetId FD001
# 对比结果： fc results\gpt4rul_summary.csv results\expected\gpt4rul_summary.csv
```

**Linux / Mac**

```bash
bash scripts/setup.sh && bash scripts/reproduce.sh
```

---

## Quick Start（手动分步）

```bash
cd GPT4RUL-Reproduction

# 1) Install dependencies
pip install -r requirements.txt
python scripts/download_gpt2.py

# 2) Download C-MAPSS → data/CMaps/ (see data/README.md)
python scripts/check_data.py

# 3) Preprocess all datasets
python src/preprocess_gpt4rul.py --all

# 4) Train all datasets
python src/train_gpt4rul.py --all

# 5) Evaluate
python scripts/evaluate_all.py
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset-id` | FD001 / FD002 / FD003 / FD004 | required |
| `--all` | Train all 4 datasets | — |
| `--tag-suffix` | Experiment tag for outputs | auto |
| `--pcf-style` | `sequential` or `parallel` | sequential |
| `--val-mode` | `all` or `last_window` | all |
| `--early-stop-metric` | `val_loss` or `val_rmse` | val_rmse |
| `--grad-clip` | Gradient clipping norm (0 = disabled) | 0 |
| `--device` | `cuda` / `cpu` | auto |

---

## Reproduction Diagnostics (12+ Experiments)

| Investigation | Attempt | Conclusion |
|--------------|---------|------------|
| PCF LayerNorm | Single shared LN → Dual Pre-LN | ✅ Aligned with MLP-Mixer |
| Validation leakage | val_last_idx not filtering train engines | ✅ Fixed |
| Validation mode | last_window → all windows | ✅ Eliminated spurious val_rmse=0 |
| Feature Corr calculation | Per-engine avg → Global Pearson | ✅ Improved FD003/004 |
| FD004 feature count | threshold → top-14 | ✅ FD004: 14.31→13.97 |
| FD003 forced 14 feat | 13→14 | ❌ Natural optimum is 13 |
| Feature selection scope | train+test full set | ❌ Introduced noise |
| Test RUL clipping | Remove clipping | ❌ FD002/004 collapsed |
| Dropout | 0.2→0.3 | ❌ Degraded |
| Learning rate | 0.002 / 0.004 | ❌ Both worse than 0.005 |
| Data split | Temporal split | ❌ Cross-engine leakage |
| **Condition Embedding** | Add operating params to input | ✅ **Improved all 4 datasets** |

---

## Project Structure

```
GPT4RUL/
├── README.md                       # This file
├── LICENSE                         # MIT
├── .gitignore
├── requirements.txt
├── config_gpt4rul.py               # Global config (paths, hyperparams, dataset params)
├── data/
│   └── README.md                   # C-MAPSS download guide
├── src/
│   ├── __init__.py
│   ├── preprocess_gpt4rul.py       # KMeans → Z-score → MinMax → Feature select → Window
│   ├── model_gpt4rul.py            # Patching + PCF + Frozen GPT-2
│   ├── model_hybrid_prompt.py      # Soft prompt variant
│   ├── model_text_prompt.py        # Text prompt variant
│   ├── text_encoder.py             # Text encoder for prompt variants
│   ├── train_gpt4rul.py            # Training loop + early stop + validation
│   ├── train_hybrid_prompt.py      # Training for hybrid prompt variant
│   ├── run_ablation_gpt4rul.py     # Ablation experiment manager
│   ├── evaluate_gpt4rul.py         # Standalone evaluation + visualization
│   └── utils.py                    # set_seed, save/load checkpoint, plot helpers
├── scripts/
│   ├── setup.ps1 / setup.sh        # Environment setup
│   ├── reproduce.ps1 / reproduce.sh / reproduce.bat  # One-click pipeline
│   ├── check_data.py               # Verify C-MAPSS files
│   ├── download_gpt2.py            # Cache GPT-2 locally
│   ├── evaluate_all.py             # Batch evaluation
│   └── run_fd001.sh ~ run_fd004.sh # Per-dataset (legacy)
├── results/
│   └── expected/                   # Reference metrics (committed)
│       └── gpt4rul_summary.csv
├── logs/                           # Reproduction logs (gitignored)
├── figures/                        # Visualization outputs (created lazily)
├── 论文/                           # Reference papers
└── docs/
    └── reproducibility.md          # Paper-vs-code comparison table
```

---

## Reproducibility

See [docs/reproducibility.md](docs/reproducibility.md) for a detailed 37-item comparison table covering preprocessing, model architecture, and training configuration.

---

## Citation

```bibtex
@inproceedings{tan2025gpt4rul,
  title     = {Pre-trained LLM-based Remaining Useful Life Prediction of Aircraft Engines},
  author    = {Tan, Qingcheng and Yang, Lechang and Zhu, Feng and Wang, Zhe},
  booktitle = {15th International Conference on Quality, Reliability, Risk,
               Maintenance, and Safety Engineering (QR2MSE)},
  year      = {2025},
  address   = {Hohhot, China},
  month     = jul
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
