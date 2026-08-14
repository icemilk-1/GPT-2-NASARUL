# 一键复现指南 / Reproduction Guide

## 前置条件

- Python 3.8+
- Git
- C-MAPSS 数据集（**不包含在仓库中**，需自行下载）

## 1. 下载 C-MAPSS 数据

参见 [data/README.md](../data/README.md)，将 12 个 `.txt` 文件放入 `data/CMaps/`。

## 2. 一键复现 CE 基线（Windows）

```powershell
git clone https://github.com/icemilk-1/GPT-2-NASARUL.git
cd GPT-2-NASARUL

# 环境安装 + GPT-2 下载 + 数据校验
.\scripts\setup.ps1

# 完整复现 FD001–FD004（GPT4RUL + Condition Embedding）
.\scripts\reproduce.ps1

# 或只跑单个数据集
.\scripts\reproduce.ps1 -DatasetId FD001
```

## 3. 一键复现（Linux / Mac）

```bash
git clone https://github.com/icemilk-1/GPT-2-NASARUL.git
cd GPT-2-NASARUL
bash scripts/setup.sh
bash scripts/reproduce.sh          # 全部四数据集
bash scripts/reproduce.sh FD001    # 单个数据集
```

## 4. 手动分步（CE 基线）

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python scripts/download_gpt2.py
python scripts/check_data.py
python src/preprocess_gpt4rul.py --all
python src/train_gpt4rul.py --all
python scripts/evaluate_all.py
```

## 5. Hybrid Soft Prompt / No-GPT2

预处理完成后可跑消融与 Hybrid 变体：

```powershell
python src/train_hybrid_prompt.py --all --mode hybrid --n-soft-prompts 4
python src/train_hybrid_prompt.py --all --mode no_gpt2

# 单数据集
python src/train_hybrid_prompt.py --dataset-id FD002 --mode hybrid --n-soft-prompts 4
```

## 6. 结果对比

### GPT4RUL + CE（一键复现默认）

```powershell
fc results\gpt4rul_summary.csv results\expected\gpt4rul_summary.csv
```

| Dataset | Expected RMSE (CE) | Paper (Tan et al.) |
|---------|--------------------|--------------------|
| FD001 | 14.87 | 11.23 |
| FD002 | 12.87 | 11.05 |
| FD003 | 13.69 | 11.45 |
| FD004 | 13.38 | 12.68 |

### Hybrid / No-GPT2

将生成的 `results/hybrid_summary_hybrid.csv`、`results/hybrid_summary_no_gpt2.csv` 与：

- [`results/expected/hybrid_summary_hybrid.csv`](../results/expected/hybrid_summary_hybrid.csv)
- [`results/expected/hybrid_summary_no_gpt2.csv`](../results/expected/hybrid_summary_no_gpt2.csv)

对比。参考 RMSE：

| Dataset | No-GPT2 | Hybrid+SP (M=4) |
|---------|---------|-----------------|
| FD001 | 14.16 | **13.80** |
| FD002 | 13.61 | **12.55** |
| FD003 | **12.99** | 13.59 |
| FD004 | 14.52 | 13.76 |

## 7. 推送到已有远程仓库

本仓库远程为 `https://github.com/icemilk-1/GPT-2-NASARUL.git`。本地有提交后：

```powershell
git status
git push -u origin main
```

首次建库时可使用 GitHub CLI：

```powershell
gh repo create GPT-2-NASARUL --public --source=. --push
```

提交前请确认 `data/CMaps/`、`.venv*`、`*.pt` 未被加入（见 `.gitignore`）。

## 8. 不包含在仓库中的文件

| 路径 | 原因 |
|------|------|
| `data/CMaps/*.txt` | NASA 数据，版权/体积 |
| `.venv/`、`.venv_cuda/` | 本地 Python 环境 |
| `.hf_cache/` | GPT-2 权重（setup 自动下载） |
| `outputs/checkpoints/*.pt` | 训练产物（需自行训练） |
| `results/`（除 `expected/`） | 实验输出 |

## 9. 常见问题

**Q: GPT-2 下载失败？**  
A: 检查网络，或手动设置 `HF_ENDPOINT` 镜像后重跑 `python scripts/download_gpt2.py`。

**Q: 评估脚本报错？**  
A: 确保先完成训练和预处理；checkpoint 路径为 `outputs/checkpoints/best_gpt4rul_FD00x.pt`。

**Q: 与论文结果有差距？**  
A: 参见 [docs/reproducibility.md](reproducibility.md) 与 [paper/revisiting_gpt4rul_zh.pdf](../paper/revisiting_gpt4rul_zh.pdf)。Hybrid 在 FD002 上可优于论文公开值；其余子集仍有差距，属已知复现现象。
