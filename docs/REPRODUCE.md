# 一键复现指南 / Reproduction Guide

## 前置条件

- Python 3.8+
- Git（上传/克隆仓库）
- C-MAPSS 数据集（**不包含在仓库中**，需自行下载）

## 1. 下载 C-MAPSS 数据

参见 [data/README.md](../data/README.md)，将 12 个 `.txt` 文件放入 `data/CMaps/`。

## 2. 一键复现（Windows）

```powershell
git clone https://github.com/icemilk-1/GPT4RUL-Reproduction.git
cd GPT4RUL-Reproduction

# 环境安装 + GPT-2 下载 + 数据校验
.\scripts\setup.ps1

# 完整复现 FD001-FD004（CPU 可能需要数小时）
.\scripts\reproduce.ps1

# 或只跑单个数据集
.\scripts\reproduce.ps1 -DatasetId FD001
```

## 3. 一键复现（Linux / Mac）

```bash
git clone https://github.com/icemilk-1/GPT4RUL-Reproduction.git
cd GPT4RUL-Reproduction
bash scripts/setup.sh
bash scripts/reproduce.sh          # 全部四数据集
bash scripts/reproduce.sh FD001      # 单个数据集
```

## 4. 手动分步执行

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python scripts/download_gpt2.py
python scripts/check_data.py
python src/preprocess_gpt4rul.py --all
python src/train_gpt4rul.py --all
python scripts/evaluate_all.py
```

## 5. 结果对比

训练完成后查看 `results/gpt4rul_summary.csv`，与参考结果对比：

```powershell
fc results\gpt4rul_summary.csv results\expected\gpt4rul_summary.csv
```

| Dataset | Expected RMSE | Paper (Tan et al.) |
|---------|---------------|---------------------|
| FD001 | 14.87 | 11.23 |
| FD002 | 12.87 | 11.05 |
| FD003 | 13.69 | 11.45 |
| FD004 | 13.38 | 12.68 |

## 6. 上传到 GitHub

本机需安装 [Git](https://git-scm.com/) 和 [GitHub CLI](https://cli.github.com/)（可选）。

```powershell
cd E:\2026research\RUL_prompt
git init
git add .
git status   # 确认无 data/CMaps、.venv、*.pt 被加入
git commit -m "Initial release: GPT4RUL reproduction with one-click scripts"
gh repo create GPT4RUL-Reproduction --public --source=. --push
```

若无 `gh`，在 GitHub 网页新建空仓库后：

```powershell
git remote add origin https://github.com/YOUR_USERNAME/GPT4RUL-Reproduction.git
git branch -M main
git push -u origin main
```

## 7. 不包含在仓库中的文件

| 路径 | 原因 |
|------|------|
| `data/CMaps/*.txt` | NASA 数据，版权/体积 |
| `.venv/` | 本地 Python 环境 |
| `.hf_cache/` | GPT-2 权重（setup 自动下载） |
| `outputs/checkpoints/*.pt` | 训练产物（需自行训练） |
| `results/`（除 expected） | 实验输出 |

## 8. 常见问题

**Q: GPT-2 下载失败？**  
A: 检查网络，或手动设置 `HF_ENDPOINT` 镜像后重跑 `python scripts/download_gpt2.py`。

**Q: 评估脚本报错？**  
A: 确保先完成训练和预处理；checkpoint 路径为 `outputs/checkpoints/best_gpt4rul_FD00x.pt`。

**Q: 与论文结果有差距？**  
A: 参见 [docs/reproducibility.md](reproducibility.md) 和 [paper/revisiting_gpt4rul_zh.pdf](../paper/revisiting_gpt4rul_zh.pdf)。
