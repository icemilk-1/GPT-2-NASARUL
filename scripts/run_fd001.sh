#!/bin/bash
# Train GPT4RUL on FD001 (single condition, single fault mode)
set -e
cd "$(dirname "$0")/.."
echo "=== Preprocessing FD001 ==="
python src/preprocess_gpt4rul.py --dataset-id FD001
echo "=== Training FD001 ==="
python src/train_gpt4rul.py --dataset-id FD001
echo "=== Evaluating FD001 ==="
python src/evaluate_gpt4rul.py --dataset-id FD001
echo "=== Done: FD001 ==="
