#!/bin/bash
# Train GPT4RUL on FD002 (6 conditions, single fault mode)
set -e
cd "$(dirname "$0")/.."
echo "=== Preprocessing FD002 ==="
python src/preprocess_gpt4rul.py --dataset-id FD002
echo "=== Training FD002 ==="
python src/train_gpt4rul.py --dataset-id FD002
echo "=== Evaluating FD002 ==="
python src/evaluate_gpt4rul.py --dataset-id FD002
echo "=== Done: FD002 ==="
