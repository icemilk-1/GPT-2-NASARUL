#!/bin/bash
# Train GPT4RUL on FD004 (6 conditions, dual fault modes)
set -e
cd "$(dirname "$0")/.."
echo "=== Preprocessing FD004 ==="
python src/preprocess_gpt4rul.py --dataset-id FD004
echo "=== Training FD004 ==="
python src/train_gpt4rul.py --dataset-id FD004
echo "=== Evaluating FD004 ==="
python src/evaluate_gpt4rul.py --dataset-id FD004
echo "=== Done: FD004 ==="
