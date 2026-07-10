#!/bin/bash
# Train GPT4RUL on FD003 (single condition, dual fault modes)
set -e
cd "$(dirname "$0")/.."
echo "=== Preprocessing FD003 ==="
python src/preprocess_gpt4rul.py --dataset-id FD003
echo "=== Training FD003 ==="
python src/train_gpt4rul.py --dataset-id FD003
echo "=== Evaluating FD003 ==="
python src/evaluate_gpt4rul.py --dataset-id FD003
echo "=== Done: FD003 ==="
