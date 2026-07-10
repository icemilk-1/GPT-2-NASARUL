#!/usr/bin/env bash
# GPT4RUL one-click full reproduction (Linux/Mac)
# Usage:
#   bash scripts/reproduce.sh           # all 4 datasets
#   bash scripts/reproduce.sh FD001     # single dataset
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET="${1:-}"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/reproduce_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

log "=== GPT4RUL Reproduction ==="
log "NOTE: Full 4-dataset run on CPU may take several hours."

bash scripts/setup.sh 2>&1 | tee -a "$LOG_FILE"

if [ -n "$DATASET" ]; then
  log "Running single dataset: $DATASET"
  python src/preprocess_gpt4rul.py --dataset-id "$DATASET" 2>&1 | tee -a "$LOG_FILE"
  python src/train_gpt4rul.py --dataset-id "$DATASET" 2>&1 | tee -a "$LOG_FILE"
  python src/evaluate_gpt4rul.py --dataset-id "$DATASET" 2>&1 | tee -a "$LOG_FILE"
else
  log "Running all datasets"
  python src/preprocess_gpt4rul.py --all 2>&1 | tee -a "$LOG_FILE"
  python src/train_gpt4rul.py --all 2>&1 | tee -a "$LOG_FILE"
  python scripts/evaluate_all.py 2>&1 | tee -a "$LOG_FILE"
fi

log "=== Done ==="
log "Results: results/gpt4rul_summary.csv"
