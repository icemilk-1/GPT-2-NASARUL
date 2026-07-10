#!/usr/bin/env bash
# GPT4RUL one-click environment setup (Linux/Mac)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== GPT4RUL Setup ==="
python3 --version

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -r requirements.txt -q
python scripts/download_gpt2.py
python scripts/check_data.py

echo "=== Setup complete ==="
echo "Next: bash scripts/reproduce.sh"
