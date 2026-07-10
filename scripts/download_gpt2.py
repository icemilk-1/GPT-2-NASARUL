"""Download and cache GPT-2 to .hf_cache/gpt2 for offline use."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".hf_cache" / "gpt2"
MODEL_ID = "openai-community/gpt2"


def main() -> None:
    if (CACHE_DIR / "config.json").exists():
        print(f"GPT-2 already cached at {CACHE_DIR}")
        return

    print(f"Downloading {MODEL_ID} → {CACHE_DIR} ...")
    try:
        from transformers import GPT2Model, GPT2Tokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model = GPT2Model.from_pretrained(MODEL_ID)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    model.save_pretrained(CACHE_DIR)
    tokenizer.save_pretrained(CACHE_DIR)
    print(f"Done. GPT-2 saved to {CACHE_DIR}")


if __name__ == "__main__":
    main()
