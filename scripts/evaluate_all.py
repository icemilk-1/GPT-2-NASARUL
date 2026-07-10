"""Evaluate all datasets and write results/gpt4rul_summary.csv."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_gpt4rul import ALL_DATASETS, RESULTS_DIR
from src.evaluate_gpt4rul import evaluate


def main() -> None:
    rows = []
    for ds in ALL_DATASETS:
        print(f"\n{'='*40}\nEvaluating {ds}\n{'='*40}")
        result = evaluate(ds)
        train_json = RESULTS_DIR / f"test_gpt4rul_{ds}.json"
        best_epoch = train_time = val_mode = early_stop = ""
        if train_json.exists():
            with open(train_json, encoding="utf-8") as f:
                t = json.load(f)
            best_epoch = t.get("best_epoch", "")
            train_time = t.get("train_time_sec", "")
            val_mode = t.get("val_mode", "")
            early_stop = t.get("early_stop_metric", "")
        rows.append({
            "dataset": ds,
            "test_rmse": f"{result['test_rmse']:.4f}",
            "test_score": f"{result['test_score']:.2f}",
            "best_epoch": best_epoch,
            "train_time_sec": train_time,
            "val_mode": val_mode,
            "early_stop_metric": early_stop,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "gpt4rul_summary.csv"
    fields = ["dataset", "test_rmse", "test_score", "best_epoch", "train_time_sec", "val_mode", "early_stop_metric"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
