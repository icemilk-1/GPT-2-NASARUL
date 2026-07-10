"""Run GPT4RUL ablation experiments into outputs/ablation/ and results/ablation/."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_gpt4rul import (
    ABLATION_OUTPUT_DIR,
    ABLATION_RESULTS_DIR,
    ALL_DATASETS,
)
from src.preprocess_gpt4rul import run_preprocess
from src.train_gpt4rul import TrainConfig, train_one


def run_experiment(
    dataset_ids: list[str],
    val_mode: str,
    early_stop_metric: str,
    pcf_style: str = "sequential",
    grad_clip: float = 1.0,
    tag_suffix: str = "",
    device: str | None = None,
) -> list[dict]:
    tag = tag_suffix or f"{val_mode}_{early_stop_metric}"
    out_dir = ABLATION_OUTPUT_DIR
    res_dir = ABLATION_RESULTS_DIR
    ckpt_dir = out_dir / "checkpoints"

    for ds in dataset_ids:
        run_preprocess(ds, output_dir=out_dir)

    cfg = TrainConfig(
        output_dir=out_dir,
        results_dir=res_dir,
        checkpoint_dir=ckpt_dir,
        val_mode=val_mode,
        early_stop_metric=early_stop_metric,
        pcf_style=pcf_style,
        grad_clip=grad_clip,
        tag_suffix=tag,
    )

    results = []
    for ds in dataset_ids:
        results.append(train_one(ds, device, cfg))
    return results


def main():
    p = argparse.ArgumentParser(description="GPT4RUL ablation runner")
    p.add_argument("--device", default=None)
    p.add_argument("--screen-fd001", action="store_true", help="Screen val protocols on FD001")
    p.add_argument("--all", action="store_true", help="Run best config on all 4 datasets")
    args = p.parse_args()

    ABLATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    if args.screen_fd001:
        configs = [
            ("all", "val_loss", "sequential", 1.0, "all_valloss"),
            ("last_window", "val_loss", "sequential", 1.0, "lastwin_valloss"),
            ("last_window", "val_rmse", "sequential", 1.0, "lastwin_valrmse"),
            ("last_window", "val_rmse", "parallel", 1.0, "lastwin_valrmse_parallel"),
            ("last_window", "val_rmse", "sequential", 0.0, "lastwin_valrmse_noclip"),
        ]
        for val_mode, metric, pcf, clip, tag in configs:
            print(f"\n{'='*60}\nAblation FD001: {tag}\n{'='*60}")
            rows = run_experiment(
                ["FD001"], val_mode, metric, pcf, clip, tag, args.device,
            )
            all_rows.extend(rows)

    if args.all:
        print(f"\n{'='*60}\nBest config on all datasets\n{'='*60}")
        rows = run_experiment(
            ALL_DATASETS, "last_window", "val_rmse", "sequential", 1.0, "best", args.device,
        )
        all_rows.extend(rows)

    if not args.screen_fd001 and not args.all:
        rows = run_experiment(
            ALL_DATASETS, "last_window", "val_rmse", "sequential", 1.0, "best", args.device,
        )
        all_rows.extend(rows)

    summary_path = ABLATION_RESULTS_DIR / "ablation_summary.csv"
    if all_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nAblation summary → {summary_path}")


if __name__ == "__main__":
    main()
