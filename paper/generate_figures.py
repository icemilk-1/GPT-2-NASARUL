"""Generate publication figures for revisiting_gpt4rul.tex from results JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PAPER_RMSE = {"FD001": 11.23, "FD002": 11.05, "FD003": 11.45, "FD004": 12.68}
DATASETS = ["FD001", "FD002", "FD003", "FD004"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "paper": "#4878A8",
    "repro": "#E07A5F",
    "lastwin": "#C1121F",
    "allval": "#2A9D8F",
    "train": "#6C757D",
    "val_last": "#E63946",
    "test": "#1D3557",
}


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig3_validation_trap() -> None:
    """Panel (a): RUL decay schematic; Panel (b): last_window vs all-val RMSE."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    ax = axes[0]
    cycles = np.arange(0, 200)
    rul = np.clip(125 - np.maximum(0, cycles - 30), 0, 125)
    ax.plot(cycles, rul, color="#333333", lw=1.8, label="True RUL")
    ax.axvspan(0, 136, alpha=0.12, color=COLORS["train"], label="Train windows")
    ax.axvspan(136, 170, alpha=0.15, color=COLORS["val_last"], label="Val (last-window only)")
    ax.axvline(199, color=COLORS["test"], ls="--", lw=1.5, label="Test (last window)")
    ax.scatter([199], [0], s=60, c=COLORS["test"], zorder=5, edgecolors="white")
    ax.scatter([170], [0], s=60, c=COLORS["val_last"], zorder=5, edgecolors="white")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")
    ax.set_title("(a) RUL decay and window placement")
    ax.set_xlim(0, 210)
    ax.set_ylim(-5, 135)
    ax.legend(loc="upper right", framealpha=0.9)

    ax = axes[1]
    lastwin, allval = [], []
    for ds in DATASETS[:3]:
        lw = load_json(RESULTS / "ablation" / f"test_gpt4rul_{ds}_best.json")
        av = load_json(RESULTS / f"test_gpt4rul_{ds}.json")
        lastwin.append(lw["test_rmse"])
        allval.append(av["test_rmse"])

    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, lastwin, w, label="last_window val", color=COLORS["lastwin"])
    ax.bar(x + w / 2, allval, w, label="all-val (engine-disjoint)", color=COLORS["allval"])
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS[:3])
    ax.set_ylabel("Test RMSE")
    ax.set_title("(b) Validation protocol impact")
    ax.legend(framealpha=0.9)
    for i, (lw, av) in enumerate(zip(lastwin, allval)):
        ax.text(i - w / 2, lw + 0.4, f"{lw:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, av + 0.4, f"{av:.1f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_validation_trap.pdf")
    fig.savefig(OUT / "fig3_validation_trap.png")
    plt.close(fig)


def fig4_rmse_comparison() -> None:
    repro = []
    for ds in DATASETS:
        d = load_json(RESULTS / f"test_gpt4rul_{ds}.json")
        repro.append(d["test_rmse"])

    paper = [PAPER_RMSE[ds] for ds in DATASETS]
    x = np.arange(len(DATASETS))
    w = 0.35

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(x - w / 2, paper, w, label="Tan et al. (2025)", color=COLORS["paper"])
    ax.bar(x + w / 2, repro, w, label="This reproduction", color=COLORS["repro"])

    for i, (p, r) in enumerate(zip(paper, repro)):
        gap = r - p
        ax.text(i + w / 2, r + 0.25, f"+{gap:.2f}", ha="center", va="bottom", fontsize=7, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel("Test RMSE")
    ax.set_title("Reproduction vs. reported results (GPT4RUL + condition embedding)")
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 18)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_rmse_comparison.pdf")
    fig.savefig(OUT / "fig4_rmse_comparison.png")
    plt.close(fig)


def fig5_rul_schematic() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    axes = axes.flatten()
    titles = [
        "FD001 (1 op. / 1 fault)", "FD002 (6 op. / 1 fault)",
        "FD003 (1 op. / 2 faults)", "FD004 (6 op. / 2 faults)",
    ]
    rng = np.random.default_rng(42)

    for ax, title in zip(axes, titles):
        t = np.linspace(0, 1, 80)
        true = np.clip(125 * (1 - t) ** 0.85, 0, 125)
        noise = rng.normal(0, 3, size=true.shape)
        pred = np.clip(true + noise + 2 * np.sin(t * 8), 0, 125)
        ax.plot(t * 100, true, "k-", lw=1.5, label="True RUL")
        ax.plot(t * 100, pred, color=COLORS["repro"], lw=1.2, alpha=0.85, label="Predicted RUL")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("RUL")
        ax.set_ylim(-5, 140)
        if ax in (axes[-1], axes[-2]):
            ax.set_xlabel("Normalized engine life (%)")

    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle("Illustrative RUL prediction trends on test engines (schematic)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_rul_trends.pdf")
    fig.savefig(OUT / "fig5_rul_trends.png")
    plt.close(fig)


def fig6_training_curves() -> None:
    normal = load_json(RESULTS / "test_gpt4rul_FD001.json")
    lastwin = load_json(RESULTS / "ablation" / "test_gpt4rul_FD001_best.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    for ax, data, title in zip(
        axes,
        [normal, lastwin],
        [
            "(a) val_mode=all, engine-disjoint split",
            "(b) val_mode=last_window (failed protocol)",
        ],
    ):
        hist = data["history"]
        epochs = [h["epoch"] for h in hist]
        train_loss = [h["train_loss"] for h in hist]
        val_rmse = [h["val_rmse"] for h in hist]
        best = data["best_epoch"]

        ax2 = ax.twinx()
        l1, = ax.plot(epochs, train_loss, color=COLORS["train"], lw=1.5, label="Train loss")
        l2, = ax2.plot(epochs, val_rmse, color=COLORS["repro"], lw=1.5, label="Val RMSE")
        ax.axvline(best, color=COLORS["lastwin"], ls=":", lw=1.2, label=f"Best epoch ({best})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss", color=COLORS["train"])
        ax2.set_ylabel("Val RMSE", color=COLORS["repro"])
        ax.set_title(title, fontsize=9)
        ax.legend([l1, l2], ["Train loss", "Val RMSE"], loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fig6_training_curves.pdf")
    fig.savefig(OUT / "fig6_training_curves.png")
    plt.close(fig)


def main() -> None:
    fig3_validation_trap()
    fig4_rmse_comparison()
    fig5_rul_schematic()
    fig6_training_curves()
    print(f"Figures saved to {OUT}")


if __name__ == "__main__":
    main()
