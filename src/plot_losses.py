import argparse
import csv
import os

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DETR losses from CSV.")
    parser.add_argument("--losses-csv", default="artifacts/detr_2/losses.csv")
    parser.add_argument("--output", default="artifacts/detr_2/loss_curves.png")
    return parser.parse_args()


def load_losses(path: str) -> dict:
    data = {"train": [], "val": []}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            row_parsed = {k: float(row[k]) if k not in {"epoch", "split"} else row[k] for k in row}
            row_parsed["epoch"] = int(row["epoch"])
            data[split].append(row_parsed)
    return data


def main() -> int:
    args = parse_args()
    data = load_losses(args.losses_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for split, series in data.items():
        epochs = [row["epoch"] for row in series]
        axes[0].plot(epochs, [row["loss_ce"] for row in series], label=f"{split}")
        axes[1].plot(epochs, [row["loss_bbox"] for row in series], label=f"{split}")

    axes[0].set_title("Classification loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss_ce")
    axes[0].legend()

    axes[1].set_title("BBox regression loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss_bbox")
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
