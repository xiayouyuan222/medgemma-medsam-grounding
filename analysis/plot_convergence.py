import argparse
import ast
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    weights = np.ones(window) / window
    valid = np.convolve(values, weights, mode="valid")
    left = window // 2
    right = len(values) - len(valid) - left
    return np.pad(valid, (left, right), mode="edge")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Trainer log containing Python dictionaries")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--smooth-window", type=int, default=15)
    args = parser.parse_args()

    train = []
    validation = []
    for line in args.input.read_text(encoding="utf-8-sig").splitlines():
        match = re.search(r"(\{.*\})", line)
        if not match:
            continue
        try:
            row = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if "loss" in row and "epoch" in row:
            train.append((float(row["epoch"]), float(row["loss"])))
        if "eval_loss" in row and "epoch" in row:
            validation.append((float(row["epoch"]), float(row["eval_loss"])))

    if not train or not validation:
        raise RuntimeError("Could not find both training and validation losses.")

    train_epoch = np.array([x[0] for x in train])
    train_loss = np.array([x[1] for x in train])
    val_epoch = np.array([x[0] for x in validation])
    val_loss = np.array([x[1] for x in validation])
    smooth_loss = moving_average(train_loss, window=args.smooth_window)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1]},
    )
    ax1.plot(train_epoch, train_loss, color="#78A7D0", alpha=0.35,
             linewidth=1.0, label="Training loss (logged batches)")
    ax1.plot(train_epoch, smooth_loss, color="#1368AA", linewidth=2.5,
             label=f"Training loss ({args.smooth_window}-point moving average)")
    ax1.plot(val_epoch, val_loss, color="#D1495B", marker="o",
             markersize=6, linewidth=2.2, label="Validation loss")
    ax1.set_ylabel("Loss")
    ax1.set_title("14-Class TotalSegmentator Training Convergence")
    ax1.legend(frameon=True, loc="upper right")

    ax2.plot(val_epoch, val_loss, color="#D1495B", marker="o",
             markersize=7, linewidth=2.4)
    for epoch, loss in validation:
        ax2.annotate(f"{loss:.3f}", (epoch, loss), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=9)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation loss")
    ax2.set_title("Validation Loss Detail")
    ax2.set_xlim(0, max(train_epoch.max(), val_epoch.max()))
    margin = max(0.01, (val_loss.max() - val_loss.min()) * 0.25)
    ax2.set_ylim(val_loss.min() - margin, val_loss.max() + margin)

    fig.text(0.5, 0.01,
             f"Validation loss changed from {val_loss[0]:.3f} to {val_loss[-1]:.3f}.",
             ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "totalseg14_convergence.png"
    pdf_path = args.output_dir / "totalseg14_convergence.pdf"
    csv_path = args.output_dir / "totalseg14_loss_points.csv"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "epoch", "loss"])
        writer.writerows(("train", epoch, loss) for epoch, loss in train)
        writer.writerows(("validation", epoch, loss) for epoch, loss in validation)

    print(f"train_points={len(train)}")
    print(f"validation_points={len(validation)}")
    print(f"first_validation_loss={val_loss[0]:.6f}")
    print(f"last_validation_loss={val_loss[-1]:.6f}")
    print(png_path)
    print(pdf_path)
    print(csv_path)


if __name__ == "__main__":
    main()
