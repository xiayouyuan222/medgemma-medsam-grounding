import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from PIL import Image


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metric(row, *names):
    for name in names:
        value = row.get(name)
        if value is not None:
            return float(value)
    raise KeyError(f"None of {names} is present in result row {row.get('id')}")


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def bootstrap_ci(values, samples=2000, seed=42):
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(mean(rng.choice(values) for _ in values))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def subject_id(row):
    for key in ("subject", "subject_id", "patient", "patient_id", "case_id"):
        if row.get(key):
            return str(row[key])
    text = " ".join(str(row.get(key, "")) for key in ("id", "image", "mask"))
    match = re.search(r"(?<![A-Za-z0-9])s\d{4}(?!\d)", text, flags=re.I)
    return match.group(0).lower() if match else "unknown"


def source_lookup(rows):
    by_id = {}
    by_image_region = {}
    for row in rows:
        if row.get("id") is not None:
            by_id[str(row["id"])] = row
        key = (str(row.get("image", "")), str(row.get("region", "")))
        by_image_region[key] = row
    return by_id, by_image_region


def enrich_results(results, source_rows, data_root):
    by_id, by_image_region = source_lookup(source_rows)
    data_root = Path(data_root)
    enriched = []
    for result in results:
        source = by_id.get(str(result.get("id")))
        if source is None:
            source = by_image_region.get(
                (str(result.get("image", "")), str(result.get("region", "")))
            )
        combined = dict(source or {})
        combined.update(result)
        combined["mask_iou"] = metric(combined, "mask_iou", "iou")
        combined["mask_dice"] = metric(combined, "mask_dice", "dice")
        combined["subject_id"] = subject_id(combined)
        pixels = combined.get("mask_pixels") or combined.get("target_mask_pixels")
        if pixels is None and combined.get("mask"):
            mask_path = data_root / combined["mask"]
            with Image.open(mask_path).convert("L") as mask:
                pixels = sum(value > 0 for value in mask.getdata())
        combined["mask_pixels"] = int(pixels) if pixels is not None else None
        enriched.append(combined)
    return enriched


def summarize(rows):
    ious = [row["mask_iou"] for row in rows]
    dices = [row["mask_dice"] for row in rows]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["region"]].append(row)
    region_ious = [mean(item["mask_iou"] for item in group) for group in grouped.values()]
    region_dices = [mean(item["mask_dice"] for item in group) for group in grouped.values()]
    iou_low, iou_high = bootstrap_ci(ious)
    dice_low, dice_high = bootstrap_ci(dices)
    return {
        "n": len(rows),
        "regions": len(grouped),
        "mean_iou": mean(ious),
        "iou_ci_low": iou_low,
        "iou_ci_high": iou_high,
        "median_iou": median(ious),
        "macro_iou": mean(region_ious),
        "mean_dice": mean(dices),
        "dice_ci_low": dice_low,
        "dice_ci_high": dice_high,
        "median_dice": median(dices),
        "macro_dice": mean(region_dices),
        "iou_at_0_5": mean(value >= 0.5 for value in ious),
        "failure_iou_lt_0_1": mean(value < 0.1 for value in ious),
    }


def per_region_rows(split, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["region"]].append(row)
    output = []
    for region, group in groups.items():
        ious = [row["mask_iou"] for row in group]
        dices = [row["mask_dice"] for row in group]
        pixels = [row["mask_pixels"] for row in group if row["mask_pixels"] is not None]
        output.append({
            "split": split,
            "region": region,
            "n": len(group),
            "mean_iou": mean(ious),
            "median_iou": median(ious),
            "mean_dice": mean(dices),
            "median_dice": median(dices),
            "iou_at_0_5": mean(value >= 0.5 for value in ious),
            "failure_iou_lt_0_1": mean(value < 0.1 for value in ious),
            "median_mask_pixels": median(pixels) if pixels else "",
        })
    return sorted(output, key=lambda row: row["mean_iou"])


def per_patient_rows(split, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["subject_id"]].append(row)
    output = []
    for subject, group in groups.items():
        output.append({
            "split": split,
            "subject_id": subject,
            "n": len(group),
            "mean_iou": mean(row["mask_iou"] for row in group),
            "mean_dice": mean(row["mask_dice"] for row in group),
        })
    return sorted(output, key=lambda row: row["mean_iou"])


def size_bin_rows(split, rows, thresholds):
    labels = ("small", "medium", "large", "very_large")
    groups = defaultdict(list)
    for row in rows:
        pixels = row["mask_pixels"]
        if pixels is None:
            continue
        index = sum(pixels > threshold for threshold in thresholds)
        groups[labels[index]].append(row)
    output = []
    for label in labels:
        group = groups[label]
        if not group:
            continue
        output.append({
            "split": split,
            "size_bin": label,
            "n": len(group),
            "mean_iou": mean(row["mask_iou"] for row in group),
            "mean_dice": mean(row["mask_dice"] for row in group),
            "median_mask_pixels": median(row["mask_pixels"] for row in group),
        })
    return output


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-results")
    parser.add_argument("--validation-results", required=True)
    parser.add_argument("--test-results", required=True)
    parser.add_argument("--train-source", default="train.jsonl")
    parser.add_argument("--validation-source", default="validation.jsonl")
    parser.add_argument("--test-source", default="test.jsonl")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specifications = [
        ("validation", args.validation_results, args.validation_source),
        ("test", args.test_results, args.test_source),
    ]
    if args.train_results:
        specifications.insert(0, ("train", args.train_results, args.train_source))

    splits = {}
    for split, result_path, source_name in specifications:
        splits[split] = enrich_results(
            read_jsonl(result_path), read_jsonl(data_root / source_name), data_root
        )

    train_pixels = [
        row["mask_pixels"] for row in splits.get("train", splits["validation"])
        if row["mask_pixels"] is not None
    ]
    thresholds = [percentile(train_pixels, q) for q in (0.25, 0.5, 0.75)]

    overall = []
    regions = []
    patients = []
    size_bins = []
    failures = []
    for split, rows in splits.items():
        overall.append({"split": split, **summarize(rows)})
        regions.extend(per_region_rows(split, rows))
        patients.extend(per_patient_rows(split, rows))
        size_bins.extend(size_bin_rows(split, rows, thresholds))
        failures.extend(
            {"split": split, **row}
            for row in sorted(rows, key=lambda item: item["mask_iou"])[:50]
        )

    write_csv(output_dir / "overall.csv", overall)
    write_csv(output_dir / "per_region.csv", regions)
    write_csv(output_dir / "per_patient.csv", patients)
    write_csv(output_dir / "size_bins.csv", size_bins)
    failure_columns = (
        "split", "id", "subject_id", "region", "image", "mask",
        "mask_pixels", "mask_iou", "mask_dice",
    )
    write_csv(
        output_dir / "worst_samples.csv",
        [{key: row.get(key, "") for key in failure_columns} for row in failures],
    )

    report = ["# Fourteen-class segmentation analysis", ""]
    report.append("## Overall results")
    report.append("")
    report.append("| Split | N | Mean IoU (95% CI) | Macro IoU | Mean Dice (95% CI) | Macro Dice | IoU@0.5 | IoU<0.1 |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in overall:
        report.append(
            f"| {row['split']} | {row['n']} | {row['mean_iou']:.3f} "
            f"[{row['iou_ci_low']:.3f}, {row['iou_ci_high']:.3f}] | "
            f"{row['macro_iou']:.3f} | {row['mean_dice']:.3f} "
            f"[{row['dice_ci_low']:.3f}, {row['dice_ci_high']:.3f}] | "
            f"{row['macro_dice']:.3f} | {row['iou_at_0_5']:.3f} | "
            f"{row['failure_iou_lt_0_1']:.3f} |"
        )
    report.extend(["", "## Mask-size analysis", ""])
    report.append(
        "Training-derived mask-pixel quartiles: "
        + ", ".join(f"{value:.0f}" for value in thresholds)
    )
    report.append("")
    report.append("| Split | Size bin | N | Mean IoU | Mean Dice | Median pixels |")
    report.append("|---|---|---:|---:|---:|---:|")
    for row in size_bins:
        report.append(
            f"| {row['split']} | {row['size_bin']} | {row['n']} | "
            f"{row['mean_iou']:.3f} | {row['mean_dice']:.3f} | "
            f"{row['median_mask_pixels']:.0f} |"
        )
    if "train" in splits:
        train_summary = summarize(splits["train"])
        test_summary = summarize(splits["test"])
        report.extend(["", "## Generalization gaps", ""])
        report.append(
            f"- Train-test mean IoU gap: "
            f"{train_summary['mean_iou'] - test_summary['mean_iou']:+.3f}"
        )
        report.append(
            f"- Train-test mean Dice gap: "
            f"{train_summary['mean_dice'] - test_summary['mean_dice']:+.3f}"
        )
    report.extend([
        "",
        "## Files",
        "",
        "- `per_region.csv`: class-level performance and failure rates.",
        "- `per_patient.csv`: patient-level stability.",
        "- `size_bins.csv`: dependence on target-mask size.",
        "- `worst_samples.csv`: 50 lowest-IoU samples per split.",
    ])
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
