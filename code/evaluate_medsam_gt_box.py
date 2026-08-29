import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

from vividmed_lite_seg_common import load_medsam, medsam_image_tensor


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mask_scores(prediction, target, eps=1e-6):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (
        prediction.sum() + target.sum() + eps
    )
    return float(iou), float(dice)


def clamp_box(box, width, height):
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return [x1, y1, x2, y2]


def save_overlay(image, gold, prediction, box, path):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    result = base.copy()
    gold = gold.astype(bool)
    prediction = prediction.astype(bool)
    result[gold] = 0.55 * result[gold] + 0.45 * np.array([0, 255, 0])
    result[prediction] = 0.55 * result[prediction] + 0.45 * np.array([255, 0, 0])
    overlap = gold & prediction
    result[overlap] = 0.45 * base[overlap] + 0.55 * np.array([255, 255, 0])
    output = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)
    draw.rectangle(box, outline=(0, 140, 255), width=2)
    output.save(path)


@torch.no_grad()
def predict(model, image, box, device, threshold):
    width, height = image.size
    pixels = medsam_image_tensor(image).unsqueeze(0).to(device, dtype=torch.float32)
    image_embedding = model.image_encoder(pixels)
    scale = torch.tensor(
        [1024.0 / width, 1024.0 / height, 1024.0 / width, 1024.0 / height],
        device=device,
    )
    box_tensor = torch.tensor(box, dtype=torch.float32, device=device).unsqueeze(0)
    box_tensor = (box_tensor * scale).unsqueeze(1)
    sparse, dense = model.prompt_encoder(points=None, boxes=box_tensor, masks=None)
    logits, quality = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
    probability = logits.sigmoid()[0, 0].cpu().numpy()
    return (probability >= threshold).astype(np.uint8), float(quality[0, 0].item())


def main():
    parser = argparse.ArgumentParser(description="Evaluate the MedSAM GT-box upper bound.")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--test-file", default="validation.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bbox-field", default="bbox_from_mask_xyxy")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows = load_jsonl(data_root / args.test_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "pred"
    overlay_dir = output_dir / "overlay"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_medsam(args.sam_checkpoint, args.sam_model_type).to(device)
    model.eval()
    totals = defaultdict(float)
    per_region = defaultdict(lambda: defaultdict(float))
    results = []

    for row in tqdm(rows, desc="MedSAM GT box"):
        image = Image.open(data_root / row["image"]).convert("RGB")
        width, height = image.size
        if args.bbox_field not in row:
            raise KeyError(f"{row.get('id')} has no field {args.bbox_field}")
        box = clamp_box(row[args.bbox_field], width, height)
        prediction, quality = predict(model, image, box, device, args.threshold)
        gold = np.asarray(Image.open(data_root / row["mask"]).convert("L")) > 0
        iou, dice = mask_scores(prediction, gold)
        region = row["region"]

        totals["n"] += 1
        totals["iou"] += iou
        totals["dice"] += dice
        totals["iou_05"] += int(iou >= 0.5)
        per_region[region]["n"] += 1
        per_region[region]["iou"] += iou
        per_region[region]["dice"] += dice

        pred_path = mask_dir / f"{row['id']}.png"
        overlay_path = overlay_dir / f"{row['id']}.jpg"
        Image.fromarray(prediction * 255).save(pred_path)
        save_overlay(image, gold, prediction, box, overlay_path)
        results.append(
            {
                "id": row["id"],
                "image": row["image"],
                "region": region,
                "bbox": box,
                "mask_iou": iou,
                "mask_dice": dice,
                "predicted_quality": quality,
                "pred_mask": str(pred_path),
                "overlay": str(overlay_path),
            }
        )

    n = max(int(totals["n"]), 1)
    region_summary = {}
    for region, values in sorted(per_region.items()):
        region_n = max(int(values["n"]), 1)
        region_summary[region] = {
            "n": int(values["n"]),
            "mean_iou": values["iou"] / region_n,
            "mean_dice": values["dice"] / region_n,
        }
    summary = {
        "sam_checkpoint": args.sam_checkpoint,
        "test_file": str(data_root / args.test_file),
        "bbox_field": args.bbox_field,
        "threshold": args.threshold,
        "num_samples": int(totals["n"]),
        "mean_mask_iou": totals["iou"] / n,
        "mean_dice": totals["dice"] / n,
        "mask_iou_at_0_5": totals["iou_05"] / n,
        "per_region": region_summary,
    }
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
