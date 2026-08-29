#!/usr/bin/env python3
"""Ablate the two visual branches in the MedGemma-MedSAM grounding model.

This evaluator keeps the phrase and target mask fixed while independently
replacing the image seen by MedGemma and the image seen by MedSAM. It is a
no-retraining diagnostic for determining whether both visual encoders
contribute to the final mask.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from test_visual_grounding_causality import (
    binary_iou,
    decode_hidden,
    extract_hidden,
    load_jsonl,
    load_model,
    prepare_sam_context,
)
from vividmed_lite_seg_common import medsam_image_tensor, pooled_phrase_hidden


MODES = (
    "full",
    "vlm_image_shuffled",
    "vlm_image_blank",
    "sam_image_shuffled",
    "both_images_shuffled",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether the MedGemma and MedSAM image branches are necessary."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--test-file", default="validation.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument(
        "--prompt-mode",
        choices=("phrase-only", "original-question"),
        default="phrase-only",
    )
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(prediction, target).sum(dtype=np.float64)
    denominator = prediction.sum(dtype=np.float64) + target.sum(dtype=np.float64)
    return float((2.0 * intersection + 1e-8) / (denominator + 1e-8))


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    denominator = left.norm() * right.norm()
    if denominator.item() == 0:
        return 0.0
    return float(torch.dot(left, right).div(denominator).item())


def patient_key(record: dict) -> str | None:
    for key in ("patient_id", "subject_id", "case_id", "study_id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return None


def make_donor_map(groups: dict[str, list[dict]], seed: int) -> dict[str, str]:
    image_paths = sorted(groups)
    if len(image_paths) < 2:
        raise ValueError("At least two distinct images are required for image shuffling.")

    rng = random.Random(seed)
    donors: dict[str, str] = {}
    for image_path in image_paths:
        source_patient = patient_key(groups[image_path][0])
        candidates = [path for path in image_paths if path != image_path]
        different_patient = [
            path
            for path in candidates
            if source_patient is None
            or patient_key(groups[path][0]) is None
            or patient_key(groups[path][0]) != source_patient
        ]
        donors[image_path] = rng.choice(different_patient or candidates)
    return donors


def paired_bootstrap_ci(
    full_values: np.ndarray,
    ablated_values: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    differences = full_values - ablated_values
    if not len(differences) or repeats <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled_means[index] = rng.choice(
            differences, size=len(differences), replace=True
        ).mean()
    low, high = np.quantile(sampled_means, (0.025, 0.975))
    return float(low), float(high)


def aggregate(rows: list[dict], mode: str) -> dict:
    ious = np.asarray([row["metrics"][mode]["iou"] for row in rows])
    dices = np.asarray([row["metrics"][mode]["dice"] for row in rows])
    foreground = np.asarray(
        [row["metrics"][mode]["foreground_ratio"] for row in rows]
    )
    return {
        "mean_iou": float(ious.mean()),
        "mean_dice": float(dices.mean()),
        "mask_iou_at_0_5": float((ious >= 0.5).mean()),
        "mean_foreground_ratio": float(foreground.mean()),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    test_file = resolve_path(data_root, args.test_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(test_file)
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        image_path = str(resolve_path(data_root, record["image"]).resolve())
        groups[image_path].append(record)

    selected_paths = sorted(groups)
    if args.limit_images is not None:
        selected_paths = selected_paths[: args.limit_images]
        groups = {path: groups[path] for path in selected_paths}

    donor_map = make_donor_map(groups, args.seed)
    model, processor, open_token_ids, close_token_ids, device = load_model(args)
    details: list[dict] = []

    for image_path in tqdm(selected_paths, desc="Dual-vision ablation"):
        source_image = Image.open(image_path).convert("RGB")
        donor_path = donor_map[image_path]
        donor_image = Image.open(donor_path).convert("RGB")
        blank_image = Image.new("RGB", source_image.size, color=(0, 0, 0))

        source_sam_pixels = medsam_image_tensor(source_image).unsqueeze(0).to(device)
        donor_sam_pixels = medsam_image_tensor(donor_image).unsqueeze(0).to(device)
        with torch.inference_mode():
            source_sam_context = prepare_sam_context(model, source_sam_pixels)
            donor_sam_context = prepare_sam_context(model, donor_sam_pixels)

        for record in groups[image_path]:
            mask_path = resolve_path(data_root, record["mask"])
            target = load_mask(mask_path)
            height, width = target.shape

            with torch.inference_mode():
                full_hidden = extract_hidden(
                    model,
                    processor,
                    open_token_ids,
                    close_token_ids,
                    record,
                    source_image,
                    args.prompt_mode,
                )
                shuffled_hidden = extract_hidden(
                    model,
                    processor,
                    open_token_ids,
                    close_token_ids,
                    record,
                    donor_image,
                    args.prompt_mode,
                )
                blank_hidden = extract_hidden(
                    model,
                    processor,
                    open_token_ids,
                    close_token_ids,
                    record,
                    blank_image,
                    args.prompt_mode,
                )

                predictions = {
                    "full": decode_hidden(
                        model,
                        full_hidden,
                        source_sam_context,
                        height,
                        width,
                        args.threshold,
                    ),
                    "vlm_image_shuffled": decode_hidden(
                        model,
                        shuffled_hidden,
                        source_sam_context,
                        height,
                        width,
                        args.threshold,
                    ),
                    "vlm_image_blank": decode_hidden(
                        model,
                        blank_hidden,
                        source_sam_context,
                        height,
                        width,
                        args.threshold,
                    ),
                    "sam_image_shuffled": decode_hidden(
                        model,
                        full_hidden,
                        donor_sam_context,
                        height,
                        width,
                        args.threshold,
                    ),
                    "both_images_shuffled": decode_hidden(
                        model,
                        shuffled_hidden,
                        donor_sam_context,
                        height,
                        width,
                        args.threshold,
                    ),
                }

            metrics = {}
            for mode, prediction in predictions.items():
                metrics[mode] = {
                    "iou": binary_iou(prediction, target),
                    "dice": dice_score(prediction, target),
                    "foreground_ratio": float(prediction.mean()),
                }

            full_pooled = pooled_phrase_hidden(full_hidden)
            shuffled_pooled = pooled_phrase_hidden(shuffled_hidden)
            blank_pooled = pooled_phrase_hidden(blank_hidden)
            details.append(
                {
                    "id": record.get("id"),
                    "region": record.get("region"),
                    "image": image_path,
                    "donor_image": donor_path,
                    "source_patient": patient_key(record),
                    "donor_patient": patient_key(groups[donor_path][0]),
                    "hidden_cosine": {
                        "full_vs_vlm_shuffled": cosine_similarity(
                            full_pooled, shuffled_pooled
                        ),
                        "full_vs_vlm_blank": cosine_similarity(
                            full_pooled, blank_pooled
                        ),
                    },
                    "metrics": metrics,
                }
            )

    details_path = output_dir / "details.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    mode_summary = {mode: aggregate(details, mode) for mode in MODES}
    full_ious = np.asarray([row["metrics"]["full"]["iou"] for row in details])
    for index, mode in enumerate(MODES[1:], start=1):
        ablated_ious = np.asarray(
            [row["metrics"][mode]["iou"] for row in details]
        )
        low, high = paired_bootstrap_ci(
            full_ious,
            ablated_ious,
            args.bootstrap_repeats,
            args.seed + index,
        )
        mode_summary[mode]["mean_iou_drop_from_full"] = float(
            (full_ious - ablated_ious).mean()
        )
        mode_summary[mode]["iou_drop_bootstrap_95_ci"] = [low, high]

    hidden_summary = {
        key: float(np.mean([row["hidden_cosine"][key] for row in details]))
        for key in ("full_vs_vlm_shuffled", "full_vs_vlm_blank")
    }
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "test_file": str(test_file),
        "prompt_mode": args.prompt_mode,
        "threshold": args.threshold,
        "num_images": len(selected_paths),
        "num_samples": len(details),
        "donor_policy": "different image; different patient when metadata permits",
        "modes": mode_summary,
        "hidden_state_cosine": hidden_summary,
        "interpretation": {
            "medgemma_vision_used": (
                "Supported when VLM image shuffle/blank causes a positive IoU drop "
                "whose paired 95% CI excludes zero."
            ),
            "medsam_vision_used": (
                "Supported when SAM image shuffle causes a large positive IoU drop."
            ),
            "medgemma_vision_redundant": (
                "Suggested when VLM shuffle/blank has near-zero IoU effect and phrase "
                "hidden-state cosine remains near one."
            ),
            "scope": "Teacher-forced phrase grounding; no free-form generation.",
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Detailed results: {details_path}")


if __name__ == "__main__":
    main()
