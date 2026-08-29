import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from vividmed_lite_seg_common import (
    CLOSE_TAG,
    OPEN_TAG,
    SYSTEM_PROMPT,
    PhraseBBoxHead,
    PhraseProjector,
    DirectPixelMaskHead,
    GatedSemanticSpatialAligner,
    build_open_tag_token_variants,
    get_hidden_size,
    get_tokenizer,
    load_binary_mask,
    load_medsam,
    load_phrase_projector_state,
    medsam_image_tensor,
    norm_text,
    resize_logits,
    select_phrase_hidden,
    select_phrase_token_hiddens,
    pooled_phrase_hidden,
)


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def user_text(record):
    for item in record["messages"][0]["content"]:
        if item.get("type") == "text":
            return item["text"]
    raise ValueError(f"No user text in {record['id']}")


def replay_messages(record, image, prompt_mode, use_vlm_image=True):
    # Use the gold region to isolate whether the localization branch follows
    # phrase semantics. This does not evaluate free-form phrase generation.
    assistant_text = (
        f"Relevant region: <p>{record['region']}</p>\n"
        f"Answer: {record['answer']}"
    )
    if prompt_mode == "phrase-only":
        prompt_text = (
            "Identify and ground the referenced medical region at pixel level.\n"
            "Use the exact text format:\n"
            "Relevant region: <p>region</p>\n"
            "Answer: answer\n"
            "Question: Locate the referenced medical region."
        )
    else:
        prompt_text = user_text(record)
    user_content = [{"type": "text", "text": prompt_text}]
    if use_vlm_image:
        user_content.append({"type": "image", "image": image})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        },
    ]


def binary_iou(prediction, target, eps=1e-6):
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return float((intersection + eps) / (union + eps))


def save_overlay(image, gold, prediction, path):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    result = base.copy()
    gold = gold.astype(bool)
    prediction = prediction.astype(bool)
    result[gold] = 0.55 * result[gold] + 0.45 * np.array([0, 255, 0])
    result[prediction] = 0.55 * result[prediction] + 0.45 * np.array([255, 0, 0])
    overlap = gold & prediction
    result[overlap] = 0.45 * base[overlap] + 0.55 * np.array([255, 255, 0])
    Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)).save(path)


def load_model(args):
    adapter = Path(args.adapter)
    meta = json.loads(
        (adapter / "vividmed_lite_seg_meta.json").read_text(encoding="utf-8")
    )
    processor = AutoProcessor.from_pretrained(adapter)
    tokenizer = get_tokenizer(processor)
    close_token_ids = meta.get("close_token_ids") or tokenizer.encode(
        CLOSE_TAG, add_special_tokens=False
    )
    open_token_ids = meta.get("open_token_ids") or build_open_tag_token_variants(
        tokenizer
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = load_medsam(args.sam_checkpoint, meta.get("sam_model_type", "vit_b")).to(
        device
    )
    num_prompt_tokens = int(meta.get("num_prompt_tokens", 1))
    model.phrase_projector = PhraseProjector(
        get_hidden_size(model.config), 256, num_prompt_tokens
    ).to(device)
    model.use_predicted_bbox = bool(meta.get("use_predicted_bbox", False))
    if model.use_predicted_bbox:
        model.phrase_bbox_head = PhraseBBoxHead(get_hidden_size(model.config)).to(device)
    model.localization_image_encoder = sam.image_encoder
    model.localization_prompt_encoder = sam.prompt_encoder
    model.localization_mask_decoder = sam.mask_decoder
    model.sam_image_features_enabled = bool(
        meta.get("sam_image_features_enabled", True)
    )
    model.grounding_decoder_mode = meta.get("grounding_decoder_mode", "medsam")
    model.vlm_image_enabled = bool(meta.get("vlm_image_enabled", True))
    if model.grounding_decoder_mode != "medsam":
        model.semantic_spatial_aligner = GatedSemanticSpatialAligner(
            language_dim=get_hidden_size(model.config),
            prompt_dim=256,
            num_heads=int(meta.get("aligner_heads", 8)),
            spatial_merge_factor=int(meta.get("spatial_merge_factor", 2)),
            spatial_topk=int(meta.get("spatial_topk", 64)),
            swiglu_expansion=int(meta.get("swiglu_expansion", 4)),
            dropout=float(meta.get("aligner_dropout", 0.0)),
        ).to(device)
        model.semantic_spatial_aligner.load_state_dict(
            torch.load(adapter / "semantic_spatial_aligner.pt", map_location=device)
        )
    if model.grounding_decoder_mode == "direct-mask":
        model.direct_mask_head = DirectPixelMaskHead(dim=256, output_size=256).to(device)
        model.direct_mask_head.load_state_dict(
            torch.load(adapter / "direct_mask_head.pt", map_location=device)
        )
    load_phrase_projector_state(
        model.phrase_projector,
        torch.load(adapter / "phrase_projector.pt", map_location=device),
    )
    model.localization_mask_decoder.load_state_dict(
        torch.load(adapter / "mask_decoder.pt", map_location=device)
    )
    if model.use_predicted_bbox:
        model.phrase_bbox_head.load_state_dict(
            torch.load(adapter / "phrase_bbox_head.pt", map_location=device)
        )
    if (adapter / "image_encoder.pt").exists():
        model.localization_image_encoder.load_state_dict(
            torch.load(adapter / "image_encoder.pt", map_location=device)
        )
    model.eval()
    return model, processor, open_token_ids, close_token_ids, device


def extract_hidden(
    model, processor, open_token_ids, close_token_ids, record, image, prompt_mode
):
    replay = processor.apply_chat_template(
        replay_messages(
            record,
            image,
            prompt_mode,
            use_vlm_image=getattr(model, "vlm_image_enabled", True),
        ),
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    replay = {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in replay.items()
    }
    with torch.no_grad():
        outputs = model(**replay, output_hidden_states=True)
        if model.grounding_decoder_mode == "medsam":
            return select_phrase_hidden(
                outputs.hidden_states[-1], replay["input_ids"], close_token_ids
            )
        phrase_hidden, _ = select_phrase_token_hiddens(
            outputs.hidden_states[-1],
            replay["input_ids"],
            open_token_ids,
            close_token_ids,
        )
        return phrase_hidden


def prepare_sam_context(model, sam_pixels):
    device = next(model.phrase_projector.parameters()).device
    sam_pixels = sam_pixels.to(device=device, dtype=torch.float32)
    model.localization_image_encoder.eval()
    model.localization_prompt_encoder.eval()
    with torch.no_grad():
        image_embeddings = model.localization_image_encoder(sam_pixels)
        if getattr(model, "sam_image_features_enabled", True) is False:
            image_embeddings = torch.zeros_like(image_embeddings)
        _, dense_embeddings = model.localization_prompt_encoder(
            points=None, boxes=None, masks=None
        )
        image_pe = model.localization_prompt_encoder.get_dense_pe()
    if dense_embeddings.shape[0] != image_embeddings.shape[0]:
        dense_embeddings = dense_embeddings.expand(image_embeddings.shape[0], -1, -1, -1)
    if image_pe.shape[0] != image_embeddings.shape[0]:
        image_pe = image_pe.expand(image_embeddings.shape[0], -1, -1, -1)
    return image_embeddings, image_pe, dense_embeddings, sam_pixels.shape[-2:]


def decode_hidden(model, hidden, sam_context, height, width, threshold):
    image_embeddings, image_pe, dense_embeddings, sam_size = sam_context
    device = next(model.phrase_projector.parameters()).device
    with torch.no_grad():
        if model.grounding_decoder_mode == "medsam":
            phrase_prompt = model.phrase_projector(hidden.to(device))
        else:
            phrase_prompt = model.semantic_spatial_aligner(
                hidden.to(device), image_embeddings
            )
        if model.grounding_decoder_mode == "direct-mask":
            logits, _ = model.direct_mask_head(phrase_prompt, image_embeddings)
            full_logits = resize_logits(logits, height, width)
            return (
                full_logits.sigmoid()[0, 0].detach().cpu().numpy() >= threshold
            ).astype(np.uint8)
        if model.use_predicted_bbox:
            predicted_boxes = model.phrase_bbox_head(
                pooled_phrase_hidden(hidden.to(device))
            )
            scale = torch.tensor(
                [sam_size[1], sam_size[0]] * 2,
                device=device,
                dtype=predicted_boxes.dtype,
            )
            box_embeddings, dense_embeddings = model.localization_prompt_encoder(
                points=None, boxes=predicted_boxes * scale, masks=None
            )
            phrase_prompt = torch.cat((phrase_prompt, box_embeddings), dim=1)
        logits, _ = model.localization_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=phrase_prompt,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        full_logits = resize_logits(logits, height, width)
    return (
        full_logits.sigmoid()[0, 0].detach().cpu().numpy() >= threshold
    ).astype(np.uint8)


def choose_multiregion_records(rows, min_regions):
    by_image = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_image[row["image"]][norm_text(row["region"])].append(row)

    selected = {}
    for image_path, region_rows in by_image.items():
        if len(region_rows) < min_regions:
            continue
        # Prefer a direct organ question when duplicates exist, then shortest text.
        representatives = {}
        for region, candidates in region_rows.items():
            representatives[region] = min(
                candidates,
                key=lambda row: (
                    region not in norm_text(row["question"]),
                    len(row["question"]),
                ),
            )
        selected[image_path] = representatives
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Causal same-image multi-target visual-grounding test."
    )
    parser.add_argument("--model", default="/root/models")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--test-file", default="test.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min-regions", type=int, default=2)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument(
        "--image-substring",
        default=None,
        help="Optionally test only image paths containing this text, e.g. xmlab13.",
    )
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-mode",
        choices=["original-question", "phrase-only"],
        default="original-question",
        help=(
            "original-question tests the complete question-plus-phrase pathway; "
            "phrase-only keeps the user prompt fixed and isolates phrase semantics."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    rows = load_jsonl(data_root / args.test_file)
    groups = choose_multiregion_records(rows, args.min_regions)
    if args.image_substring:
        groups = {
            path: records
            for path, records in groups.items()
            if args.image_substring.lower() in path.lower()
        }
    if args.limit_images is not None:
        groups = dict(list(sorted(groups.items()))[: args.limit_images])
    if not groups:
        raise ValueError("No multi-region images were found in the selected split.")

    model, processor, open_token_ids, close_token_ids, device = load_model(args)
    image_results = []
    pair_results = []
    ablation_correct = []
    ablation_zero = []
    ablation_random = []
    all_off_diagonal = []

    for image_path, records in tqdm(groups.items(), desc="Grounding test"):
        image = Image.open(data_root / image_path).convert("RGB")
        width, height = image.size
        sam_pixels = medsam_image_tensor(image).unsqueeze(0)
        sam_context = prepare_sam_context(model, sam_pixels)
        regions = sorted(records)
        hiddens = {}
        golds = {}
        predictions = {}

        for region in regions:
            record = records[region]
            hiddens[region] = extract_hidden(
                model,
                processor,
                open_token_ids,
                close_token_ids,
                record,
                image,
                args.prompt_mode,
            )
            golds[region] = (
                np.asarray(Image.open(data_root / record["mask"]).convert("L")) > 0
            )
            predictions[region] = decode_hidden(
                model,
                hiddens[region],
                sam_context,
                height,
                width,
                args.threshold,
            )

        matrix = {
            pred_region: {
                gold_region: binary_iou(predictions[pred_region], golds[gold_region])
                for gold_region in regions
            }
            for pred_region in regions
        }

        for region in regions:
            correct_iou = matrix[region][region]
            ablation_correct.append(correct_iou)
            zero_prediction = decode_hidden(
                model,
                torch.zeros_like(hiddens[region]),
                sam_context,
                height,
                width,
                args.threshold,
            )
            zero_iou = binary_iou(zero_prediction, golds[region])
            ablation_zero.append(zero_iou)

            hidden = hiddens[region]
            random_ious = []
            hidden_mean = hidden.mean()
            hidden_std = hidden.std().clamp_min(1e-6)
            for _ in range(args.random_repeats):
                random_hidden = torch.randn_like(hidden) * hidden_std + hidden_mean
                random_prediction = decode_hidden(
                    model,
                    random_hidden,
                    sam_context,
                    height,
                    width,
                    args.threshold,
                )
                random_ious.append(binary_iou(random_prediction, golds[region]))
            ablation_random.append(float(np.mean(random_ious)))

            safe_image = Path(image_path).stem
            save_overlay(
                image,
                golds[region],
                predictions[region],
                overlay_dir / f"{safe_image}_{region.replace(' ', '_')}.jpg",
            )

        pair_gaps = []
        pair_passes = []
        for region_a, region_b in itertools.combinations(regions, 2):
            i_aa = matrix[region_a][region_a]
            i_ab = matrix[region_a][region_b]
            i_ba = matrix[region_b][region_a]
            i_bb = matrix[region_b][region_b]
            gap = ((i_aa - i_ab) + (i_bb - i_ba)) / 2.0
            passed = i_aa > i_ab and i_bb > i_ba
            pair_gaps.append(gap)
            pair_passes.append(passed)
            all_off_diagonal.extend([i_ab, i_ba])
            pair_results.append(
                {
                    "image": image_path,
                    "region_a": region_a,
                    "region_b": region_b,
                    "I_AA": i_aa,
                    "I_AB": i_ab,
                    "I_BA": i_ba,
                    "I_BB": i_bb,
                    "grounding_gap": gap,
                    "pairwise_grounding_success": passed,
                }
            )

        image_results.append(
            {
                "image": image_path,
                "regions": regions,
                "iou_matrix": matrix,
                "mean_grounding_gap": float(np.mean(pair_gaps)),
                "pairwise_success_rate": float(np.mean(pair_passes)),
            }
        )

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "test_file": str(data_root / args.test_file),
        "threshold": args.threshold,
        "prompt_mode": args.prompt_mode,
        "num_multi_region_images": len(image_results),
        "num_region_pairs": len(pair_results),
        "mean_correct_iou": mean(ablation_correct),
        "mean_swapped_iou": mean(all_off_diagonal),
        "mean_zero_hidden_iou": mean(ablation_zero),
        "mean_random_hidden_iou": mean(ablation_random),
        "mean_grounding_gap": mean([row["grounding_gap"] for row in pair_results]),
        "pairwise_grounding_accuracy": mean(
            [row["pairwise_grounding_success"] for row in pair_results]
        ),
        "interpretation": {
            "positive_gap": "Correct phrase masks overlap their own targets more than other targets.",
            "strong_evidence": "Correct IoU should exceed swapped, zero-hidden, and random-hidden IoU.",
            "scope": "Teacher-forced phrase-to-mask grounding; free-form phrase generation is not evaluated.",
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "images.json").write_text(
        json.dumps(image_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "pairs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_results[0].keys()))
        writer.writeheader()
        writer.writerows(pair_results)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Detailed results: {output_dir}")


if __name__ == "__main__":
    main()
