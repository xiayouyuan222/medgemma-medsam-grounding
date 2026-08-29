import argparse
import json
from collections import defaultdict
from pathlib import Path

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
    binary_iou_and_dice,
    box_iou_xyxy,
    decode_phrase_masks,
    get_hidden_size,
    get_tokenizer,
    load_binary_mask,
    load_medsam,
    load_phrase_projector_state,
    masks_to_normalized_boxes,
    medsam_image_tensor,
    select_phrase_hidden,
    select_phrase_token_hiddens,
)


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def replay_messages(record, image, prompt_mode, use_vlm_image=True):
    if prompt_mode == "phrase-only":
        user_text = (
            "Identify and ground the referenced medical region at pixel level.\n"
            "Use the exact text format:\n"
            f"Relevant region: {OPEN_TAG}region{CLOSE_TAG}\n"
            "Question: Locate the referenced medical region."
        )
    else:
        user_text = (
            f'Ground the medical phrase "{record["region"]}" in this image. '
            f"Return it as {OPEN_TAG}phrase{CLOSE_TAG}."
        )
    user_content = [{"type": "text", "text": user_text}]
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
            "content": [
                {
                    "type": "text",
                    "text": f"Relevant region: {OPEN_TAG}{record['region']}{CLOSE_TAG}",
                }
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--test-file", default="validation.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument(
        "--prompt-mode",
        choices=["gold-query", "phrase-only"],
        default="gold-query",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    adapter = Path(args.adapter)
    data_root = Path(args.data_root)
    rows = load_jsonl(data_root / args.test_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    meta = json.loads(
        (adapter / "vividmed_lite_seg_meta.json").read_text(encoding="utf-8")
    )
    use_vlm_image = bool(meta.get("vlm_image_enabled", True))

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
    sam = load_medsam(
        args.sam_checkpoint, meta.get("sam_model_type", "vit_b")
    ).to(device)
    num_prompt_tokens = int(meta.get("num_prompt_tokens", 1))
    model.phrase_projector = PhraseProjector(
        get_hidden_size(model.config), 256, num_prompt_tokens
    ).to(device)
    use_predicted_bbox = bool(meta.get("use_predicted_bbox", False))
    if use_predicted_bbox:
        model.phrase_bbox_head = PhraseBBoxHead(get_hidden_size(model.config)).to(device)
    model.localization_image_encoder = sam.image_encoder
    model.localization_prompt_encoder = sam.prompt_encoder
    model.localization_mask_decoder = sam.mask_decoder
    model.sam_image_features_enabled = bool(
        meta.get("sam_image_features_enabled", True)
    )
    grounding_mode = meta.get("grounding_decoder_mode", "medsam")
    model.grounding_decoder_mode = grounding_mode
    if grounding_mode != "medsam":
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
    if grounding_mode == "direct-mask":
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
    if use_predicted_bbox:
        model.phrase_bbox_head.load_state_dict(
            torch.load(adapter / "phrase_bbox_head.pt", map_location=device)
        )
    if (adapter / "image_encoder.pt").exists():
        model.localization_image_encoder.load_state_dict(
            torch.load(adapter / "image_encoder.pt", map_location=device)
        )
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    totals = {"iou": 0.0, "dice": 0.0, "iou_at_0_5": 0, "bbox_iou": 0.0}
    per_region = defaultdict(list)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(rows, desc="Gold-phrase evaluation"):
            image = Image.open(data_root / record["image"]).convert("RGB")
            replay = processor.apply_chat_template(
                replay_messages(
                    record, image, args.prompt_mode, use_vlm_image=use_vlm_image
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
                phrase_mask = None
                if grounding_mode == "medsam":
                    phrase_hidden = select_phrase_hidden(
                        outputs.hidden_states[-1], replay["input_ids"], close_token_ids
                    )
                else:
                    phrase_hidden, phrase_mask = select_phrase_token_hiddens(
                        outputs.hidden_states[-1],
                        replay["input_ids"],
                        open_token_ids,
                        close_token_ids,
                    )
                logits, _, predicted_boxes = decode_phrase_masks(
                    model,
                    phrase_hidden,
                    medsam_image_tensor(image).unsqueeze(0),
                    freeze_image_encoder=meta.get("sam_image_encoder_frozen", True),
                    use_predicted_bbox=use_predicted_bbox,
                    phrase_mask=phrase_mask,
                )
            target = load_binary_mask(data_root / record["mask"]).unsqueeze(0).to(device)
            target_boxes = masks_to_normalized_boxes(target)
            bbox_iou = (
                float(box_iou_xyxy(predicted_boxes, target_boxes).item())
                if predicted_boxes is not None
                else None
            )
            iou_tensor, dice_tensor = binary_iou_and_dice(
                logits, target, args.threshold
            )
            iou = float(iou_tensor.item())
            dice = float(dice_tensor.item())
            totals["iou"] += iou
            totals["dice"] += dice
            totals["iou_at_0_5"] += int(iou >= 0.5)
            if bbox_iou is not None:
                totals["bbox_iou"] += bbox_iou
            per_region[record["region"]].append((iou, dice))
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "image": record["image"],
                        "region": record["region"],
                        "mask_iou": iou,
                        "mask_dice": dice,
                        "predicted_bbox_norm_xyxy": (
                            predicted_boxes[0].detach().cpu().tolist()
                            if predicted_boxes is not None
                            else None
                        ),
                        "gold_bbox_norm_xyxy": target_boxes[0].detach().cpu().tolist(),
                        "bbox_iou": bbox_iou,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n = max(len(rows), 1)
    summary = {
        "model": args.model,
        "adapter": str(adapter),
        "test_file": str(data_root / args.test_file),
        "evaluation_mode": f"teacher-forced-{args.prompt_mode}",
        "vlm_image_enabled": use_vlm_image,
        "grounding_decoder_mode": grounding_mode,
        "num_samples": len(rows),
        "mean_mask_iou": totals["iou"] / n,
        "mean_dice": totals["dice"] / n,
        "mask_iou_at_0_5": totals["iou_at_0_5"] / n,
        "mean_bbox_iou": totals["bbox_iou"] / n if use_predicted_bbox else None,
        "threshold": args.threshold,
        "per_region": {
            region: {
                "n": len(values),
                "mean_iou": sum(value[0] for value in values) / len(values),
                "mean_dice": sum(value[1] for value in values) / len(values),
            }
            for region, values in sorted(per_region.items())
        },
        "output": str(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
