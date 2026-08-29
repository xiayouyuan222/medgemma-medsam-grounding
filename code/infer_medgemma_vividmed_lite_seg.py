import argparse
import json
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
    DirectPixelMaskHead,
    GatedSemanticSpatialAligner,
    build_open_tag_token_variants,
    PhraseBBoxHead,
    PhraseProjector,
    binary_iou_and_dice,
    decode_phrase_masks,
    extract_answer,
    extract_region,
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
)


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prompt_messages(rec, image):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": rec["messages"][0]["content"][1]["text"]},
                {"type": "image", "image": image},
            ],
        },
    ]


def save_overlay(image, gold, prediction, path):
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    result = base.copy()
    gold = gold.astype(bool)
    prediction = prediction.astype(bool)
    # Gold is green, prediction is red, overlap is yellow.
    result[gold] = 0.55 * result[gold] + 0.45 * np.array([0, 255, 0])
    result[prediction] = 0.55 * result[prediction] + 0.45 * np.array([255, 0, 0])
    overlap = gold & prediction
    result[overlap] = 0.45 * base[overlap] + 0.55 * np.array([255, 255, 0])
    Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--test-file", default="test.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    adapter = Path(args.adapter)
    data_root = Path(args.data_root)
    rows = load_jsonl(data_root / args.test_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    meta = json.loads((adapter / "vividmed_lite_seg_meta.json").read_text(encoding="utf-8"))

    processor = AutoProcessor.from_pretrained(adapter)
    tokenizer = get_tokenizer(processor)
    close_token_ids = meta.get("close_token_ids")
    if not close_token_ids:
        close_token_ids = tokenizer.encode(CLOSE_TAG, add_special_tokens=False)
    open_token_ids = meta.get("open_token_ids")
    if not open_token_ids:
        open_token_ids = build_open_tag_token_variants(tokenizer)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = load_medsam(args.sam_checkpoint, meta.get("sam_model_type", "vit_b")).to(device)
    num_prompt_tokens = int(meta.get("num_prompt_tokens", 1))
    grounding_decoder_mode = meta.get("grounding_decoder_mode", "medsam")
    model.phrase_projector = PhraseProjector(
        get_hidden_size(model.config), 256, num_prompt_tokens
    ).to(device)
    use_predicted_bbox = bool(meta.get("use_predicted_bbox", False))
    if use_predicted_bbox:
        model.phrase_bbox_head = PhraseBBoxHead(get_hidden_size(model.config)).to(device)
    model.localization_image_encoder = sam.image_encoder
    model.localization_prompt_encoder = sam.prompt_encoder
    model.localization_mask_decoder = sam.mask_decoder
    load_phrase_projector_state(
        model.phrase_projector,
        torch.load(adapter / "phrase_projector.pt", map_location=device),
    )
    model.localization_mask_decoder.load_state_dict(torch.load(adapter / "mask_decoder.pt", map_location=device))
    if grounding_decoder_mode != "medsam":
        model.semantic_spatial_aligner = GatedSemanticSpatialAligner(
            language_dim=get_hidden_size(model.config),
            prompt_dim=256,
            num_heads=int(meta.get("aligner_heads", 8)),
            merge_factor=int(meta.get("spatial_merge_factor", 2)),
            topk=int(meta.get("spatial_topk", 64)),
            swiglu_expansion=float(meta.get("swiglu_expansion", 2.0)),
            dropout=float(meta.get("aligner_dropout", 0.0)),
        ).to(device)
        model.semantic_spatial_aligner.load_state_dict(
            torch.load(adapter / "semantic_spatial_aligner.pt", map_location=device)
        )
    if grounding_decoder_mode == "direct-mask":
        model.direct_mask_head = DirectPixelMaskHead(prompt_dim=256).to(device)
        model.direct_mask_head.load_state_dict(
            torch.load(adapter / "direct_mask_head.pt", map_location=device)
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
    mask_dir = Path(args.mask_output_dir) if args.mask_output_dir else output_path.with_suffix("").parent / (output_path.stem + "_masks")
    pred_dir = mask_dir / "pred"
    overlay_dir = mask_dir / "overlay"
    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "samples": 0,
        "mask_predictions": 0,
        "iou": 0.0,
        "dice": 0.0,
        "iou_at_0_5": 0,
        "answer_correct": 0,
        "region_correct": 0,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        for rec in tqdm(rows, desc="Infer"):
            image = Image.open(data_root / rec["image"]).convert("RGB")
            width, height = image.size
            messages = prompt_messages(rec, image)
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(model.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            input_length = inputs["input_ids"].shape[-1]
            pred_text = processor.decode(
                generated[0][input_length:], skip_special_tokens=False
            ).strip()
            pred_region = extract_region(pred_text)
            pred_answer = extract_answer(pred_text)

            pred_mask = np.zeros((height, width), dtype=np.uint8)
            iou = 0.0
            dice = 0.0
            # Canonical replay makes localization independent of minor format
            # mistakes while preserving the model-predicted phrase semantics.
            # The base tokenizer's existing tokens encode <p> and </p>.
            has_region = bool(pred_region)
            if has_region:
                canonical_text = (
                    f"Relevant region: <p>{pred_region}</p>\n"
                    f"Answer: {pred_answer}"
                )
                full_messages = messages + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": canonical_text}],
                    }
                ]
                replay = processor.apply_chat_template(
                    full_messages,
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
                    if grounding_decoder_mode == "medsam":
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
                    low_res_logits, _, _ = decode_phrase_masks(
                        model,
                        phrase_hidden,
                        medsam_image_tensor(image).unsqueeze(0),
                        freeze_image_encoder=True,
                        use_predicted_bbox=use_predicted_bbox,
                        phrase_mask=phrase_mask,
                    )
                    full_logits = resize_logits(low_res_logits, height, width)
                gold_full = load_binary_mask(data_root / rec["mask"], size=256).unsqueeze(0).to(device)
                iou_tensor, dice_tensor = binary_iou_and_dice(
                    low_res_logits, gold_full, args.threshold
                )
                iou = float(iou_tensor.item())
                dice = float(dice_tensor.item())
                pred_mask = (full_logits.sigmoid()[0, 0].cpu().numpy() >= args.threshold).astype(np.uint8)
                totals["mask_predictions"] += 1

            gold_mask = np.asarray(Image.open(data_root / rec["mask"]).convert("L")) > 0
            pred_path = pred_dir / f"{rec['id']}_pred.png"
            overlay_path = overlay_dir / f"{rec['id']}_overlay.jpg"
            Image.fromarray(pred_mask * 255).save(pred_path)
            save_overlay(image, gold_mask, pred_mask, overlay_path)

            answer_correct = pred_answer == norm_text(rec["answer"])
            region_correct = pred_region == norm_text(rec["region"])
            totals["samples"] += 1
            totals["iou"] += iou
            totals["dice"] += dice
            totals["iou_at_0_5"] += int(iou >= 0.5)
            totals["answer_correct"] += int(answer_correct)
            totals["region_correct"] += int(region_correct)
            result = dict(rec)
            result.update(
                {
                    "prediction": pred_text,
                    "pred_region": pred_region,
                    "pred_answer": pred_answer,
                    "answer_correct": answer_correct,
                    "region_correct": region_correct,
                    "mask_predicted": has_region,
                    "mask_iou": iou,
                    "mask_dice": dice,
                    "pred_mask": str(pred_path),
                    "overlay": str(overlay_path),
                }
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    n = max(totals["samples"], 1)
    summary = {
        "model": args.model,
        "adapter": str(adapter),
        "test_file": str(data_root / args.test_file),
        "num_samples": totals["samples"],
        "answer_accuracy": totals["answer_correct"] / n,
        "region_accuracy": totals["region_correct"] / n,
        "mask_prediction_rate": totals["mask_predictions"] / n,
        "mean_mask_iou": totals["iou"] / n,
        "mean_dice": totals["dice"] / n,
        "mask_iou_at_0_5": totals["iou_at_0_5"] / n,
        "threshold": args.threshold,
        "output": str(output_path),
        "mask_output_dir": str(mask_dir),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
