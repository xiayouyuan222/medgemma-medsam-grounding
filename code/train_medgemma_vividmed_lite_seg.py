import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from vividmed_lite_seg_common import (
    CLOSE_TAG,
    OPEN_TAG,
    SYSTEM_PROMPT,
    PhraseBBoxHead,
    PhraseProjector,
    DirectPixelMaskHead,
    GatedSemanticSpatialAligner,
    build_open_tag_token_variants,
    decode_phrase_masks,
    dice_loss_with_logits,
    generalized_box_iou_loss,
    get_hidden_size,
    get_tokenizer,
    load_medsam,
    load_phrase_projector_state,
    masks_to_normalized_boxes,
    medsam_image_tensor,
    select_phrase_hidden,
    select_phrase_token_hiddens,
)


def edge_dice_loss_with_logits(logits, targets, kernel_size=3, eps=1e-6):
    """Soft morphological boundary Dice loss for binary masks."""
    padding = kernel_size // 2
    probabilities = logits.sigmoid()

    def boundary_map(values):
        dilated = F.max_pool2d(
            values, kernel_size=kernel_size, stride=1, padding=padding
        )
        eroded = -F.max_pool2d(
            -values, kernel_size=kernel_size, stride=1, padding=padding
        )
        return (dilated - eroded).clamp(0.0, 1.0)

    predicted_edges = boundary_map(probabilities)
    target_edges = boundary_map(targets)
    dims = tuple(range(1, predicted_edges.ndim))
    intersection = (predicted_edges * target_edges).sum(dim=dims)
    denominator = predicted_edges.sum(dim=dims) + target_edges.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


class SlakeSegDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        data_root,
        processor,
        include_switch_target=False,
        training_mode="vqa",
        use_vlm_image=True,
        region_loss_weights=None,
        augment_tail=False,
        tail_max_count=3,
        augmentation_probability=0.8,
        augmentation_rotation_degrees=10.0,
        augmentation_translation_fraction=0.05,
        augmentation_intensity_fraction=0.1,
    ):
        self.data_root = Path(data_root)
        self.processor = processor
        self.training_mode = training_mode
        self.use_vlm_image = bool(use_vlm_image)
        self.region_loss_weights = region_loss_weights or {}
        self.rows = [
            json.loads(line)
            for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.region_counts = Counter(row["region"] for row in self.rows)
        self.augment_tail = bool(augment_tail)
        self.tail_regions = {
            region
            for region, count in self.region_counts.items()
            if count <= int(tail_max_count)
        }
        self.augmentation_probability = float(augmentation_probability)
        self.augmentation_rotation_degrees = float(augmentation_rotation_degrees)
        self.augmentation_translation_fraction = float(
            augmentation_translation_fraction
        )
        self.augmentation_intensity_fraction = float(
            augmentation_intensity_fraction
        )
        self.switch_target_by_index = {}
        if include_switch_target:
            indices_by_image = {}
            for index, row in enumerate(self.rows):
                indices_by_image.setdefault(row["image"], []).append(index)
            for indices in indices_by_image.values():
                for index in indices:
                    region = self.rows[index]["region"].strip().lower()
                    alternatives = [
                        other
                        for other in indices
                        if self.rows[other]["region"].strip().lower() != region
                    ]
                    if alternatives:
                        # A deterministic target keeps experiments reproducible.
                        self.switch_target_by_index[index] = alternatives[index % len(alternatives)]

    @staticmethod
    def _translate(image, x_offset, y_offset, resample):
        return image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, -x_offset, 0, 1, -y_offset),
            resample=resample,
            fillcolor=0,
        )

    def _augment_pair(self, image, masks):
        angle = random.uniform(
            -self.augmentation_rotation_degrees,
            self.augmentation_rotation_degrees,
        )
        max_x = self.augmentation_translation_fraction * image.width
        max_y = self.augmentation_translation_fraction * image.height
        x_offset = random.uniform(-max_x, max_x)
        y_offset = random.uniform(-max_y, max_y)

        image = image.rotate(
            angle, resample=Image.Resampling.BILINEAR, fillcolor=0
        )
        image = self._translate(
            image, x_offset, y_offset, Image.Resampling.BILINEAR
        )
        transformed_masks = []
        for mask in masks:
            mask = mask.rotate(
                angle, resample=Image.Resampling.NEAREST, fillcolor=0
            )
            transformed_masks.append(
                self._translate(
                    mask, x_offset, y_offset, Image.Resampling.NEAREST
                )
            )

        jitter = self.augmentation_intensity_fraction
        if jitter > 0:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(1.0 - jitter, 1.0 + jitter)
            )
            image = ImageEnhance.Contrast(image).enhance(
                random.uniform(1.0 - jitter, 1.0 + jitter)
            )
        return image, transformed_masks

    @staticmethod
    def _mask_tensor(mask, size=256):
        mask = mask.resize((size, size), Image.Resampling.NEAREST)
        array = (np.asarray(mask) > 0).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0)

    def __len__(self):
        return len(self.rows)

    def messages(self, rec, image, include_answer):
        if self.training_mode == "phrase-only":
            user_text = (
                "Identify and ground the referenced medical region at pixel level.\n"
                "Use the exact text format:\n"
                f"Relevant region: {OPEN_TAG}region{CLOSE_TAG}\n"
                "Question: Locate the referenced medical region."
            )
            assistant_text = f"Relevant region: {OPEN_TAG}{rec['region']}{CLOSE_TAG}"
        elif self.training_mode == "gold-phrase":
            user_text = (
                f'Ground the medical phrase "{rec["region"]}" in this image. '
                f"Return it as {OPEN_TAG}phrase{CLOSE_TAG}."
            )
            assistant_text = f"Relevant region: {OPEN_TAG}{rec['region']}{CLOSE_TAG}"
        else:
            user_text = rec["messages"][0]["content"][1]["text"]
            assistant_text = (
                f"Relevant region: {OPEN_TAG}{rec['region']}{CLOSE_TAG}\n"
                f"Answer: {rec['answer']}"
            )
        user_content = [{"type": "text", "text": user_text}]
        if self.use_vlm_image:
            user_content.append({"type": "image", "image": image})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": user_content,
            },
        ]
        if include_answer:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": assistant_text,
                        }
                    ],
                }
            )
        return messages

    def __getitem__(self, index):
        rec = self.rows[index]
        image = Image.open(self.data_root / rec["image"]).convert("RGB")
        target_mask_image = Image.open(self.data_root / rec["mask"]).convert("L")
        switch_index = self.switch_target_by_index.get(index)
        switch_mask_image = None
        if switch_index is not None:
            switch_rec = self.rows[switch_index]
            switch_mask_image = Image.open(
                self.data_root / switch_rec["mask"]
            ).convert("L")

        mask_images = [target_mask_image]
        if switch_mask_image is not None:
            mask_images.append(switch_mask_image)
        if (
            self.augment_tail
            and rec["region"] in self.tail_regions
            and random.random() < self.augmentation_probability
        ):
            image, mask_images = self._augment_pair(image, mask_images)
        target_mask_image = mask_images[0]
        if switch_mask_image is not None:
            switch_mask_image = mask_images[1]

        full = self.processor.apply_chat_template(
            self.messages(rec, image, True),
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt = self.processor.apply_chat_template(
            self.messages(rec, image, False),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        item = {}
        for key, value in full.items():
            item[key] = value.squeeze(0) if torch.is_tensor(value) and value.shape[0] == 1 else value
        labels = item["input_ids"].clone()
        labels[: prompt["input_ids"].shape[-1]] = -100
        item["labels"] = labels
        item["sam_pixel_values"] = medsam_image_tensor(image)
        item["target_mask"] = self._mask_tensor(target_mask_image)
        item["class_loss_weight"] = torch.tensor(
            self.region_loss_weights.get(rec["region"], 1.0), dtype=torch.float32
        )
        if switch_index is not None:
            item["switch_target_mask"] = self._mask_tensor(switch_mask_image)
            item["has_switch_target"] = torch.tensor(1.0)
        else:
            item["switch_target_mask"] = torch.zeros_like(item["target_mask"])
            item["has_switch_target"] = torch.tensor(0.0)
        return item


class SingleBatchCollator:
    def __call__(self, features):
        if len(features) != 1:
            raise ValueError("This script requires per_device_train_batch_size=1.")
        return {
            key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in features[0].items()
        }


class GroundingModulesCheckpointCallback(TrainerCallback):
    """Persist trainable modules that PEFT checkpoints do not include."""

    def __init__(self, save_image_encoder=False):
        self.save_image_encoder = save_image_encoder

    def on_save(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            model.phrase_projector.state_dict(),
            checkpoint_dir / "phrase_projector.pt",
        )
        torch.save(
            model.localization_mask_decoder.state_dict(),
            checkpoint_dir / "mask_decoder.pt",
        )
        if hasattr(model, "semantic_spatial_aligner"):
            torch.save(
                model.semantic_spatial_aligner.state_dict(),
                checkpoint_dir / "semantic_spatial_aligner.pt",
            )
        if hasattr(model, "direct_mask_head"):
            torch.save(
                model.direct_mask_head.state_dict(),
                checkpoint_dir / "direct_mask_head.pt",
            )
        if hasattr(model, "phrase_bbox_head"):
            torch.save(
                model.phrase_bbox_head.state_dict(),
                checkpoint_dir / "phrase_bbox_head.pt",
            )
        if self.save_image_encoder:
            torch.save(
                model.localization_image_encoder.state_dict(),
                checkpoint_dir / "image_encoder.pt",
            )
        return control


def restore_grounding_modules(model, checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    load_phrase_projector_state(
        model.phrase_projector,
        torch.load(checkpoint_dir / "phrase_projector.pt", map_location=device),
    )
    model.localization_mask_decoder.load_state_dict(
        torch.load(checkpoint_dir / "mask_decoder.pt", map_location=device)
    )
    aligner_path = checkpoint_dir / "semantic_spatial_aligner.pt"
    if hasattr(model, "semantic_spatial_aligner") and aligner_path.exists():
        model.semantic_spatial_aligner.load_state_dict(
            torch.load(aligner_path, map_location=device)
        )
    direct_head_path = checkpoint_dir / "direct_mask_head.pt"
    if hasattr(model, "direct_mask_head") and direct_head_path.exists():
        model.direct_mask_head.load_state_dict(
            torch.load(direct_head_path, map_location=device)
        )
    bbox_path = checkpoint_dir / "phrase_bbox_head.pt"
    if hasattr(model, "phrase_bbox_head") and bbox_path.exists():
        model.phrase_bbox_head.load_state_dict(
            torch.load(bbox_path, map_location=device)
        )
    image_encoder_path = checkpoint_dir / "image_encoder.pt"
    if image_encoder_path.exists():
        model.localization_image_encoder.load_state_dict(
            torch.load(image_encoder_path, map_location=device)
        )


class SegTrainer(Trainer):
    def __init__(
        self,
        *args,
        close_token_ids,
        open_token_ids,
        text_loss_weight=1.0,
        dice_loss_weight=1.0,
        bce_loss_weight=1.0,
        focal_loss_weight=0.0,
        focal_alpha=0.75,
        focal_gamma=2.0,
        edge_loss_weight=0.0,
        edge_kernel_size=3,
        freeze_image_encoder=True,
        train_sample_weights=None,
        switch_loss_weight=0.0,
        switch_margin=0.2,
        bbox_l1_loss_weight=0.0,
        bbox_giou_loss_weight=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.close_token_ids = [int(token_id) for token_id in close_token_ids]
        self.open_token_ids = [
            [int(token_id) for token_id in pattern]
            for pattern in open_token_ids
        ]
        self.text_loss_weight = float(text_loss_weight)
        self.dice_loss_weight = float(dice_loss_weight)
        self.bce_loss_weight = float(bce_loss_weight)
        self.focal_loss_weight = float(focal_loss_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.edge_loss_weight = float(edge_loss_weight)
        self.edge_kernel_size = int(edge_kernel_size)
        self.freeze_image_encoder = bool(freeze_image_encoder)
        self.train_sample_weights = train_sample_weights
        self.switch_loss_weight = float(switch_loss_weight)
        self.switch_margin = float(switch_margin)
        self.bbox_l1_loss_weight = float(bbox_l1_loss_weight)
        self.bbox_giou_loss_weight = float(bbox_giou_loss_weight)
        self._gradient_reported = False

    def _get_train_sampler(self, train_dataset=None):
        if self.train_sample_weights is not None:
            return WeightedRandomSampler(
                weights=self.train_sample_weights,
                num_samples=len(self.train_sample_weights),
                replacement=True,
            )
        return super()._get_train_sampler(train_dataset)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if not self._gradient_reported:
            groups = {
                "LoRA": [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad],
                "Projector": [p for n, p in model.named_parameters() if "phrase_projector" in n and p.requires_grad],
                "MaskDecoder": [p for n, p in model.named_parameters() if "localization_mask_decoder" in n and p.requires_grad],
                "Aligner": [p for n, p in model.named_parameters() if "semantic_spatial_aligner" in n and p.requires_grad],
                "DirectMaskHead": [p for n, p in model.named_parameters() if "direct_mask_head" in n and p.requires_grad],
                "BBoxHead": [p for n, p in model.named_parameters() if "phrase_bbox_head" in n and p.requires_grad],
            }
            report = [f"{name} {sum(p.grad is not None for p in params)}/{len(params)}" for name, params in groups.items()]
            print("Gradient check: " + "; ".join(report))
            self._gradient_reported = True
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        grounding_model = model.module if hasattr(model, "module") else model
        sam_pixels = inputs.pop("sam_pixel_values")
        class_loss_weight = inputs.pop("class_loss_weight", None)
        switch_target_mask = inputs.pop("switch_target_mask", None)
        has_switch_target = inputs.pop("has_switch_target", None)
        target_mask = inputs.pop("target_mask").to(
            device=next(grounding_model.phrase_projector.parameters()).device,
            dtype=torch.float32,
        )
        outputs = model(**inputs, output_hidden_states=True)
        grounding_mode = getattr(grounding_model, "grounding_decoder_mode", "medsam")
        phrase_mask = None
        if grounding_mode == "medsam":
            phrase_hidden = select_phrase_hidden(
                outputs.hidden_states[-1],
                inputs["input_ids"],
                self.close_token_ids,
                labels=inputs["labels"],
            )
        else:
            phrase_hidden, phrase_mask = select_phrase_token_hiddens(
                outputs.hidden_states[-1],
                inputs["input_ids"],
                self.open_token_ids,
                self.close_token_ids,
                labels=inputs["labels"],
            )
        mask_logits, quality, predicted_boxes = decode_phrase_masks(
            grounding_model,
            phrase_hidden,
            sam_pixels,
            self.freeze_image_encoder,
            use_predicted_bbox=hasattr(grounding_model, "phrase_bbox_head"),
            phrase_mask=phrase_mask,
        )
        bce_loss = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        element_bce = F.binary_cross_entropy_with_logits(
            mask_logits, target_mask, reduction="none"
        )
        probabilities = mask_logits.sigmoid()
        pt = probabilities * target_mask + (1.0 - probabilities) * (1.0 - target_mask)
        alpha_t = (
            self.focal_alpha * target_mask
            + (1.0 - self.focal_alpha) * (1.0 - target_mask)
        )
        focal_loss = (alpha_t * (1.0 - pt).pow(self.focal_gamma) * element_bce).mean()
        dice_loss = dice_loss_with_logits(mask_logits, target_mask)
        edge_loss = edge_dice_loss_with_logits(
            mask_logits, target_mask, kernel_size=self.edge_kernel_size
        )
        if class_loss_weight is None:
            class_loss_weight = mask_logits.new_ones(())
        else:
            class_loss_weight = class_loss_weight.to(
                device=mask_logits.device, dtype=torch.float32
            ).mean()
        weighted_seg_loss = class_loss_weight * (
            self.dice_loss_weight * dice_loss
            + self.bce_loss_weight * bce_loss
            + self.focal_loss_weight * focal_loss
            + self.edge_loss_weight * edge_loss
        )
        bbox_l1_loss = mask_logits.new_zeros(())
        bbox_giou_loss = mask_logits.new_zeros(())
        if predicted_boxes is not None:
            target_boxes = masks_to_normalized_boxes(target_mask)
            bbox_l1_loss = F.l1_loss(predicted_boxes, target_boxes)
            bbox_giou_loss = generalized_box_iou_loss(predicted_boxes, target_boxes)
        switch_loss = mask_logits.new_zeros(())
        if (
            self.switch_loss_weight > 0
            and switch_target_mask is not None
            and has_switch_target is not None
        ):
            switch_target_mask = switch_target_mask.to(
                device=mask_logits.device, dtype=torch.float32
            )
            has_switch_target = has_switch_target.to(mask_logits.device).reshape(-1)
            valid = has_switch_target > 0.5
            if valid.any():
                probabilities = mask_logits.sigmoid()
                dims = tuple(range(1, probabilities.ndim))

                def soft_dice(mask):
                    intersection = (probabilities * mask).sum(dim=dims)
                    denominator = probabilities.sum(dim=dims) + mask.sum(dim=dims)
                    return (2.0 * intersection + 1e-6) / (denominator + 1e-6)

                positive_score = soft_dice(target_mask)
                switched_score = soft_dice(switch_target_mask)
                # The current phrase should overlap its own mask more than a
                # different region from the same image by at least margin.
                switch_loss = F.relu(
                    self.switch_margin - (positive_score - switched_score)
                )[valid].mean()
        loss = (
            self.text_loss_weight * outputs.loss
            + weighted_seg_loss
            + self.switch_loss_weight * switch_loss
            + self.bbox_l1_loss_weight * bbox_l1_loss
            + self.bbox_giou_loss_weight * bbox_giou_loss
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Some PEFT/SAM branches are sample-dependent, and the grounding
            # heads run after the DDP-wrapped VLM forward. A zero-valued anchor
            # gives every trainable tensor a reduction hook without changing
            # the objective or its gradients.
            trainable_anchors = [
                parameter.reshape(-1)[0]
                for parameter in grounding_model.parameters()
                if parameter.requires_grad and parameter.numel() > 0
            ]
            if trainable_anchors:
                loss = loss + torch.stack(trainable_anchors).sum() * 0.0
        if return_outputs:
            outputs.mask_logits = mask_logits.detach()
            outputs.mask_quality = quality.detach()
            outputs.seg_bce_loss = bce_loss.detach()
            outputs.seg_dice_loss = dice_loss.detach()
            outputs.seg_focal_loss = focal_loss.detach()
            outputs.seg_edge_loss = edge_loss.detach()
            outputs.seg_switch_loss = switch_loss.detach()
            outputs.class_loss_weight = class_loss_weight.detach()
            outputs.bbox_l1_loss = bbox_l1_loss.detach()
            outputs.bbox_giou_loss = bbox_giou_loss.detach()
            return loss, outputs
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models")
    parser.add_argument(
        "--init-adapter",
        default=None,
        help="Continue from a Stage-1 adapter, projector and mask decoder.",
    )
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-file", default="train.jsonl")
    parser.add_argument("--val-file", default="validation.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--bce-loss-weight", type=float, default=1.0)
    parser.add_argument("--focal-loss-weight", type=float, default=0.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--edge-loss-weight",
        type=float,
        default=0.0,
        help="Weight for soft morphological boundary Dice loss.",
    )
    parser.add_argument(
        "--edge-kernel-size",
        type=int,
        default=3,
        help="Odd pooling kernel used to extract differentiable mask boundaries.",
    )
    parser.add_argument(
        "--training-mode",
        choices=["vqa", "gold-phrase", "phrase-only"],
        default="vqa",
        help=(
            "gold-phrase includes the target in the user query; phrase-only keeps "
            "the user query fixed so the target is available only inside <p>...</p>."
        ),
    )
    parser.add_argument(
        "--disable-vlm-image",
        action="store_true",
        help=(
            "Do not provide an image to MedGemma. The original image is still "
            "encoded by MedSAM, creating a single-vision phrase-plus-MedSAM baseline."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--num-prompt-tokens",
        type=int,
        default=1,
        help="Number K of 256-D semantic prompt tokens produced per phrase.",
    )
    parser.add_argument(
        "--grounding-decoder-mode",
        choices=["medsam", "aligner-medsam", "direct-mask"],
        default="medsam",
        help=(
            "medsam uses the legacy projector; aligner-medsam inserts the gated "
            "semantic-spatial aligner before MedSAM; direct-mask replaces the "
            "MedSAM mask decoder with a direct pixel mask head."
        ),
    )
    parser.add_argument("--aligner-heads", type=int, default=8)
    parser.add_argument("--spatial-merge-factor", type=int, default=2)
    parser.add_argument(
        "--spatial-topk",
        type=int,
        default=64,
        help="Spatial keys retained per query and attention head; 0 means dense.",
    )
    parser.add_argument("--swiglu-expansion", type=int, default=4)
    parser.add_argument("--aligner-dropout", type=float, default=0.0)
    parser.add_argument(
        "--use-predicted-bbox",
        action="store_true",
        help="Predict a spatial box from the phrase hidden state and pass it to MedSAM.",
    )
    parser.add_argument("--bbox-l1-loss-weight", type=float, default=1.0)
    parser.add_argument("--bbox-giou-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--bbox-head-only",
        action="store_true",
        help="Freeze LoRA, projector and MedSAM modules while warming up the bbox head.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument(
        "--eval-strategy",
        choices=["steps", "epoch"],
        default="steps",
        help="Use epoch for large validation sets to avoid excessive evaluation.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many evaluations without lower eval_loss; 0 disables it.",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0,
        help="Minimum eval_loss improvement required to reset early-stopping patience.",
    )
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--train-sam-image-encoder", action="store_true")
    parser.add_argument(
        "--disable-sam-image-features",
        action="store_true",
        help=(
            "Replace MedSAM image embeddings with zeros during training and "
            "evaluation. This is a negative spatial-evidence ablation."
        ),
    )
    parser.add_argument(
        "--switch-loss-weight",
        type=float,
        default=0.0,
        help="Weight for same-image, different-region grounding margin loss.",
    )
    parser.add_argument(
        "--switch-margin",
        type=float,
        default=0.2,
        help="Required soft-Dice gap between the correct and switched masks.",
    )
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument(
        "--sampler-power",
        type=float,
        default=0.5,
        help="Class weight is (1 / class_count) ** sampler_power.",
    )
    parser.add_argument(
        "--class-balanced-loss",
        action="store_true",
        help="Weight segmentation losses using the effective number of samples.",
    )
    parser.add_argument(
        "--class-balance-beta",
        type=float,
        default=0.99,
        help="Beta used by effective-number class weights.",
    )
    parser.add_argument("--class-weight-min", type=float, default=0.5)
    parser.add_argument("--class-weight-max", type=float, default=3.0)
    parser.add_argument(
        "--tail-augmentation",
        action="store_true",
        help="Apply paired spatial augmentation only to rare training classes.",
    )
    parser.add_argument(
        "--tail-max-count",
        type=int,
        default=3,
        help="A class is treated as tail when its training count is at most this value.",
    )
    parser.add_argument(
        "--tail-oversample-target",
        type=int,
        default=12,
        help="Approximate per-epoch count targeted for augmented tail classes; 0 disables oversampling.",
    )
    parser.add_argument("--augmentation-probability", type=float, default=0.8)
    parser.add_argument("--augmentation-rotation-degrees", type=float, default=10.0)
    parser.add_argument("--augmentation-translation-fraction", type=float, default=0.05)
    parser.add_argument("--augmentation-intensity-fraction", type=float, default=0.1)
    args = parser.parse_args()

    if args.tail_max_count < 1:
        raise ValueError("--tail-max-count must be at least 1.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative.")
    if args.edge_loss_weight < 0:
        raise ValueError("--edge-loss-weight cannot be negative.")
    if args.edge_kernel_size < 3 or args.edge_kernel_size % 2 == 0:
        raise ValueError("--edge-kernel-size must be an odd integer of at least 3.")
    if args.early_stopping_patience > 0 and args.no_eval:
        raise ValueError("Early stopping requires evaluation; remove --no-eval.")
    if args.tail_oversample_target < 0:
        raise ValueError("--tail-oversample-target cannot be negative.")
    if not 0.0 <= args.augmentation_probability <= 1.0:
        raise ValueError("--augmentation-probability must be in [0, 1].")
    if not 0.0 <= args.augmentation_translation_fraction <= 0.25:
        raise ValueError("--augmentation-translation-fraction must be in [0, 0.25].")
    if not 0.0 <= args.augmentation_intensity_fraction < 1.0:
        raise ValueError("--augmentation-intensity-fraction must be in [0, 1).")

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed GPU training requires CUDA.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(
            f"DDP enabled: rank={os.environ.get('RANK', '0')}; "
            f"local_rank={local_rank}; world_size={world_size}; device={device}"
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = get_tokenizer(processor)
    # Do not add randomly initialized special tokens: LoRA freezes the base
    # embedding/lm-head matrices. Use the base tokenizer's existing subword
    # sequence for the textual closing marker instead.
    close_token_ids = tokenizer.encode(CLOSE_TAG, add_special_tokens=False)
    open_token_ids = build_open_tag_token_variants(tokenizer)
    if not close_token_ids:
        raise ValueError(f"Tokenizer cannot encode {CLOSE_TAG}.")
    if not open_token_ids:
        raise ValueError(f"Tokenizer cannot encode {OPEN_TAG}.")

    model_load_kwargs = {"dtype": torch.bfloat16}
    if not distributed:
        model_load_kwargs["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, **model_load_kwargs
    )
    model.config.use_cache = False
    # Non-reentrant checkpointing is compatible with DDP parameters used by
    # the grounding objective outside the base VLM forward. Reentrant mode can
    # fire the same LoRA reduction hook twice.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )

    sam = load_medsam(args.sam_checkpoint, args.sam_model_type).to(device)
    freeze_encoder = not args.train_sam_image_encoder
    for parameter in sam.image_encoder.parameters():
        parameter.requires_grad = not freeze_encoder
    for parameter in sam.prompt_encoder.parameters():
        parameter.requires_grad = False
    for parameter in sam.mask_decoder.parameters():
        parameter.requires_grad = args.grounding_decoder_mode != "direct-mask"
    model.phrase_projector = PhraseProjector(
        get_hidden_size(model.config), 256, args.num_prompt_tokens
    ).to(device)
    if args.use_predicted_bbox:
        model.phrase_bbox_head = PhraseBBoxHead(get_hidden_size(model.config)).to(device)
    model.localization_image_encoder = sam.image_encoder
    model.localization_prompt_encoder = sam.prompt_encoder
    model.localization_mask_decoder = sam.mask_decoder
    model.sam_image_features_enabled = not args.disable_sam_image_features
    model.grounding_decoder_mode = args.grounding_decoder_mode
    if args.grounding_decoder_mode != "medsam":
        model.semantic_spatial_aligner = GatedSemanticSpatialAligner(
            language_dim=get_hidden_size(model.config),
            prompt_dim=256,
            num_heads=args.aligner_heads,
            spatial_merge_factor=args.spatial_merge_factor,
            spatial_topk=args.spatial_topk,
            swiglu_expansion=args.swiglu_expansion,
            dropout=args.aligner_dropout,
        ).to(device)
        for parameter in model.phrase_projector.parameters():
            parameter.requires_grad = False
    if args.grounding_decoder_mode == "direct-mask":
        model.direct_mask_head = DirectPixelMaskHead(dim=256, output_size=256).to(device)
    if args.init_adapter:
        init_adapter = Path(args.init_adapter)
        load_phrase_projector_state(
            model.phrase_projector,
            torch.load(init_adapter / "phrase_projector.pt", map_location=device),
        )
        model.localization_mask_decoder.load_state_dict(
            torch.load(init_adapter / "mask_decoder.pt", map_location=device)
        )
        if (init_adapter / "image_encoder.pt").exists():
            model.localization_image_encoder.load_state_dict(
                torch.load(init_adapter / "image_encoder.pt", map_location=device)
            )
        if args.use_predicted_bbox and (init_adapter / "phrase_bbox_head.pt").exists():
            model.phrase_bbox_head.load_state_dict(
                torch.load(init_adapter / "phrase_bbox_head.pt", map_location=device)
            )
        aligner_path = init_adapter / "semantic_spatial_aligner.pt"
        if hasattr(model, "semantic_spatial_aligner") and aligner_path.exists():
            model.semantic_spatial_aligner.load_state_dict(
                torch.load(aligner_path, map_location=device)
            )
        direct_head_path = init_adapter / "direct_mask_head.pt"
        if hasattr(model, "direct_mask_head") and direct_head_path.exists():
            model.direct_mask_head.load_state_dict(
                torch.load(direct_head_path, map_location=device)
            )
        print(f"Initialized grounding modules from: {init_adapter}")
    if args.bbox_head_only:
        if not args.use_predicted_bbox:
            raise ValueError("--bbox-head-only requires --use-predicted-bbox.")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.phrase_bbox_head.parameters():
            parameter.requires_grad = True
        print("BBox warm-up: only phrase_bbox_head is trainable.")
    model.print_trainable_parameters()
    print(f"No vocabulary resize; close token ids: {close_token_ids}; frozen SAM encoder: {freeze_encoder}")

    data_root = Path(args.data_root)
    train_dataset = SlakeSegDataset(
        data_root / args.train_file,
        data_root,
        processor,
        include_switch_target=args.switch_loss_weight > 0,
        training_mode=args.training_mode,
        use_vlm_image=not args.disable_vlm_image,
        augment_tail=args.tail_augmentation,
        tail_max_count=args.tail_max_count,
        augmentation_probability=args.augmentation_probability,
        augmentation_rotation_degrees=args.augmentation_rotation_degrees,
        augmentation_translation_fraction=args.augmentation_translation_fraction,
        augmentation_intensity_fraction=args.augmentation_intensity_fraction,
    )
    eval_dataset = None if args.no_eval else SlakeSegDataset(
        data_root / args.val_file,
        data_root,
        processor,
        training_mode=args.training_mode,
        use_vlm_image=not args.disable_vlm_image,
    )
    region_counts = Counter(row["region"] for row in train_dataset.rows)
    class_loss_weights = {}
    if args.class_balanced_loss:
        if not 0.0 <= args.class_balance_beta < 1.0:
            raise ValueError("--class-balance-beta must be in [0, 1).")
        if args.class_weight_min <= 0 or args.class_weight_max < args.class_weight_min:
            raise ValueError("Invalid class-weight clipping range.")
        beta = args.class_balance_beta
        raw_weights = {
            region: (1.0 - beta) / (1.0 - beta ** count)
            if beta > 0.0 else 1.0
            for region, count in region_counts.items()
        }
        mean_weight = sum(raw_weights.values()) / len(raw_weights)
        class_loss_weights = {
            region: min(
                args.class_weight_max,
                max(args.class_weight_min, weight / mean_weight),
            )
            for region, weight in raw_weights.items()
        }
        train_dataset.region_loss_weights = class_loss_weights
        if eval_dataset is not None:
            eval_dataset.region_loss_weights = class_loss_weights
        print(
            "Effective-number class-balanced loss enabled: "
            f"beta={beta}; weights={class_loss_weights}"
        )
    train_sample_weights = None
    if args.balanced_sampling:
        train_sample_weights = torch.tensor(
            [
                (1.0 / region_counts[row["region"]]) ** args.sampler_power
                for row in train_dataset.rows
            ],
            dtype=torch.double,
        )
        print(
            "Balanced sampling enabled: "
            f"power={args.sampler_power}; class counts={dict(region_counts.most_common())}"
        )
    elif args.tail_augmentation and args.tail_oversample_target > 0:
        train_sample_weights = torch.tensor(
            [
                max(
                    1.0,
                    args.tail_oversample_target / region_counts[row["region"]],
                )
                if region_counts[row["region"]] <= args.tail_max_count
                else 1.0
                for row in train_dataset.rows
            ],
            dtype=torch.double,
        )
        print(
            "Tail augmentation enabled: "
            f"max_count={args.tail_max_count}; "
            f"oversample_target={args.tail_oversample_target}; "
            f"regions={sorted(train_dataset.tail_regions)}"
        )
    if distributed and train_sample_weights is not None:
        raise ValueError(
            "Weighted sampling is not yet distributed-safe. Disable balanced "
            "sampling/tail oversampling for DDP, or run this experiment on one GPU."
        )
    use_early_stopping = args.early_stopping_patience > 0
    evaluation_strategy = "no" if args.no_eval else args.eval_strategy
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy=evaluation_strategy,
        save_strategy=args.eval_strategy if not args.no_eval else "steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss" if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False if distributed else None,
    )
    callbacks = [GroundingModulesCheckpointCallback(not freeze_encoder)]
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    trainer = SegTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SingleBatchCollator(),
        close_token_ids=close_token_ids,
        open_token_ids=open_token_ids,
        text_loss_weight=args.text_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        bce_loss_weight=args.bce_loss_weight,
        focal_loss_weight=args.focal_loss_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        edge_loss_weight=args.edge_loss_weight,
        edge_kernel_size=args.edge_kernel_size,
        freeze_image_encoder=freeze_encoder,
        train_sample_weights=train_sample_weights,
        switch_loss_weight=args.switch_loss_weight,
        switch_margin=args.switch_margin,
        bbox_l1_loss_weight=args.bbox_l1_loss_weight,
        bbox_giou_loss_weight=args.bbox_giou_loss_weight,
        callbacks=callbacks,
    )
    trainer.train()
    if use_early_stopping and trainer.state.best_model_checkpoint:
        restore_grounding_modules(
            model, trainer.state.best_model_checkpoint, device
        )
        print(
            "Restored best grounding modules from: "
            f"{trainer.state.best_model_checkpoint}; "
            f"best eval_loss={trainer.state.best_metric}"
        )
    trainer.save_model(args.output_dir)
    trainer.accelerator.wait_for_everyone()
    if not trainer.is_world_process_zero():
        return
    processor.save_pretrained(args.output_dir)
    output_dir = Path(args.output_dir)
    torch.save(model.phrase_projector.state_dict(), output_dir / "phrase_projector.pt")
    torch.save(model.localization_mask_decoder.state_dict(), output_dir / "mask_decoder.pt")
    if hasattr(model, "semantic_spatial_aligner"):
        torch.save(
            model.semantic_spatial_aligner.state_dict(),
            output_dir / "semantic_spatial_aligner.pt",
        )
    if hasattr(model, "direct_mask_head"):
        torch.save(
            model.direct_mask_head.state_dict(),
            output_dir / "direct_mask_head.pt",
        )
    if hasattr(model, "phrase_bbox_head"):
        torch.save(model.phrase_bbox_head.state_dict(), output_dir / "phrase_bbox_head.pt")
    if not freeze_encoder:
        torch.save(model.localization_image_encoder.state_dict(), output_dir / "image_encoder.pt")
    (output_dir / "vividmed_lite_seg_meta.json").write_text(
        json.dumps(
            {
                "sam_model_type": args.sam_model_type,
                "sam_checkpoint_initialization": str(args.sam_checkpoint),
                "sam_image_encoder_frozen": freeze_encoder,
                "sam_image_features_enabled": not args.disable_sam_image_features,
                "close_token_ids": close_token_ids,
                "open_token_ids": open_token_ids,
                "loss_weights": {
                    "text": args.text_loss_weight,
                    "dice": args.dice_loss_weight,
                    "bce": args.bce_loss_weight,
                    "focal": args.focal_loss_weight,
                    "edge": args.edge_loss_weight,
                    "bbox_l1": args.bbox_l1_loss_weight,
                    "bbox_giou": args.bbox_giou_loss_weight,
                },
                "focal_alpha": args.focal_alpha,
                "focal_gamma": args.focal_gamma,
                "training_mode": args.training_mode,
                "vlm_image_enabled": not args.disable_vlm_image,
                "num_prompt_tokens": args.num_prompt_tokens,
                "grounding_decoder_mode": args.grounding_decoder_mode,
                "aligner_heads": args.aligner_heads,
                "spatial_merge_factor": args.spatial_merge_factor,
                "spatial_topk": args.spatial_topk,
                "swiglu_expansion": args.swiglu_expansion,
                "aligner_dropout": args.aligner_dropout,
                "use_predicted_bbox": args.use_predicted_bbox,
                "bbox_head_only": args.bbox_head_only,
                "initialized_from": args.init_adapter,
                "balanced_sampling": args.balanced_sampling,
                "sampler_power": args.sampler_power,
                "class_balanced_loss": args.class_balanced_loss,
                "class_balance_beta": args.class_balance_beta,
                "class_weight_min": args.class_weight_min,
                "class_weight_max": args.class_weight_max,
                "class_loss_weights": class_loss_weights,
                "tail_augmentation": args.tail_augmentation,
                "tail_max_count": args.tail_max_count,
                "tail_oversample_target": args.tail_oversample_target,
                "augmentation_probability": args.augmentation_probability,
                "augmentation_rotation_degrees": args.augmentation_rotation_degrees,
                "augmentation_translation_fraction": args.augmentation_translation_fraction,
                "augmentation_intensity_fraction": args.augmentation_intensity_fraction,
                "switch_loss_weight": args.switch_loss_weight,
                "switch_margin": args.switch_margin,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
