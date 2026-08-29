import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


SYSTEM_PROMPT = "You are a medical visual question answering assistant."
OPEN_TAG = "<p>"
CLOSE_TAG = "</p>"


def get_tokenizer(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def get_hidden_size(config):
    for obj in (config, getattr(config, "text_config", None), getattr(config, "language_config", None)):
        if obj is not None and hasattr(obj, "hidden_size"):
            return int(obj.hidden_size)
    raise ValueError("Cannot infer language-model hidden size from config.")


class PhraseProjector(nn.Module):
    def __init__(self, hidden_size, prompt_dim=256, num_prompt_tokens=1):
        super().__init__()
        if num_prompt_tokens < 1:
            raise ValueError("num_prompt_tokens must be at least 1.")
        self.prompt_dim = int(prompt_dim)
        self.num_prompt_tokens = int(num_prompt_tokens)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, prompt_dim * num_prompt_tokens),
        )
        self.query_offsets = (
            nn.Parameter(torch.empty(num_prompt_tokens, prompt_dim))
            if num_prompt_tokens > 1
            else None
        )
        if self.query_offsets is not None:
            nn.init.normal_(self.query_offsets, mean=0.0, std=0.02)

    def forward(self, hidden):
        prompts = self.net(hidden.float()).reshape(
            *hidden.shape[:-1], self.num_prompt_tokens, self.prompt_dim
        )
        if self.query_offsets is not None:
            prompts = prompts + self.query_offsets
        return prompts


class PhraseBBoxHead(nn.Module):
    """Predict a normalized xyxy box from a grounded phrase hidden state."""

    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 4),
        )

    def forward(self, hidden):
        raw = self.net(hidden.float())
        center = raw[..., :2].sigmoid()
        size = raw[..., 2:].sigmoid()
        xy_min = (center - 0.5 * size).clamp(0.0, 1.0)
        xy_max = (center + 0.5 * size).clamp(0.0, 1.0)
        return torch.cat((xy_min, xy_max), dim=-1)


class SpatialTokenMerger(nn.Module):
    """Reduce dense MedSAM tokens while preserving their local spatial context."""

    def __init__(self, dim=256, merge_factor=2):
        super().__init__()
        if merge_factor < 1:
            raise ValueError("merge_factor must be at least 1.")
        self.merge_factor = int(merge_factor)
        merged_dim = dim * self.merge_factor * self.merge_factor
        self.norm = nn.LayerNorm(merged_dim)
        self.fc1 = nn.Linear(merged_dim, merged_dim)
        self.fc2 = nn.Linear(merged_dim, dim)

    def forward(self, image_embeddings):
        batch, channels, height, width = image_embeddings.shape
        factor = self.merge_factor
        if height % factor or width % factor:
            raise ValueError(
                f"Image feature size {(height, width)} is not divisible by {factor}."
            )
        patches = F.unfold(
            image_embeddings, kernel_size=factor, stride=factor
        ).transpose(1, 2)
        return self.fc2(F.gelu(self.fc1(self.norm(patches))))


class TopKCrossAttention(nn.Module):
    """Cross-attention with optional per-query hard spatial top-k selection."""

    def __init__(self, dim=256, num_heads=8, topk=0, dropout=0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.topk = int(topk)
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _heads(self, tensor):
        batch, length, _ = tensor.shape
        return tensor.reshape(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, queries, image_tokens):
        query = self._heads(self.q_proj(queries))
        key = self._heads(self.k_proj(image_tokens))
        value = self._heads(self.v_proj(image_tokens))
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.head_dim**-0.5
        if 0 < self.topk < scores.shape[-1]:
            selected = scores.topk(self.topk, dim=-1).indices
            keep = torch.zeros_like(scores, dtype=torch.bool)
            keep.scatter_(-1, selected, True)
            scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous()
        attended = attended.reshape(queries.shape[0], queries.shape[1], self.dim)
        return self.out_proj(attended), weights


class SwiGLU(nn.Module):
    def __init__(self, dim=256, expansion=4):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, tensor):
        return self.down_proj(F.silu(self.gate_proj(tensor)) * self.up_proj(tensor))


class GatedSemanticSpatialAligner(nn.Module):
    """Turn phrase-token states into image-conditioned MedSAM prompt tokens."""

    def __init__(
        self,
        language_dim,
        prompt_dim=256,
        num_heads=8,
        spatial_merge_factor=2,
        spatial_topk=64,
        swiglu_expansion=4,
        dropout=0.0,
    ):
        super().__init__()
        self.language_norm = nn.LayerNorm(language_dim)
        self.query_proj = nn.Linear(language_dim, prompt_dim)
        self.query_norm = nn.LayerNorm(prompt_dim)
        self.spatial_merger = SpatialTokenMerger(
            dim=prompt_dim, merge_factor=spatial_merge_factor
        )
        self.spatial_norm = nn.LayerNorm(prompt_dim)
        self.cross_attention = TopKCrossAttention(
            dim=prompt_dim,
            num_heads=num_heads,
            topk=spatial_topk,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(prompt_dim)
        self.swiglu = SwiGLU(prompt_dim, expansion=swiglu_expansion)
        self.residual_gate = nn.Linear(prompt_dim * 2, prompt_dim)
        # Start close to the identity mapping and introduce visual evidence gradually.
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, -2.0)
        self.output_norm = nn.LayerNorm(prompt_dim)

    def forward(self, phrase_hidden, image_embeddings, return_attention=False):
        if phrase_hidden.ndim == 2:
            phrase_hidden = phrase_hidden.unsqueeze(1)
        queries = self.query_proj(self.language_norm(phrase_hidden.float()))
        image_tokens = self.spatial_norm(self.spatial_merger(image_embeddings.float()))
        retrieved, attention = self.cross_attention(
            self.query_norm(queries), image_tokens
        )
        transformed = self.swiglu(self.ffn_norm(retrieved))
        gate = torch.sigmoid(self.residual_gate(torch.cat((queries, transformed), dim=-1)))
        aligned = self.output_norm(queries + gate * transformed)
        if return_attention:
            return aligned, attention
        return aligned


class DirectPixelMaskHead(nn.Module):
    """Predict one mask directly from aligned phrase queries and MedSAM pixels."""

    def __init__(self, dim=256, output_size=256):
        super().__init__()
        self.output_size = int(output_size)
        self.pixel_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.query_score = nn.Linear(dim, 1)
        self.logit_scale = nn.Parameter(torch.tensor(dim**-0.5).log())

    def forward(self, aligned_queries, image_embeddings):
        pixels = self.pixel_refine(image_embeddings.float())
        queries = self.query_proj(aligned_queries.float())
        per_query_logits = torch.einsum("bld,bdhw->blhw", queries, pixels)
        per_query_logits = per_query_logits * self.logit_scale.exp().clamp(max=10.0)
        query_weights = F.softmax(self.query_score(aligned_queries.float()), dim=1)
        logits = (per_query_logits * query_weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
        if logits.shape[-2:] != (self.output_size, self.output_size):
            logits = F.interpolate(
                logits,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        quality = logits.sigmoid().flatten(1).mean(dim=1, keepdim=True)
        return logits, quality


def masks_to_normalized_boxes(masks):
    """Convert non-empty binary masks [B,1,H,W] to normalized xyxy boxes."""
    boxes = []
    for mask in masks[:, 0] >= 0.5:
        coordinates = mask.nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError("Cannot derive a bounding box from an empty target mask.")
        height, width = mask.shape
        y_min, x_min = coordinates.min(dim=0).values
        y_max, x_max = coordinates.max(dim=0).values
        boxes.append(
            torch.stack(
                (
                    x_min.float() / width,
                    y_min.float() / height,
                    (x_max.float() + 1.0) / width,
                    (y_max.float() + 1.0) / height,
                )
            )
        )
    return torch.stack(boxes, dim=0)


def box_iou_xyxy(boxes_a, boxes_b, eps=1e-6):
    intersection_min = torch.maximum(boxes_a[..., :2], boxes_b[..., :2])
    intersection_max = torch.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    intersection_size = (intersection_max - intersection_min).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area_a = (boxes_a[..., 2] - boxes_a[..., 0]).clamp(min=0) * (
        boxes_a[..., 3] - boxes_a[..., 1]
    ).clamp(min=0)
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]).clamp(min=0) * (
        boxes_b[..., 3] - boxes_b[..., 1]
    ).clamp(min=0)
    union = area_a + area_b - intersection
    return (intersection + eps) / (union + eps)


def generalized_box_iou_loss(boxes_a, boxes_b, eps=1e-6):
    iou = box_iou_xyxy(boxes_a, boxes_b, eps=eps)
    enclosing_min = torch.minimum(boxes_a[..., :2], boxes_b[..., :2])
    enclosing_max = torch.maximum(boxes_a[..., 2:], boxes_b[..., 2:])
    enclosing_size = (enclosing_max - enclosing_min).clamp(min=0)
    enclosing_area = enclosing_size[..., 0] * enclosing_size[..., 1]
    area_a = (boxes_a[..., 2] - boxes_a[..., 0]).clamp(min=0) * (
        boxes_a[..., 3] - boxes_a[..., 1]
    ).clamp(min=0)
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]).clamp(min=0) * (
        boxes_b[..., 3] - boxes_b[..., 1]
    ).clamp(min=0)
    intersection_min = torch.maximum(boxes_a[..., :2], boxes_b[..., :2])
    intersection_max = torch.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    intersection_size = (intersection_max - intersection_min).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    union = area_a + area_b - intersection
    giou = iou - (enclosing_area - union) / enclosing_area.clamp(min=eps)
    return (1.0 - giou).mean()


def load_phrase_projector_state(projector, state_dict):
    """Load a projector, expanding a legacy K=1 output layer when needed."""
    target_state = projector.state_dict()
    output_weight_key = "net.3.weight"
    output_bias_key = "net.3.bias"
    source_weight = state_dict.get(output_weight_key)
    target_weight = target_state.get(output_weight_key)
    if (
        source_weight is not None
        and target_weight is not None
        and source_weight.shape != target_weight.shape
    ):
        if target_weight.shape[0] % source_weight.shape[0] != 0:
            raise ValueError(
                "Cannot expand phrase projector output layer from "
                f"{tuple(source_weight.shape)} to {tuple(target_weight.shape)}."
            )
        repeats = target_weight.shape[0] // source_weight.shape[0]
        state_dict = dict(state_dict)
        state_dict[output_weight_key] = source_weight.repeat(repeats, 1)
        source_bias = state_dict.get(output_bias_key)
        if source_bias is not None:
            state_dict[output_bias_key] = source_bias.repeat(repeats)
    missing, unexpected = projector.load_state_dict(state_dict, strict=False)
    allowed_missing = {"query_offsets"} if projector.query_offsets is not None else set()
    invalid_missing = set(missing) - allowed_missing
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Phrase projector checkpoint mismatch: "
            f"missing={sorted(invalid_missing)}, unexpected={sorted(unexpected)}"
        )


def load_medsam(checkpoint, model_type="vit_b"):
    try:
        from segment_anything import sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "segment-anything is required. Install it with: "
            "pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc
    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type: {model_type}")
    return sam_model_registry[model_type](checkpoint=str(checkpoint))


def medsam_image_tensor(image, size=1024):
    # MedSAM's released inference pipeline uses square resize followed by
    # per-image min-max normalization rather than SAM's RGB mean/std transform.
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    minimum = float(array.min())
    maximum = float(array.max())
    array = (array - minimum) / max(maximum - minimum, 1e-8)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_binary_mask(path, size=256):
    mask = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    array = (np.asarray(mask) > 0).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def find_last_subsequence(sequence, pattern, valid_mask=None):
    if not pattern:
        return None
    sequence = sequence.tolist()
    pattern = list(pattern)
    valid = valid_mask.tolist() if valid_mask is not None else None
    for start in range(len(sequence) - len(pattern), -1, -1):
        end = start + len(pattern)
        if sequence[start:end] != pattern:
            continue
        if valid is not None and not all(valid[start:end]):
            continue
        return end - 1
    return None


def build_open_tag_token_variants(tokenizer):
    """Encode opening-tag contexts that SentencePiece may segment differently."""
    texts = (
        OPEN_TAG,
        f" {OPEN_TAG}",
        f":{OPEN_TAG}",
        f": {OPEN_TAG}",
        f"Relevant region: {OPEN_TAG}",
        f" Relevant region: {OPEN_TAG}",
        f"\nRelevant region: {OPEN_TAG}",
    )
    variants = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids and token_ids not in variants:
            variants.append(token_ids)
    return variants


def select_phrase_hidden(hidden, input_ids, close_token_ids, labels=None):
    selected = []
    for batch_index in range(input_ids.shape[0]):
        valid_mask = labels[batch_index] != -100 if labels is not None else None
        position = find_last_subsequence(
            input_ids[batch_index], close_token_ids, valid_mask=valid_mask
        )
        if position is None:
            raise ValueError(f"No closing phrase token {CLOSE_TAG} was found.")
        selected.append(hidden[batch_index, position])
    return torch.stack(selected, dim=0)


def select_phrase_token_hiddens(
    hidden,
    input_ids,
    open_token_ids,
    close_token_ids,
    labels=None,
):
    """Select all content-token states inside the final <p>...</p> span."""
    spans = []
    for batch_index in range(input_ids.shape[0]):
        sequence = input_ids[batch_index]
        valid_mask = labels[batch_index] != -100 if labels is not None else None
        close_end = find_last_subsequence(
            sequence, close_token_ids, valid_mask=valid_mask
        )
        if close_end is None:
            raise ValueError(f"No closing phrase token {CLOSE_TAG} was found.")
        close_start = close_end - len(close_token_ids) + 1
        prefix = sequence[:close_start]
        prefix_valid = valid_mask[:close_start] if valid_mask is not None else None
        if open_token_ids and isinstance(open_token_ids[0], int):
            open_patterns = [open_token_ids]
        else:
            open_patterns = open_token_ids
        open_matches = [
            find_last_subsequence(prefix, pattern, valid_mask=prefix_valid)
            for pattern in open_patterns
        ]
        open_matches = [position for position in open_matches if position is not None]
        open_end = max(open_matches) if open_matches else None
        if open_end is None:
            # SentencePiece may encode the opening tag differently after a
            # space even though the final '>' token remains stable. Because
            # the assistant format is fixed, the last '>' before </p> is the
            # end of the matching <p> marker.
            marker_id = int(close_token_ids[-1])
            marker_positions = (prefix == marker_id).nonzero(as_tuple=False)
            if prefix_valid is not None and marker_positions.numel() > 0:
                marker_positions = marker_positions[
                    prefix_valid[marker_positions[:, 0]]
                ]
            if marker_positions.numel() == 0:
                raise ValueError(
                    f"No opening phrase token {OPEN_TAG} or fallback marker was found."
                )
            open_end = int(marker_positions[-1, 0].item())
        start = open_end + 1
        if start >= close_start:
            # Fall back to the closing marker state for malformed empty spans.
            spans.append(hidden[batch_index, close_end : close_end + 1])
        else:
            spans.append(hidden[batch_index, start:close_start])

    max_length = max(span.shape[0] for span in spans)
    padded = hidden.new_zeros((len(spans), max_length, hidden.shape[-1]))
    padding_mask = torch.zeros(
        (len(spans), max_length), dtype=torch.bool, device=hidden.device
    )
    for batch_index, span in enumerate(spans):
        padded[batch_index, : span.shape[0]] = span
        padding_mask[batch_index, : span.shape[0]] = True
    return padded, padding_mask


def pooled_phrase_hidden(phrase_hidden, phrase_mask=None):
    if phrase_hidden.ndim == 2:
        return phrase_hidden
    if phrase_mask is None:
        return phrase_hidden.mean(dim=1)
    weights = phrase_mask.to(phrase_hidden.dtype).unsqueeze(-1)
    return (phrase_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def decode_phrase_masks(
    model,
    phrase_hidden,
    sam_pixels,
    freeze_image_encoder=True,
    use_predicted_bbox=False,
    phrase_mask=None,
):
    device = next(model.phrase_projector.parameters()).device
    sam_pixels = sam_pixels.to(device=device, dtype=torch.float32)

    if freeze_image_encoder:
        model.localization_image_encoder.eval()
        with torch.no_grad():
            image_embeddings = model.localization_image_encoder(sam_pixels)
    else:
        image_embeddings = model.localization_image_encoder(sam_pixels)

    if getattr(model, "sam_image_features_enabled", True) is False:
        # Keep the decoder interface and positional encoding intact while
        # removing all image-dependent MedSAM spatial evidence.
        image_embeddings = torch.zeros_like(image_embeddings)

    grounding_mode = getattr(model, "grounding_decoder_mode", "medsam")
    if grounding_mode == "medsam":
        phrase_prompt = model.phrase_projector(phrase_hidden.to(device))
    else:
        phrase_prompt = model.semantic_spatial_aligner(
            phrase_hidden.to(device), image_embeddings
        )
        if phrase_mask is not None:
            phrase_prompt = phrase_prompt * phrase_mask.to(
                device=device, dtype=phrase_prompt.dtype
            ).unsqueeze(-1)

    if grounding_mode == "direct-mask":
        low_res_logits, quality = model.direct_mask_head(
            phrase_prompt, image_embeddings
        )
        return low_res_logits, quality, None

    model.localization_prompt_encoder.eval()
    predicted_boxes = None
    if use_predicted_bbox:
        predicted_boxes = model.phrase_bbox_head(
            pooled_phrase_hidden(phrase_hidden.to(device), phrase_mask)
        )
        scale = torch.tensor(
            [sam_pixels.shape[-1], sam_pixels.shape[-2]] * 2,
            device=device,
            dtype=predicted_boxes.dtype,
        )
        box_embeddings, dense_embeddings = model.localization_prompt_encoder(
            points=None, boxes=predicted_boxes * scale, masks=None
        )
        phrase_prompt = torch.cat((phrase_prompt, box_embeddings), dim=1)
        with torch.no_grad():
            image_pe = model.localization_prompt_encoder.get_dense_pe()
    else:
        with torch.no_grad():
            _, dense_embeddings = model.localization_prompt_encoder(
                points=None, boxes=None, masks=None
            )
            image_pe = model.localization_prompt_encoder.get_dense_pe()
    if dense_embeddings.shape[0] != image_embeddings.shape[0]:
        dense_embeddings = dense_embeddings.expand(image_embeddings.shape[0], -1, -1, -1)
    if image_pe.shape[0] != image_embeddings.shape[0]:
        image_pe = image_pe.expand(image_embeddings.shape[0], -1, -1, -1)

    low_res_logits, quality = model.localization_mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=phrase_prompt,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_logits, quality, predicted_boxes


def dice_loss_with_logits(logits, target, eps=1e-6):
    probabilities = logits.sigmoid()
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * target).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def binary_iou_and_dice(logits, target, threshold=0.5, eps=1e-6):
    prediction = logits.sigmoid() >= threshold
    target = target >= 0.5
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction & target).sum(dim=dims).float()
    union = (prediction | target).sum(dim=dims).float()
    pred_area = prediction.sum(dim=dims).float()
    target_area = target.sum(dim=dims).float()
    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (pred_area + target_area + eps)
    return iou, dice


def norm_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\b(?:end of turn|start of turn)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_region(text):
    match = re.search(r"<p>\s*(.*?)\s*</p>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return norm_text(match.group(1))
    match = re.search(r"relevant\s+region\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    return norm_text(match.group(1)) if match else ""


def extract_answer(text):
    match = re.search(
        r"answer\s*:\s*(.*?)(?:\n|<end_of_turn>|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return norm_text(match.group(1)) if match else norm_text(text)


def resize_logits(logits, height, width):
    return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
