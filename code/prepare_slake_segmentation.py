import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


EXCLUDED_CONTENT_TYPES = {"KG", "Modality", "Plane", "Quality"}

# Conservative aliases: ambiguous samples are rejected instead of receiving a
# potentially unrelated mask.
LABEL_ALIASES = {
    "Brain Edema": ["brain edema", "cerebral edema", "edema"],
    "Brain Non-enhancing Tumor": ["brain non enhancing tumor", "non enhancing tumor", "non-enhancing tumor"],
    "Brain Enhancing Tumor": ["brain enhancing tumor", "enhancing tumor"],
    "Small Bowel": ["small bowel", "small intestine"],
    "Spinal Cord": ["spinal cord"],
    "Liver Cancer": ["liver cancer", "hepatic cancer"],
    "Lung Cancer": ["lung cancer", "pulmonary cancer"],
    "Kidney Cancer": ["kidney cancer", "renal cancer"],
    "Left Kidney": ["left kidney", "left renal"],
    "Right Kidney": ["right kidney", "right renal"],
    "Right Lung": ["right lung", "right pulmonary"],
    "Left Lung": ["left lung", "left pulmonary"],
    "Brain Stem": ["brain stem", "brainstem"],
    "Right Temporal Lobe": ["right temporal lobe"],
    "Left Temporal Lobe": ["left temporal lobe"],
    "Left Parotid": ["left parotid"],
    "Right Parotid": ["right parotid"],
    "Left Mandible": ["left mandible", "left jaw"],
    "Right Mandible": ["right mandible", "right jaw"],
    "Right Femoral Head": ["right femoral head"],
    "Left Femoral Head": ["left femoral head"],
    "Left Humerus Head": ["left humerus head"],
    "Right Humerus Head": ["right humerus head"],
    "Left Eye": ["left eye"],
    "Right Eye": ["right eye"],
    "Left Ear": ["left ear"],
    "Right Ear": ["right ear"],
    "Liver": ["liver", "hepatic organ"],
    "Colon": ["colon", "large bowel", "large intestine"],
    "Spleen": ["spleen", "splenic organ"],
    "Rectum": ["rectum", "rectal"],
    "Esophagus": ["esophagus", "oesophagus"],
    "Heart": ["heart", "cardiac"],
    "Bladder": ["bladder"],
    "Duodenum": ["duodenum"],
    "Stomach": ["stomach", "gastric"],
    "Larynx": ["larynx", "laryngeal"],
    "Trachea": ["trachea", "tracheal"],
    "Tooth": ["tooth", "teeth"],
}


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def contains_term(text, term):
    return f" {normalize(term)} " in f" {normalize(text)} "


def load_mask_map(path):
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        value, label = line.split(":", 1)
        mapping[label.strip()] = int(value.strip())
    return mapping


def read_label_mask(path):
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        rgb = array[..., :3]
        if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])):
            raise ValueError(f"Expected a discrete grayscale label mask: {path}")
        array = rgb[..., 0]
    return array


def choose_label(sample, available_labels):
    question = sample.get("question", "")
    answer = sample.get("answer", "")
    scored = []
    for label in available_labels:
        aliases = LABEL_ALIASES.get(label, [label])
        q_match = any(contains_term(question, alias) for alias in aliases)
        a_match = any(contains_term(answer, alias) for alias in aliases)
        if not q_match and not a_match:
            continue
        # An answer match resolves comparison questions; otherwise prefer a
        # target explicitly named in the question.
        score = 3 * int(a_match) + 2 * int(q_match)
        scored.append((score, label))

    if not scored:
        return None, "no_mask_label_match"
    best_score = max(score for score, _ in scored)
    best = sorted({label for score, label in scored if score == best_score})
    if len(best) != 1:
        return None, "ambiguous_mask_label_match"
    return best[0], None


def make_messages(question, region, answer, image_path):
    user_text = (
        "Identify the medical region needed to answer the question, answer briefly, "
        "and ground the region at pixel level.\n"
        "Use the exact text format:\n"
        "Relevant region: <p>region</p>\n"
        "Answer: answer\n"
        f"Question: {question}"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"Relevant region: <p>{region}</p>\nAnswer: {answer}"}
            ],
        },
    ]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slake-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-mask-pixels", type=int, default=16)
    parser.add_argument("--answer-type", choices=["ALL", "CLOSED"], default="ALL")
    args = parser.parse_args()

    slake_root = Path(args.slake_root)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    mask_map = load_mask_map(slake_root / "mask.txt")
    valid_values = set(mask_map.values()) | {0}
    counts = Counter()
    labels = Counter()
    split_rows = {}
    rejected = []

    for split in ("train", "validation", "test"):
        source_rows = json.loads((slake_root / f"{split}.json").read_text(encoding="utf-8"))
        accepted = []
        for sample in source_rows:
            reason = None
            if sample.get("q_lang") != "en":
                reason = "non_english"
            elif sample.get("content_type") in EXCLUDED_CONTENT_TYPES:
                reason = "excluded_content_type"
            elif args.answer_type == "CLOSED" and sample.get("answer_type") != "CLOSED":
                reason = "answer_type_filter"

            case_dir = slake_root / "imgs" / "imgs" / Path(sample["img_name"]).parent
            source_image = slake_root / "imgs" / "imgs" / sample["img_name"]
            source_mask = case_dir / "mask.png"
            if reason is None and (not source_image.exists() or not source_mask.exists()):
                reason = "missing_image_or_mask"

            label = None
            binary = None
            if reason is None:
                try:
                    label_mask = read_label_mask(source_mask)
                except ValueError:
                    reason = "invalid_label_mask"
                else:
                    unique_values = {int(v) for v in np.unique(label_mask)}
                    if not unique_values.issubset(valid_values):
                        reason = "unknown_mask_values"
                    else:
                        available_labels = [name for name, value in mask_map.items() if value in unique_values]
                        label, reason = choose_label(sample, available_labels)
                        if reason is None:
                            binary = label_mask == mask_map[label]
                            if int(binary.sum()) < args.min_mask_pixels:
                                reason = "mask_too_small"

            if reason is not None:
                counts[f"rejected_{reason}"] += 1
                rejected.append({"split": split, "qid": sample.get("qid"), "reason": reason})
                continue

            item_id = f"{split}_{sample['qid']}"
            image_rel = f"images/{Path(sample['img_name']).parent.name}_source.jpg"
            mask_rel = f"masks/{item_id}_{normalize(label).replace(' ', '_')}.png"
            target_image = output_dir / image_rel
            if not target_image.exists():
                shutil.copy2(source_image, target_image)
            Image.fromarray(binary.astype(np.uint8) * 255).save(output_dir / mask_rel)

            width, height = Image.open(source_image).size
            ys, xs = np.nonzero(binary)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            row = {
                "id": item_id,
                "qid": sample["qid"],
                "img_id": sample["img_id"],
                "image": image_rel,
                "mask": mask_rel,
                "question": sample["question"],
                "region": label,
                "answer": sample["answer"],
                "answer_type": sample.get("answer_type"),
                "content_type": sample.get("content_type"),
                "modality": sample.get("modality"),
                "location": sample.get("location"),
                "mask_value": mask_map[label],
                "mask_pixels": int(binary.sum()),
                "image_size": [width, height],
                "bbox_from_mask_xyxy": bbox,
                "messages": make_messages(sample["question"], label, sample["answer"], image_rel),
            }
            accepted.append(row)
            labels[label] += 1
            counts[f"accepted_{split}"] += 1

        split_rows[split] = accepted
        write_jsonl(output_dir / f"{split}.jsonl", accepted)

    write_jsonl(output_dir / "rejected.jsonl", rejected)
    summary = {
        "source": str(slake_root),
        "mask_source": "official SLAKE per-image mask.png + mask.txt",
        "target_mask": "question-specific binary mask selected by region label",
        "answer_type_filter": args.answer_type,
        "counts": dict(counts),
        "labels": dict(labels.most_common()),
        "unique_images": len({row["image"] for rows in split_rows.values() for row in rows}),
        "notes": [
            "Each target mask is mask.png == mask.txt[label].",
            "Ambiguous and missing label matches are rejected.",
            "Official SLAKE train/validation/test membership is preserved.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
