import argparse
import json
import re
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prepare_slake_official_bbox import aliases_for, contains_term


PREFERRED_LABELS = [
    "Liver",
    "Spleen",
    "Brain Edema",
    "Colon",
    "Spinal Cord",
    "Left Kidney",
]


def load_mask_map(path):
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        value, label = line.split(":", 1)
        mapping[label.strip()] = int(value.strip())
    return mapping


def load_rows(data_root):
    rows = []
    for split in ("train", "validation", "test"):
        path = data_root / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["split"] = split
                rows.append(row)
    return rows


def load_source_index(slake_root):
    index = {}
    for split in ("train", "validation", "test"):
        rows = json.loads((slake_root / f"{split}.json").read_text(encoding="utf-8"))
        for row in rows:
            index[(split, row.get("qid"))] = row
    return index


def load_discrete_mask(path, valid_values):
    image = Image.open(path)
    array = np.array(image)
    if array.ndim == 3:
        rgb = array[..., :3]
        if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])):
            return None
        array = rgb[..., 0]
    unique = {int(value) for value in np.unique(array)}
    if not unique.issubset(valid_values | {0}):
        return None
    return array


def get_font(size):
    for path in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def safe_name(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def select_examples(rows, source_index, slake_root, mask_map):
    selected = []
    used_cases = set()
    valid_values = set(mask_map.values())
    for label in PREFERRED_LABELS:
        if label not in mask_map:
            continue
        candidates = []
        value = mask_map[label]
        for row in rows:
            if row["region"] != label:
                continue
            source = source_index.get((row["split"], row.get("qid")))
            if source is None:
                continue
            case_dir = slake_root / "imgs" / "imgs" / Path(source["img_name"]).parent
            if case_dir.name in used_cases:
                continue
            mask_array = load_discrete_mask(case_dir / "mask.png", valid_values)
            if mask_array is None:
                continue
            binary = mask_array == value
            area_ratio = float(binary.mean())
            if area_ratio <= 0:
                continue
            # Prefer masks that are easy to see while avoiding nearly full images.
            target_ratio = 0.08 if label in {"Brain Edema", "Spinal Cord"} else 0.16
            explicit_in_question = any(
                contains_term(row["question"], alias) for alias in aliases_for(label)
            )
            candidates.append(
                (
                    0 if explicit_in_question else 1,
                    abs(area_ratio - target_ratio),
                    row,
                    source,
                    binary,
                )
            )
        if candidates:
            _, _, row, source, binary = min(candidates, key=lambda item: item[:2])
            selected.append((row, source, binary, mask_map[label]))
            used_cases.add(Path(source["img_name"]).parent.name)
    return selected


def add_overlay(source, binary, color=(255, 30, 30), alpha=0.48):
    base = np.array(source.convert("RGB"), dtype=np.float32)
    if binary.shape != base.shape[:2]:
        binary = np.array(
            Image.fromarray(binary.astype(np.uint8) * 255).resize(
                source.size, Image.Resampling.NEAREST
            )
        ) > 0
    overlay = base.copy()
    overlay[binary] = (1 - alpha) * overlay[binary] + alpha * np.array(color)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)), binary


def fit_image(image, size):
    canvas = Image.new("RGB", size, "black")
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def render_example(row, source, binary, mask_value, slake_root, output_dir):
    source_path = slake_root / "imgs" / "imgs" / source["img_name"]
    image = Image.open(source_path).convert("RGB")
    overlay, binary = add_overlay(image, binary)
    binary_image = Image.fromarray(binary.astype(np.uint8) * 255).convert("RGB")

    panel_width = 360
    image_height = 330
    header_height = 125
    panel = Image.new("RGB", (panel_width * 3, header_height + image_height), "white")
    draw = ImageDraw.Draw(panel)
    title = f"{row['id']} | target={row['region']} | mask value={mask_value}"
    draw.text((12, 8), title, fill="black", font=get_font(23))
    question = "Question: " + row["question"]
    for index, line in enumerate(textwrap.wrap(question, width=100)[:2]):
        draw.text((12, 40 + index * 24), line, fill=(30, 30, 30), font=get_font(19))
    draw.text((12, 91), f"Answer: {row['answer']}", fill=(30, 30, 30), font=get_font(19))

    views = [
        ("Original", image),
        (f"Question-specific {row['region']} mask", overlay),
        ("Binary target mask", binary_image),
    ]
    for index, (label, view) in enumerate(views):
        fitted = fit_image(view, (panel_width, image_height - 30))
        panel.paste(fitted, (index * panel_width, header_height + 30))
        draw.text((index * panel_width + 10, header_height + 3), label, fill="black", font=get_font(20))

    name = f"{row['id']}_{safe_name(row['region'])}.jpg"
    panel.save(output_dir / name, quality=95)
    return panel, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slake-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/segmentation_question_examples"),
    )
    args = parser.parse_args()

    slake_root = args.slake_root
    data_root = args.data_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_map = load_mask_map(slake_root / "mask.txt")
    rows = load_rows(data_root)
    source_index = load_source_index(slake_root)
    examples = select_examples(rows, source_index, slake_root, mask_map)

    rendered = []
    manifest = []
    for row, source, binary, value in examples:
        panel, filename = render_example(
            row, source, binary, value, slake_root, output_dir
        )
        rendered.append(panel)
        manifest.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "target_region": row["region"],
                "mask_value": value,
                "source_image": source["img_name"],
                "output": filename,
            }
        )

    columns = 1
    width = max(panel.width for panel in rendered)
    height = sum(panel.height for panel in rendered)
    sheet = Image.new("RGB", (width, height), (225, 225, 225))
    y = 0
    for panel in rendered:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet_path = output_dir / "segmentation_question_examples_contact_sheet.jpg"
    sheet.save(sheet_path, quality=94)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"count": len(manifest), "contact_sheet": str(sheet_path), "examples": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
