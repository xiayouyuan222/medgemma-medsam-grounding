import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_rows(rows, count, seed):
    rng = random.Random(seed)
    by_region = defaultdict(list)
    for row in rows:
        by_region[row["region"]].append(row)
    for values in by_region.values():
        rng.shuffle(values)

    selected = []
    regions = sorted(by_region)
    while len(selected) < count and regions:
        next_regions = []
        for region in regions:
            if by_region[region] and len(selected) < count:
                selected.append(by_region[region].pop())
            if by_region[region]:
                next_regions.append(region)
        regions = next_regions
    return selected


def overlay_sample(data_root, row, tile_size):
    image = Image.open(data_root / row["image"]).convert("RGB")
    mask = np.asarray(Image.open(data_root / row["mask"]).convert("L")) > 0
    pixels = np.asarray(image).copy()
    red = np.zeros_like(pixels)
    red[..., 0] = 255
    pixels[mask] = np.rint(0.55 * pixels[mask] + 0.45 * red[mask]).astype(np.uint8)
    overlay = Image.fromarray(pixels)
    overlay.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
    if overlay.width < tile_size or overlay.height < tile_size:
        scale = min(tile_size / overlay.width, tile_size / overlay.height)
        size = (
            max(1, int(round(overlay.width * scale))),
            max(1, int(round(overlay.height * scale))),
        )
        overlay = overlay.resize(size, Image.Resampling.NEAREST)

    canvas = Image.new("RGB", (tile_size, tile_size + 48), "white")
    x = (tile_size - overlay.width) // 2
    y = (tile_size - overlay.height) // 2
    canvas.paste(overlay, (x, y))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = f'{row["region"]} | {row["patient_id"]} | z={row["slice_index"]}'
    detail = f'mask_pixels={row["mask_pixels"]}'
    draw.text((6, tile_size + 6), title, fill="black", font=font)
    draw.text((6, tile_size + 24), detail, fill="black", font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--jsonl", default="train.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=28)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows = load_rows(data_root / args.jsonl)
    selected = select_rows(rows, args.count, args.seed)
    tiles = [overlay_sample(data_root, row, args.tile_size) for row in selected]

    columns = max(1, args.columns)
    rows_count = math.ceil(len(tiles) / columns)
    tile_height = args.tile_size + 48
    sheet = Image.new(
        "RGB", (columns * args.tile_size, rows_count * tile_height), "#eeeeee"
    )
    for index, tile in enumerate(tiles):
        x = (index % columns) * args.tile_size
        y = (index // columns) * tile_height
        sheet.paste(tile, (x, y))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"Saved {len(tiles)} samples to {output}")


if __name__ == "__main__":
    main()
