import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT_DIR = Path("figures")
FONT_DIR = Path(r"C:\Windows\Fonts")


def font(size, bold=False):
    path = FONT_DIR / ("timesbd.ttf" if bold else "times.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def save_region_figure():
    data = [
        ("Colon", 0.280, 0.406), ("Duodenum", 0.322, 0.436),
        ("Small Bowel", 0.387, 0.526), ("Bladder", 0.436, 0.525),
        ("Spinal Cord", 0.523, 0.648), ("Stomach", 0.556, 0.652),
        ("Esophagus", 0.592, 0.696), ("Heart", 0.684, 0.778),
        ("Left Lung", 0.698, 0.776), ("Right Lung", 0.716, 0.793),
        ("Liver", 0.840, 0.887), ("Left Kidney", 0.860, 0.911),
        ("Spleen", 0.864, 0.911), ("Right Kidney", 0.874, 0.918),
    ]
    image = Image.new("RGB", (1800, 1280), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 390, 120, 1710, 1130
    draw.text(((left + right) / 2, 28), "TotalSeg-14 Held-out Test Performance by Region", fill="#17202a", font=font(34, True), anchor="ma")
    for i in range(6):
        value = i / 5
        x = left + value * (right - left)
        draw.line((x, top, x, bottom), fill="#d5d8dc", width=2)
        draw.text((x, bottom + 18), f"{value:.1f}", fill="#303030", font=font(22), anchor="ma")
    draw.line((left, top, left, bottom), fill="#4d5656", width=3)
    draw.line((left, bottom, right, bottom), fill="#4d5656", width=3)
    draw.text(((left + right) / 2, bottom + 62), "Score", fill="#303030", font=font(24), anchor="ma")
    row_h = (bottom - top) / len(data)
    for idx, (name, iou, dice) in enumerate(data):
        center = top + (idx + 0.5) * row_h
        draw.text((left - 22, center), name, fill="#202020", font=font(23), anchor="rm")
        draw.rectangle((left, center - 22, left + iou * (right - left), center - 3), fill="#1f4e79")
        draw.rectangle((left, center + 4, left + dice * (right - left), center + 23), fill="#70ad47")
    draw.rectangle((1260, 94, 1294, 112), fill="#1f4e79")
    draw.text((1307, 103), "Mean IoU", fill="#202020", font=font(21), anchor="lm")
    draw.rectangle((1460, 94, 1494, 112), fill="#70ad47")
    draw.text((1507, 103), "Mean Dice", fill="#202020", font=font(21), anchor="lm")
    image.save(OUTPUT_DIR / "totalseg14_per_region_performance.png", dpi=(300, 300))


def save_transfer_figure():
    metrics = ["Mean IoU", "Mean Dice", "IoU@0.5"]
    slake_only = [0.526, 0.610, 0.543]
    transferred = [0.657, 0.748, 0.744]
    image = Image.new("RGB", (1600, 950), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 180, 150, 1510, 760
    draw.text(((left + right) / 2, 38), "Matched SLAKE Held-out Test Comparison", fill="#17202a", font=font(36, True), anchor="ma")
    for i in range(6):
        value = i * 0.16
        y = bottom - (value / 0.8) * (bottom - top)
        draw.line((left, y, right, y), fill="#d5d8dc", width=2)
        draw.text((left - 20, y), f"{value:.2f}", fill="#303030", font=font(22), anchor="rm")
    draw.line((left, top, left, bottom), fill="#4d5656", width=3)
    draw.line((left, bottom, right, bottom), fill="#4d5656", width=3)
    group_w = (right - left) / len(metrics)
    for idx, label in enumerate(metrics):
        cx = left + (idx + 0.5) * group_w
        for value, offset, colour in ((slake_only[idx], -70, "#7f8c8d"), (transferred[idx], 70, "#1f4e79")):
            x0, x1 = cx + offset - 58, cx + offset + 58
            y0 = bottom - (value / 0.8) * (bottom - top)
            draw.rectangle((x0, y0, x1, bottom), fill=colour)
            draw.text(((x0 + x1) / 2, y0 - 10), f"{value:.3f}", fill="#202020", font=font(22), anchor="mb")
        draw.text((cx, bottom + 28), label, fill="#202020", font=font(24), anchor="ma")
    draw.rectangle((825, 105, 859, 124), fill="#7f8c8d")
    draw.text((872, 115), "SLAKE-only aligner", fill="#202020", font=font(21), anchor="lm")
    draw.rectangle((1160, 105, 1194, 124), fill="#1f4e79")
    draw.text((1207, 115), "TotalSeg-14 to SLAKE", fill="#202020", font=font(21), anchor="lm")
    draw.text((62, (top + bottom) / 2), "Score", fill="#303030", font=font(25), anchor="mm")
    image.save(OUTPUT_DIR / "slake_transfer_comparison.png", dpi=(300, 300))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_region_figure()
    save_transfer_figure()
    print(f"Created Chapter 4 figures in {OUTPUT_DIR}")
