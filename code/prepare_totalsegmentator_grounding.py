import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


REGION_FILES = {
    "Liver": ["liver.nii.gz"],
    "Spleen": ["spleen.nii.gz"],
    "Left Kidney": ["kidney_left.nii.gz"],
    "Right Kidney": ["kidney_right.nii.gz"],
    "Left Lung": ["lung_upper_lobe_left.nii.gz", "lung_lower_lobe_left.nii.gz"],
    "Right Lung": [
        "lung_upper_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_lower_lobe_right.nii.gz",
    ],
    "Heart": ["heart.nii.gz"],
    "Stomach": ["stomach.nii.gz"],
    "Esophagus": ["esophagus.nii.gz"],
    "Duodenum": ["duodenum.nii.gz"],
    "Small Bowel": ["small_bowel.nii.gz"],
    "Colon": ["colon.nii.gz"],
    "Bladder": ["urinary_bladder.nii.gz"],
    "Spinal Cord": ["spinal_cord.nii.gz"],
}


def slug(text):
    return text.lower().replace(" ", "_")


def region_name_from_filename(filename):
    """Convert a TotalSegmentator mask filename into a readable phrase."""
    stem = filename
    if stem.endswith(".nii.gz"):
        stem = stem[: -len(".nii.gz")]
    tokens = stem.split("_")
    if tokens and tokens[-1] in {"left", "right"}:
        tokens = [tokens[-1], *tokens[:-1]]
    return " ".join(token.upper() if len(token) == 1 else token.title() for token in tokens)


def discover_region_files(ct_paths):
    filenames = {
        path.name
        for ct_path in ct_paths
        for path in (ct_path.parent / "segmentations").glob("*.nii.gz")
    }
    if not filenames:
        raise FileNotFoundError("No segmentation NIfTI files found for selected subjects.")
    region_files = {}
    for filename in sorted(filenames):
        region = region_name_from_filename(filename)
        if region in region_files:
            raise ValueError(
                f"Duplicate readable region name {region!r} for {filename!r} and "
                f"{region_files[region][0]!r}."
            )
        region_files[region] = [filename]
    return region_files


def assign_split(subject_id, seed, train_ratio, validation_ratio):
    digest = hashlib.sha256(f"{seed}|{subject_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def window_ct(array, center, width):
    low = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(array, low, high)
    return np.rint((clipped - low) * 255.0 / (high - low)).astype(np.uint8)


def make_rgb_image(ct, index, center, width, image_mode):
    if image_mode == "2.5d":
        indices = [max(0, index - 1), index, min(ct.shape[2] - 1, index + 1)]
        channels = [window_ct(ct[:, :, z], center, width) for z in indices]
    else:
        center_slice = window_ct(ct[:, :, index], center, width)
        channels = [center_slice, center_slice, center_slice]
    # Canonical NIfTI arrays are indexed as (x, y, z). Rotate the axial plane
    # into the conventional display orientation used by medical image viewers.
    return np.rot90(np.stack(channels, axis=-1), k=1, axes=(0, 1))


def evenly_spaced(indices, limit):
    if limit <= 0 or len(indices) <= limit:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=limit)
    return [indices[int(round(position))] for position in positions]


def load_canonical(path, dtype):
    image = nib.as_closest_canonical(nib.load(str(path)))
    return np.asarray(image.dataobj, dtype=dtype)


def load_region_mask(segmentation_dir, filenames, expected_shape):
    combined = np.zeros(expected_shape, dtype=bool)
    found = []
    for filename in filenames:
        path = segmentation_dir / filename
        if not path.exists():
            continue
        mask = load_canonical(path, np.uint8) > 0
        if mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape mismatch for {path}: {mask.shape} != {expected_shape}"
            )
        combined |= mask
        found.append(filename)
    return combined, found


def make_messages(region, image_path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": (
                        f'Ground the medical phrase "{region}" in this image and '
                        f"return Relevant region: <p>{region}</p>."
                    ),
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"Relevant region: <p>{region}</p>"}],
        },
    ]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TotalSegmentator CT volumes into 2.5D phrase-mask grounding data."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-subjects", type=int, default=20)
    parser.add_argument("--max-slices-per-region", type=int, default=12)
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--window-center", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=400.0)
    parser.add_argument(
        "--image-mode",
        choices=["grayscale", "2.5d"],
        default="grayscale",
        help=(
            "grayscale replicates the central CT slice over RGB channels; "
            "2.5d uses the previous, central and next slices as RGB."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument(
        "--regions",
        default=",".join(REGION_FILES),
        help="Comma-separated canonical region names.",
    )
    parser.add_argument(
        "--all-regions",
        action="store_true",
        help=(
            "Use every original TotalSegmentator mask file as an independent "
            "class. This disables the manually merged canonical region list."
        ),
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.validation_ratio <= 0:
        raise ValueError("Train and validation ratios must be positive.")
    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("Train and validation ratios must sum to less than one.")

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "masks").mkdir(exist_ok=True)

    ct_paths = sorted(input_root.rglob("ct.nii.gz"))
    # Select a deterministic but non-contiguous subject subset. Subject IDs can
    # be ordered by acquisition source, so taking the first N may introduce bias.
    ct_paths.sort(
        key=lambda path: hashlib.sha256(
            f"selection|{args.split_seed}|{path.parent.name}".encode("utf-8")
        ).digest()
    )
    if args.max_subjects > 0:
        ct_paths = ct_paths[: args.max_subjects]
    if not ct_paths:
        raise FileNotFoundError(f"No ct.nii.gz files found below {input_root}")

    if args.all_regions:
        region_files = discover_region_files(ct_paths)
        requested_regions = sorted(region_files)
        print(
            f"Discovered {len(requested_regions)} original segmentation classes.",
            flush=True,
        )
    else:
        region_files = REGION_FILES
        requested_regions = [
            value.strip() for value in args.regions.split(",") if value.strip()
        ]
        unknown = sorted(set(requested_regions) - set(region_files))
        if unknown:
            raise ValueError(
                f"Unknown regions: {unknown}. Available: {sorted(region_files)}"
            )

    rows_by_split = {"train": [], "validation": [], "test": []}
    subjects_by_split = {"train": set(), "validation": set(), "test": set()}
    label_counts = Counter()
    missing_files = Counter()
    image_cache = set()
    rejected = Counter()

    for subject_number, ct_path in enumerate(ct_paths, start=1):
        subject_dir = ct_path.parent
        subject_id = subject_dir.name
        segmentation_dir = subject_dir / "segmentations"
        split = assign_split(
            subject_id, args.split_seed, args.train_ratio, args.validation_ratio
        )
        subjects_by_split[split].add(subject_id)
        print(f"[{subject_number}/{len(ct_paths)}] {subject_id} -> {split}", flush=True)

        ct = load_canonical(ct_path, np.float32)
        if ct.ndim != 3:
            rejected["non_3d_ct"] += 1
            continue

        for region in requested_regions:
            mask, found_files = load_region_mask(
                segmentation_dir, region_files[region], ct.shape
            )
            if not found_files:
                missing_files[region] += 1
                continue

            areas = mask.sum(axis=(0, 1))
            valid_slices = np.flatnonzero(areas >= args.min_mask_pixels).tolist()
            valid_slices = evenly_spaced(valid_slices, args.max_slices_per_region)
            if not valid_slices:
                rejected["no_valid_slices"] += 1
                continue

            for slice_index in valid_slices:
                image_rel = f"images/{split}_{subject_id}_z{slice_index:04d}.png"
                mask_rel = (
                    f"masks/{split}_{subject_id}_z{slice_index:04d}_{slug(region)}.png"
                )
                image_path = output_dir / image_rel
                mask_path = output_dir / mask_rel
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)

                image_key = (subject_id, slice_index)
                if image_key not in image_cache:
                    image = make_rgb_image(
                        ct,
                        slice_index,
                        args.window_center,
                        args.window_width,
                        args.image_mode,
                    )
                    Image.fromarray(image, mode="RGB").save(image_path)
                    image_cache.add(image_key)

                binary = np.rot90(mask[:, :, slice_index], k=1, axes=(0, 1))
                Image.fromarray(binary.astype(np.uint8) * 255, mode="L").save(mask_path)
                ys, xs = np.nonzero(binary)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                height, width = binary.shape
                item_id = f"{split}_{subject_id}_z{slice_index:04d}_{slug(region)}"
                row = {
                    "id": item_id,
                    "image": image_rel,
                    "mask": mask_rel,
                    "question": f"Ground the {region} in this image.",
                    "region": region,
                    "regions": [region],
                    "answer": region,
                    "answer_type": "GROUNDING",
                    "content_type": "Mask",
                    "grounding_type": "single_region_pretraining",
                    "source": "TotalSegmentator-v2.0.1",
                    "patient_id": subject_id,
                    "slice_index": int(slice_index),
                    "modality": "CT",
                    "mask_pixels": int(binary.sum()),
                    "image_size": [width, height],
                    "bbox_from_mask_xyxy": bbox,
                    "source_mask_files": found_files,
                    "messages": make_messages(region, image_rel),
                }
                rows_by_split[split].append(row)
                label_counts[(split, region)] += 1

        del ct

    for split, rows in rows_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)

    overlaps = {
        "train_validation": sorted(subjects_by_split["train"] & subjects_by_split["validation"]),
        "train_test": sorted(subjects_by_split["train"] & subjects_by_split["test"]),
        "validation_test": sorted(subjects_by_split["validation"] & subjects_by_split["test"]),
    }
    summary = {
        "source": "TotalSegmentator-v2.0.1",
        "input_root": str(input_root.resolve()),
        "requested_subjects": len(ct_paths),
        "regions": requested_regions,
        "region_files": {
            region: region_files[region] for region in requested_regions
        },
        "configuration": {
            "max_slices_per_region": args.max_slices_per_region,
            "min_mask_pixels": args.min_mask_pixels,
            "window_center": args.window_center,
            "window_width": args.window_width,
            "image_mode": args.image_mode,
            "split_seed": args.split_seed,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "all_original_regions": args.all_regions,
        },
        "samples": {split: len(rows) for split, rows in rows_by_split.items()},
        "subjects": {split: len(values) for split, values in subjects_by_split.items()},
        "unique_rendered_images": len(image_cache),
        "label_counts": {
            split: {
                region: label_counts[(split, region)]
                for region in requested_regions
            }
            for split in rows_by_split
        },
        "missing_region_files_by_subject": dict(missing_files),
        "rejected": dict(rejected),
        "patient_overlap": overlaps,
        "patient_disjoint": not any(overlaps.values()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
