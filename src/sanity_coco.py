import argparse
import csv
import os
import random
from collections import Counter

from PIL import ImageDraw

from src.data_utils import CocoDetectionSubset, DataConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO subset sanity checks.")
    parser.add_argument("--data-dir", default="data/coco_subset")
    parser.add_argument("--splits", default="train2017,val2017")
    parser.add_argument("--output-dir", default="artifacts/data_sanity")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def save_counts(path: str, class_names: list[str], counts: Counter) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "count"])
        for idx, name in enumerate(class_names):
            writer.writerow([name, counts.get(idx, 0)])


def draw_boxes(image, target, class_names: list[str]):
    draw = ImageDraw.Draw(image)
    boxes = target["boxes"].tolist()
    labels = target["labels"].tolist()
    for box, label in zip(boxes, labels, strict=False):
        x1, y1, x2, y2 = box
        name = class_names[label] if 0 <= label < len(class_names) else str(label)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), name, fill="red")
    return image


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    for split in splits:
        ann_path = os.path.join(args.data_dir, "annotations", f"instances_{split}.json")
        image_dir = os.path.join(args.data_dir, "images", split)
        config = DataConfig(
            image_dir=image_dir,
            annotation_path=ann_path,
            train=False,
            image_size=args.image_size,
            augment=True,
        )
        dataset = CocoDetectionSubset(config)

        counts = Counter()
        for ann in dataset.coco.dataset.get("annotations", []):
            cat_id = ann.get("category_id")
            if cat_id in dataset.cat_id_to_contiguous:
                counts[dataset.cat_id_to_contiguous[cat_id]] += 1

        counts_path = os.path.join(args.output_dir, f"{split}_counts.csv")
        save_counts(counts_path, dataset.class_names, counts)

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        sample_indices = indices[: args.sample_count]
        sample_dir = os.path.join(args.output_dir, split)
        os.makedirs(sample_dir, exist_ok=True)

        for idx, dataset_idx in enumerate(sample_indices):
            image, target = dataset[dataset_idx]
            image = draw_boxes(image, target, dataset.class_names)
            image.save(os.path.join(sample_dir, f"sample_{idx}.jpg"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
