import argparse
import json
import os
import random
import shutil
from collections import Counter

from pycocotools.coco import COCO

from src.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a COCO subset for detection.")
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--output-dir", default="data/coco_subset")
    parser.add_argument("--splits", default="train2017,val2017")
    parser.add_argument("--classes", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_coco(root: str, split: str) -> COCO:
    ann_path = os.path.join(root, "annotations", f"instances_{split}.json")
    return COCO(ann_path)


def select_top_classes(coco: COCO, top_k: int) -> list[int]:
    counts = Counter()
    for ann in coco.dataset.get("annotations", []):
        counts[ann["category_id"]] += 1
    most_common = [cat_id for cat_id, _ in counts.most_common(top_k)]
    return most_common


def resolve_classes(coco: COCO, names: list[str] | None, top_k: int | None) -> list[int]:
    if names:
        return coco.getCatIds(catNms=names)
    if top_k:
        return select_top_classes(coco, top_k)
    raise ValueError("Provide --classes or --top-k.")


def select_image_ids(
    coco: COCO,
    class_ids: list[int],
    max_images_per_class: int | None,
    seed: int,
) -> list[int]:
    if max_images_per_class is None:
        return sorted(coco.getImgIds(catIds=class_ids))
    rng = random.Random(seed)
    selected: set[int] = set()
    for cat_id in class_ids:
        ids = coco.getImgIds(catIds=[cat_id])
        rng.shuffle(ids)
        selected.update(ids[:max_images_per_class])
    return sorted(selected)


def build_subset(
    coco: COCO,
    class_ids: list[int],
    max_images_per_class: int | None,
    seed: int,
) -> tuple[dict, dict]:
    class_ids = sorted(class_ids)
    cats = coco.loadCats(class_ids)
    id_map = {cat_id: idx + 1 for idx, cat_id in enumerate(class_ids)}
    categories = [
        {"id": id_map[cat["id"]], "name": cat["name"], "supercategory": cat.get("supercategory", "")}
        for cat in cats
    ]

    images = []
    annotations = []
    ann_id = 1

    image_ids = select_image_ids(coco, class_ids, max_images_per_class, seed)
    for img_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=class_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue
        img_info = coco.loadImgs(img_id)[0]
        images.append(img_info)
        for ann in anns:
            if "bbox" not in ann:
                continue
            new_ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": id_map[ann["category_id"]],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
            }
            if "segmentation" in ann:
                new_ann["segmentation"] = ann["segmentation"]
            annotations.append(new_ann)
            ann_id += 1

    subset = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    meta = {
        "original_category_ids": class_ids,
        "category_id_map": id_map,
        "class_names": [cat["name"] for cat in cats],
        "max_images_per_class": max_images_per_class,
    }
    return subset, meta


def materialize_images(
    coco_root: str,
    split: str,
    output_dir: str,
    images: list[dict],
    copy_images: bool,
) -> None:
    src_dir = os.path.abspath(os.path.join(coco_root, "images", split))
    dst_dir = os.path.join(output_dir, "images", split)
    os.makedirs(dst_dir, exist_ok=True)
    for img in images:
        src = os.path.join(src_dir, img["file_name"])
        dst = os.path.join(dst_dir, img["file_name"])
        if os.path.lexists(dst):
            if os.path.islink(dst) and not os.path.exists(dst):
                os.unlink(dst)
            else:
                continue
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing source image: {src}")
        if copy_images:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    class_names = None
    if args.classes:
        class_names = [name.strip() for name in args.classes.split(",") if name.strip()]

    os.makedirs(args.output_dir, exist_ok=True)
    ann_dir = os.path.join(args.output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    base_coco = load_coco(args.coco_root, splits[0])
    class_ids = resolve_classes(base_coco, class_names, args.top_k)

    meta_path = os.path.join(args.output_dir, "meta.json")
    for split in splits:
        coco = load_coco(args.coco_root, split)
        subset, meta = build_subset(coco, class_ids, args.max_images_per_class, args.seed)
        ann_path = os.path.join(ann_dir, f"instances_{split}.json")
        save_json(ann_path, subset)
        materialize_images(args.coco_root, split, args.output_dir, subset["images"], args.copy_images)
        save_json(meta_path, meta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
