from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:
    A = None


@dataclass(frozen=True)
class DataConfig:
    image_dir: str
    annotation_path: str
    train: bool
    image_size: int = 512
    augment: bool = False


def build_augmentations(image_size: int, train: bool):
    if A is None:
        raise RuntimeError("albumentations is required when augmentations are enabled")

    transforms = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=0, value=(0, 0, 0)),
    ]

    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.HueSaturationValue(p=0.4),
                A.Rotate(limit=8, p=0.3),
                A.RandomScale(scale_limit=0.15, p=0.3),
                A.GaussNoise(p=0.2),
                A.GaussianBlur(p=0.2),
            ]
        )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels", "iscrowd"],
            min_area=1,
            min_visibility=0.1,
        ),
    )


class CocoDetectionSubset(Dataset):
    def __init__(self, config: DataConfig) -> None:
        self.coco = COCO(config.annotation_path)
        self.image_dir = config.image_dir
        self.image_size = config.image_size
        self.train = config.train
        self.transform = build_augmentations(config.image_size, config.train) if config.augment else None

        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_contiguous = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        self.contiguous_to_cat_id = {idx: cat_id for idx, cat_id in enumerate(cat_ids)}
        cats = self.coco.loadCats(cat_ids)
        self.class_names = [cat["name"] for cat in cats]
        self.image_ids = self._filter_images(cat_ids)

    def _filter_images(self, cat_ids: List[int]) -> List[int]:
        valid = []
        for img_id in sorted(self.coco.getImgIds()):
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            if any(ann.get("bbox") for ann in anns):
                valid.append(img_id)
        return valid

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = f"{self.image_dir}/{image_info['file_name']}"
        image = Image.open(image_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        iscrowd = []

        for ann in anns:
            cat_id = ann.get("category_id")
            if cat_id not in self.cat_id_to_contiguous:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_contiguous[cat_id])
            iscrowd.append(ann.get("iscrowd", 0))

        image_np = np.array(image)
        if self.transform is not None:
            out = self.transform(image=image_np, bboxes=boxes, labels=labels, iscrowd=iscrowd)
            image_np = out["image"]
            boxes = list(out["bboxes"])
            labels = list(out["labels"])
            iscrowd = list(out["iscrowd"])

        image = Image.fromarray(image_np)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            areas_tensor = torch.tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        annotations = []
        for box, label, area, crowd in zip(boxes_tensor.tolist(), labels_tensor.tolist(), areas_tensor.tolist(), iscrowd_tensor.tolist(), strict=False):
            x1, y1, x2, y2 = box
            annotations.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": float(area),
                    "iscrowd": int(crowd),
                }
            )

        target = {
            "image_id": int(image_id),
            "annotations": annotations,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        return image, target
