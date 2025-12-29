import argparse
import csv
import os

import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from src.data_utils import CocoDetectionSubset, DataConfig
from src.utils import seed_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DETR on a COCO subset.")
    parser.add_argument("--data-dir", default="data/coco_subset")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def append_metrics(path: str, row: dict) -> None:
    fieldnames = ["model", "split", "mAP", "mAP50", "checkpoint"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def simple_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main() -> int:
    args = parse_args()
    ann_path = os.path.join(args.data_dir, "annotations", f"instances_{args.split}.json")
    image_dir = os.path.join(args.data_dir, "images", args.split)
    config = DataConfig(
        image_dir=image_dir,
        annotation_path=ann_path,
        train=False,
        image_size=args.image_size,
        augment=True,
    )
    dataset = CocoDetectionSubset(config)

    generator = torch.Generator()
    generator.manual_seed(0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=simple_collate,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": args.image_size, "longest_edge": args.image_size},
    )
    model = AutoModelForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=len(dataset.class_names),
        ignore_mismatched_sizes=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for images, targets in loader:
            encoding = processor(images=images, return_tensors="pt")
            pixel_values = encoding["pixel_values"].to(device)
            pixel_mask = encoding["pixel_mask"].to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            target_sizes = torch.tensor([image.size[::-1] for image in images], device=device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=args.score_threshold
            )

            for result, target in zip(results, targets, strict=False):
                image_id = int(target["image_id"])
                boxes = result["boxes"].detach().cpu()
                scores = result["scores"].detach().cpu()
                labels = result["labels"].detach().cpu()
                for box, score, label in zip(boxes, scores, labels, strict=False):
                    x1, y1, x2, y2 = box.tolist()
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    category_id = dataset.contiguous_to_cat_id[int(label)]
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "score": score.item(),
                        }
                    )

    if not predictions:
        raise RuntimeError("No predictions to evaluate. Lower --score-threshold.")

    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    row = {
        "model": "detr",
        "split": args.split,
        "mAP": f"{coco_eval.stats[0]:.6f}",
        "mAP50": f"{coco_eval.stats[1]:.6f}",
        "checkpoint": args.checkpoint,
    }
    append_metrics(metrics_path, row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
