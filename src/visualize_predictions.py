import argparse
import os
import random

from PIL import ImageDraw
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from src.data_utils import CocoDetectionSubset, DataConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DETR predictions.")
    parser.add_argument("--data-dir", default="data/coco_subset")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/visualizations")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def draw_predictions(image, boxes, labels, scores, class_names):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores, strict=False):
        x1, y1, x2, y2 = box
        name = class_names[label] if 0 <= label < len(class_names) else str(label)
        text = f"{name}:{score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        draw.text((x1, y1), text, fill="lime")
    return image


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

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

    os.makedirs(args.output_dir, exist_ok=True)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    picked = indices[: args.count]

    saved = 0
    with torch.no_grad():
        for idx in picked:
            image, _ = dataset[idx]
            encoding = processor(images=image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].to(device)
            pixel_mask = encoding["pixel_mask"].to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            target_sizes = torch.tensor([image.size[::-1]], device=device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=args.score_threshold
            )
            result = results[0]
            if result["boxes"].numel() == 0:
                continue

            boxes = result["boxes"].detach().cpu().tolist()
            scores = result["scores"].detach().cpu().tolist()
            labels = result["labels"].detach().cpu().tolist()

            image = draw_predictions(image, boxes, labels, scores, dataset.class_names)
            image.save(os.path.join(args.output_dir, f"pred_{idx}.jpg"))
            saved += 1
            if saved >= args.count:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
