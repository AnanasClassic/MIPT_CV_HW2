import argparse
import os

from PIL import ImageDraw
import torch
from torchvision.ops import box_iou
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from src.data_utils import CocoDetectionSubset, DataConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DETR errors.")
    parser.add_argument("--data-dir", default="data/coco_subset")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="artifacts/error_analysis")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def draw_example(image, gt_boxes, gt_labels, pred_box, pred_label, pred_score, class_names):
    draw = ImageDraw.Draw(image)
    for box, label in zip(gt_boxes, gt_labels, strict=False):
        x1, y1, x2, y2 = box
        name = class_names[label] if 0 <= label < len(class_names) else str(label)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"gt:{name}", fill="red")
    x1, y1, x2, y2 = pred_box
    pred_name = class_names[pred_label] if 0 <= pred_label < len(class_names) else str(pred_label)
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
    draw.text((x1, y1), f"pred:{pred_name}:{pred_score:.2f}", fill="lime")
    return image


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

    cls_dir = os.path.join(args.output_dir, "classification")
    loc_dir = os.path.join(args.output_dir, "localization")
    os.makedirs(cls_dir, exist_ok=True)
    os.makedirs(loc_dir, exist_ok=True)

    cls_saved = 0
    loc_saved = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            if cls_saved >= args.count and loc_saved >= args.count:
                break

            image, target = dataset[idx]
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

            boxes = result["boxes"].detach().cpu()
            scores = result["scores"].detach().cpu()
            labels = result["labels"].detach().cpu()

            gt_boxes = target["boxes"].detach().cpu()
            gt_labels = target["labels"].detach().cpu()
            if gt_boxes.numel() == 0:
                continue

            ious = box_iou(boxes, gt_boxes)
            max_ious, gt_indices = ious.max(dim=1)

            for pred_idx in range(boxes.size(0)):
                iou = max_ious[pred_idx].item()
                gt_idx = gt_indices[pred_idx].item()
                pred_label = int(labels[pred_idx].item())
                gt_label = int(gt_labels[gt_idx].item())

                if iou >= args.iou_threshold and pred_label != gt_label and cls_saved < args.count:
                    example = draw_example(
                        image.copy(),
                        gt_boxes.tolist(),
                        gt_labels.tolist(),
                        boxes[pred_idx].tolist(),
                        pred_label,
                        float(scores[pred_idx].item()),
                        dataset.class_names,
                    )
                    example.save(os.path.join(cls_dir, f"cls_{idx}_{pred_idx}.jpg"))
                    cls_saved += 1

                if iou < args.iou_threshold and pred_label == gt_label and loc_saved < args.count:
                    example = draw_example(
                        image.copy(),
                        gt_boxes.tolist(),
                        gt_labels.tolist(),
                        boxes[pred_idx].tolist(),
                        pred_label,
                        float(scores[pred_idx].item()),
                        dataset.class_names,
                    )
                    example.save(os.path.join(loc_dir, f"loc_{idx}_{pred_idx}.jpg"))
                    loc_saved += 1

                if cls_saved >= args.count and loc_saved >= args.count:
                    break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
