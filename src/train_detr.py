import argparse
import csv
import math
import os
from datetime import datetime

import torch
from torch import profiler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from src.data_utils import CocoDetectionSubset, DataConfig
from src.utils import save_json, seed_worker, set_deterministic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DETR on a COCO subset.")
    parser.add_argument("--data-dir", default="data/coco_subset")
    parser.add_argument("--train-split", default="train2017")
    parser.add_argument("--val-split", default="val2017")
    parser.add_argument("--output-dir", default="artifacts/detr")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=80)
    parser.add_argument("--profile-warmup", type=int, default=10)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], default="none")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    return parser.parse_args()


def make_collate_fn(processor: AutoImageProcessor):
    def collate_fn(batch):
        images, targets = zip(*batch)
        encoding = processor(images=list(images), annotations=list(targets), return_tensors="pt")
        return encoding["pixel_values"], encoding["pixel_mask"], encoding["labels"]

    return collate_fn


def build_loaders(args: argparse.Namespace, processor: AutoImageProcessor):
    train_ann = os.path.join(args.data_dir, "annotations", f"instances_{args.train_split}.json")
    val_ann = os.path.join(args.data_dir, "annotations", f"instances_{args.val_split}.json")
    train_images = os.path.join(args.data_dir, "images", args.train_split)
    val_images = os.path.join(args.data_dir, "images", args.val_split)

    train_config = DataConfig(
        image_dir=train_images,
        annotation_path=train_ann,
        train=True,
        image_size=args.image_size,
        augment=args.augment,
    )
    val_config = DataConfig(
        image_dir=val_images,
        annotation_path=val_ann,
        train=False,
        image_size=args.image_size,
        augment=True,
    )

    train_dataset = CocoDetectionSubset(train_config)
    val_dataset = CocoDetectionSubset(val_config)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    collate_fn = make_collate_fn(processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    torch.save(state, path)


def write_losses(path: str, rows: list[dict]) -> None:
    fieldnames = [
        "epoch",
        "split",
        "loss_total",
        "loss_ce",
        "loss_bbox",
        "loss_giou",
        "loss_cardinality",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def reduce_losses(loss_dict: dict, total_loss: torch.Tensor) -> dict:
    reduced = {key: value.detach().item() for key, value in loss_dict.items()}
    if "cardinality_error" in reduced and "loss_cardinality" not in reduced:
        reduced["loss_cardinality"] = reduced["cardinality_error"]
    reduced["loss_total"] = float(total_loss.detach().item())
    return reduced


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    scheduler_type: str,
    start_step: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if scheduler_type == "none" or total_steps <= 0:
        return None

    warmup_steps = max(0, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        if scheduler_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_step - 1)


def run_profile(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: str,
    steps: int,
    warmup: int,
) -> None:
    activities = [profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(profiler.ProfilerActivity.CUDA)
    schedule = profiler.schedule(wait=0, warmup=warmup, active=steps, repeat=1)
    trace_handler = profiler.tensorboard_trace_handler(output_dir)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    with profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        step = 0
        for pixel_values, pixel_mask, labels in loader:
            pixel_values = pixel_values.to(device)
            pixel_mask = pixel_mask.to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            prof.step()
            step += 1
            if step >= steps + warmup:
                break


def main() -> int:
    args = parse_args()
    set_deterministic(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": args.image_size, "longest_edge": args.image_size},
    )

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args, processor)

    id2label = {idx: name for idx, name in enumerate(train_dataset.class_names)}
    label2id = {name: idx for idx, name in id2label.items()}

    if args.no_pretrained:
        base_config = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50").config
        base_config.num_labels = len(train_dataset.class_names)
        base_config.id2label = id2label
        base_config.label2id = label2id
        model = AutoModelForObjectDetection.from_config(base_config)
    else:
        model = AutoModelForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=len(train_dataset.class_names),
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.fp16 and device.type == "cuda")

    run_name = args.run_name or f"detr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    if args.profile:
        profiler_dir = os.path.join("artifacts", "profiler", "detr")
        os.makedirs(profiler_dir, exist_ok=True)
        run_profile(model, train_loader, device, profiler_dir, args.profile_steps, args.profile_warmup)
        if args.profile_only:
            writer.close()
            return 0

    config_payload = {
        "data_dir": args.data_dir,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "num_classes": len(train_dataset.class_names),
        "classes": train_dataset.class_names,
        "model": "facebook/detr-resnet-50",
        "pretrained": not args.no_pretrained,
        "image_size": args.image_size,
        "augment": args.augment,
        "fp16": args.fp16,
        "scheduler": args.scheduler,
        "warmup_steps": args.warmup_steps,
        "warmup_ratio": args.warmup_ratio,
    }
    save_json(os.path.join(args.output_dir, "config.json"), config_payload)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

    for param_group in optimizer.param_groups:
        param_group.setdefault("initial_lr", param_group["lr"])

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_steps
    if warmup_steps <= 0 and args.warmup_ratio > 0:
        warmup_steps = int(total_steps * args.warmup_ratio)

    start_step = start_epoch * steps_per_epoch
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps, args.scheduler, start_step)

    if args.resume and scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    best_loss = float("inf")
    best_path = os.path.join(args.output_dir, "checkpoint_best.pt")
    if os.path.exists(best_path):
        best_checkpoint = torch.load(best_path, map_location="cpu")
        best_loss = float(best_checkpoint.get("val_loss", best_loss))

    losses_csv = os.path.join(args.output_dir, "losses.csv")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_totals = []
        for step, (pixel_values, pixel_mask, labels) in enumerate(train_loader, start=1):
            pixel_values = pixel_values.to(device)
            pixel_mask = pixel_mask.to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            with autocast(enabled=args.fp16 and device.type == "cuda"):
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            reduced = reduce_losses(outputs.loss_dict, loss)
            train_totals.append(reduced)

            if step % args.log_interval == 0:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar("train/loss_total", reduced["loss_total"], global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                for key, value in reduced.items():
                    if key != "loss_total":
                        writer.add_scalar(f"train/{key}", value, global_step)

        model.eval()
        val_totals = []
        with torch.no_grad():
            for pixel_values, pixel_mask, labels in val_loader:
                pixel_values = pixel_values.to(device)
                pixel_mask = pixel_mask.to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                reduced = reduce_losses(outputs.loss_dict, outputs.loss)
                val_totals.append(reduced)

        def average(records: list[dict]) -> dict:
            if not records:
                return {"loss_total": 0.0}
            keys = records[0].keys()
            return {key: sum(r[key] for r in records) / len(records) for key in keys}

        train_avg = average(train_totals)
        val_avg = average(val_totals)

        writer.add_scalar("epoch/train_loss", train_avg["loss_total"], epoch)
        writer.add_scalar("epoch/val_loss", val_avg["loss_total"], epoch)

        rows = [
            {
                "epoch": epoch,
                "split": "train",
                "loss_total": f"{train_avg.get('loss_total', 0):.6f}",
                "loss_ce": f"{train_avg.get('loss_ce', 0):.6f}",
                "loss_bbox": f"{train_avg.get('loss_bbox', 0):.6f}",
                "loss_giou": f"{train_avg.get('loss_giou', 0):.6f}",
                "loss_cardinality": f"{train_avg.get('loss_cardinality', 0):.6f}",
            },
            {
                "epoch": epoch,
                "split": "val",
                "loss_total": f"{val_avg.get('loss_total', 0):.6f}",
                "loss_ce": f"{val_avg.get('loss_ce', 0):.6f}",
                "loss_bbox": f"{val_avg.get('loss_bbox', 0):.6f}",
                "loss_giou": f"{val_avg.get('loss_giou', 0):.6f}",
                "loss_cardinality": f"{val_avg.get('loss_cardinality', 0):.6f}",
            },
        ]
        write_losses(losses_csv, rows)

        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
        save_checkpoint(ckpt_path, model, optimizer, epoch, val_avg["loss_total"], scheduler)

        if val_avg["loss_total"] < best_loss:
            best_loss = val_avg["loss_total"]
            best_path = os.path.join(args.output_dir, "checkpoint_best.pt")
            save_checkpoint(best_path, model, optimizer, epoch, best_loss, scheduler)

    writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
