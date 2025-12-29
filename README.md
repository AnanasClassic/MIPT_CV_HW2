# HW2 Report: Minimal DETR on a COCO Subset

## Goal
Train DETR on a COCO subset, log losses and metrics, profile the training loop,
and analyze model errors.

## Dataset
- Source: COCO 2017, subset with 10 classes.
- Classes: person, car, traffic light, handbag, bottle, cup, bowl, chair, dining table, book.
- Layout: `data/coco_subset/images/{train2017,val2017}` and
  `data/coco_subset/annotations/instances_{train2017,val2017}.json`.

## Preparation and sanity checks
- Subset creation: `src/prepare_coco_subset.py`.
- Sanity checks: class counts and sample images in `artifacts/data_sanity/`.

## Model and training
- Model: `facebook/detr-resnet-50` from Hugging Face Transformers.
- Optimization: AdamW, fp16 enabled, image size 512, batch size 8.
- Hyperparameters: epochs 25, lr 1e-5, weight decay 1e-4, cosine scheduler with 0.1 warmup ratio.
- Augmentations: Albumentations (flip, brightness/contrast, hue/saturation, scale, blur, noise).
- Scheduler: cosine with warmup (ratio 0.1).
- Logs: TensorBoard in `runs/`.

## Profiling
- Profiler traces: `artifacts/profiler/detr/`.

## Evaluation
- mAP and mAP50 via COCOeval.
- Metrics table: `artifacts/metrics.csv`.

Latest results (val2017):

| model | mAP | mAP50 | checkpoint |
| --- | --- | --- | --- |
| detr | 0.005931 | 0.025098 | `artifacts/detr_2/checkpoint_best.pt` |

## Visualizations and error analysis
- Predictions: `artifacts/visualizations/`.
- Error analysis: `artifacts/error_analysis/{classification,localization}/`.

## Artifacts
- `artifacts/detr_2/`: checkpoints, `losses.csv`, `config.json`.
- `artifacts/metrics.csv`: mAP/mAP50 table.
- `artifacts/data_sanity/`: class counts and sample images.
- `artifacts/detr_2/loss_curves.png`: loss plots.
- `artifacts/profiler/detr/`: profiler traces.
- `artifacts/visualizations/`: predicted boxes.
- `artifacts/error_analysis/`: classification vs localization errors.

## Repro commands
```bash
python -m src.prepare_coco_subset --coco-root data/coco --output-dir data/coco_subset --top-k 10
python -m src.sanity_coco --data-dir data/coco_subset

python -m src.train_detr \
  --data-dir data/coco_subset \
  --epochs 30 \
  --batch-size 8 \
  --image-size 512 \
  --augment \
  --fp16 \
  --output-dir artifacts/detr_2 \
  --run-name detr_stage \
  --scheduler cosine \
  --warmup-ratio 0.1

python -m src.train_detr \
  --data-dir data/coco_subset \
  --epochs 30 \
  --batch-size 8 \
  --image-size 512 \
  --augment \
  --fp16 \
  --output-dir artifacts/detr_2 \
  --run-name detr_stage \
  --scheduler cosine \
  --warmup-ratio 0.1 \
  --resume artifacts/detr_2/checkpoint_epoch_9.pt

python -m src.train_detr \
  --data-dir data/coco_subset \
  --batch-size 8 \
  --image-size 512 \
  --augment \
  --fp16 \
  --output-dir artifacts/detr_2 \
  --profile \
  --profile-only

python -m src.plot_losses --losses-csv artifacts/detr_2/losses.csv --output artifacts/detr_2/loss_curves.png
python -m src.eval_detr --data-dir data/coco_subset --checkpoint artifacts/detr_2/checkpoint_best.pt --output-dir artifacts
python -m src.visualize_predictions --data-dir data/coco_subset --checkpoint artifacts/detr_2/checkpoint_best.pt --output-dir artifacts/visualizations
python -m src.error_analysis --data-dir data/coco_subset --checkpoint artifacts/detr_2/checkpoint_best.pt --output-dir artifacts/error_analysis
```

## Notes
- DETR typically needs substantially longer training to reach strong mAP on COCO-scale data; these results reflect a short run.
- HW2.5 (synthetic data) not completed in this submission.
