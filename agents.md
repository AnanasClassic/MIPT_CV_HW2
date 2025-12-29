# Execution plan (HW2)

## Work philosophy
1. Execute steps strictly in order; do not start the next step until the current one is correct and reproducible.
2. Before finishing, verify that every required item is implemented, runnable, and produces the expected artifacts.
3. You may create auxiliary files to speed up work, but remove anything not required for delivery before finishing.

## Style requirements
- Write code and text fully but concisely.
- No code comments.
- No Russian text in code or documentation.

## Tests
- Tests are allowed and encouraged; place them in `tests/`.
- Prefer `pytest`; run tests regularly and before finishing.

## 0) Repository setup
1. Create a clean repo structure: `src/`, `data/` (gitignored), `runs/` (optional, large), `artifacts/`.
2. Add `requirements.txt` or `pyproject.toml` with pinned versions.
3. Add reproducible entrypoints for training, evaluation, and visualization.

## 1) Data preparation (HW2)
1. Select a COCO subset with at least 10 classes and document the class list.
2. Prepare `train/val` splits in COCO format with consistent category IDs.
3. Add dataset sanity checks: class counts, sample images with boxes.

## 2) Model setup and training (HW2)
1. Choose DETR or Deformable-DETR and load pretrained weights.
2. Implement fine-tuning with a full training loop and validation.
3. Log losses and metrics to TensorBoard (classification loss, bbox loss, total loss).
4. Save checkpoints and training config (hyperparameters, class list, seed).

## 3) Profiling (HW2)
1. Add a profiling mode using `torch.profiler`.
2. Capture at least one trace (50-100 steps) and export to `artifacts/profiler/`.

## 4) Evaluation and metrics (HW2)
1. Compute mAP and mAP50 on the validation split.
2. Save a metrics table (CSV or Markdown) in `artifacts/`.

## 5) Visualization and error analysis (HW2)
1. Generate visualizations with predicted boxes and confidence scores.
2. Perform error analysis: separate classification vs localization errors.
3. Save representative examples and a short summary of findings.

## 6) Synthetic data pipeline (HW2.5)
1. Identify rare classes from the dataset statistics.
2. Generate synthetic images with Stable Diffusion + ControlNet.
3. Add synthetic data to the training set with consistent labels/boxes.
4. Train a baseline model without synthetic data and a model with synthetic data.
5. Compare metrics in an ablation table (synthetic vs no-synthetic).
6. Save sample synthetic images and describe the generation setup.

## 7) Delivery checklist
1. `README.md` includes experiment description, hyperparameters, and observations.
2. TensorBoard logs are included or linked.
3. Profiler traces are included.
4. Metrics table includes mAP and mAP50.
5. Visualizations of predictions and error analysis are included.
