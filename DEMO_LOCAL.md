# Local Demo Guide

This guide runs the final table structure demo locally without downloading the training dataset.

## 1. Environment

Use the existing local conda environment:

```bash
conda activate cv_course
pip install ultralytics
```

`cv_course` should have:

```text
torch
ultralytics
Pillow
opencv-python
matplotlib
numpy
PyYAML
```

On Apple Silicon, PyTorch can use MPS. If MPS causes a runtime issue, use `--device cpu`.

## 2. Required Local Files

Final model weight:

```text
local_outputs/weights/structure_50k_finetune_5k_v2_best.pt
```

Demo input images:

```text
local_outputs/demo_inputs/
```

The demo does not need the PubTables-1M dataset.

## 3. Run Demo

```bash
conda run -n cv_course python -m table_recon_engine.demo_final \
  --weights local_outputs/weights/structure_50k_finetune_5k_v2_best.pt \
  --source local_outputs/demo_inputs \
  --output-dir local_outputs/demo_outputs/final_demo \
  --imgsz 960 \
  --conf 0.4
```

If MPS is unstable on the local machine:

```bash
conda run -n cv_course python -m table_recon_engine.demo_final \
  --weights local_outputs/weights/structure_50k_finetune_5k_v2_best.pt \
  --source local_outputs/demo_inputs \
  --output-dir local_outputs/demo_outputs/final_demo_cpu \
  --imgsz 960 \
  --conf 0.4 \
  --device cpu
```

## 4. Outputs

The demo writes:

```text
detections_raw.json
detections_post_v3.json
box_overlays/*_boxes.png
latex_renders/latex/*.tex
latex_renders/renders/*_latex.png
latex_renders/comparisons/*_comparison.png
summary.json
```

For the demo video, show:

```text
1. local_outputs/demo_inputs/
2. Run the command.
3. Open box_overlays/*_boxes.png.
4. Open latex_renders/comparisons/*_comparison.png.
5. Open detections_post_v3.json or one generated .tex file.
```
