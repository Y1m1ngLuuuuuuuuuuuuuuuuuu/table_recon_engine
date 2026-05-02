# TSR Engine 新路线框架

主线流程：

```text
开源数据集标注 -> YOLO 标签转换 -> Albumentations 增强 -> YOLO 单元格检测
-> CellCluster 行列聚类 -> GridBuilder 虚拟网格 -> LaTeXGenerator 输出 tabular
```

## 1. 数据转换

PubTables/PubTabNet 风格 JSON：

```bash
python3 -m table_recon_engine.converters.pubtables_to_yolo \
  --annotations data/pubtables/train.jsonl \
  --image-root data/pubtables/images \
  --output-dir data/yolo_cells \
  --split train \
  --copy-images
```

ICDAR cTDaR XML：

```bash
python3 -m table_recon_engine.converters.ctdar_to_yolo \
  --annotation-root data/ctdar/xml \
  --image-root data/ctdar/images \
  --output-dir data/yolo_cells \
  --split train \
  --copy-images
```

## 2. 数据增强

```bash
python3 -m table_recon_engine.augmentation.albumentations_yolo \
  --dataset-dir data/yolo_cells \
  --split train \
  --repeats 2
```

## 3. YOLO 训练

```bash
python3 -m table_recon_engine.detection.train_yolo \
  --data data/yolo_cells/data.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 960
```

## 4. 推理与 LaTeX 重建

```bash
python3 -m table_recon_engine.demo \
  --weights runs/table_cells/yolo_cell_detector/weights/best.pt \
  --source demo_images \
  --output-dir outputs/demo \
  --save-visuals
```

## 5. 报告重点

论文里少写 YOLO API，多写三个手写模块：

- `topology/cell_cluster.py`：动态阈值行列聚类。
- `topology/grid_builder.py`：虚拟二维网格与空缺单元格填补。
- `topology/latex_generator.py`：LaTeX 表格序列化。
