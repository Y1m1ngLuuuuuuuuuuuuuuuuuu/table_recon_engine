# TSR Engine 新路线框架

主线流程现在拆成两层。当前优先完成第一层：

```text
第一层：表格图片 -> YOLO 结构检测 -> 标准结构 JSON
第二层：标准结构 JSON -> CellCluster/GridBuilder -> LaTeXGenerator 输出 tabular
```

## 1. 第一层标准 JSON

一张图片对应一条记录：

```json
{
  "image": "PMC1142515_table_2.jpg",
  "width": 640,
  "height": 480,
  "objects": [
    {
      "class": "table row",
      "class_id": 1,
      "bbox": [10.0, 35.0, 620.0, 58.0]
    }
  ]
}
```

训练标注 JSON 不带置信度；模型预测 JSON 会额外包含 `confidence`。第一层只负责复现 PubTables-1M Structure 标注中的六类结构框：

- `table`
- `table row`
- `table column`
- `table spanning cell`
- `table column header`
- `table projected row header`

## 2. XML -> 标准 JSON

PubTables-1M Structure VOC XML：

```bash
python3 -m table_recon_engine.converters.pubtables_xml_to_json \
  --extracted-dir /root/autodl-tmp/table_recon_engine_train/datasets/pubtables1m_structure/extracted \
  --output-dir /root/autodl-tmp/table_recon_engine_train/datasets/structure_json_pubtables \
  --train-samples 5000 \
  --val-samples 800
```

输出：

```text
annotations/train.jsonl
annotations/val.jsonl
annotations/manifest.json
```

## 3. 标准 JSON -> YOLO 数据集

```bash
python3 -m table_recon_engine.converters.structure_json_to_yolo \
  --train-annotations /root/autodl-tmp/table_recon_engine_train/datasets/structure_json_pubtables/annotations/train.jsonl \
  --val-annotations /root/autodl-tmp/table_recon_engine_train/datasets/structure_json_pubtables/annotations/val.jsonl \
  --image-root /root/autodl-tmp/table_recon_engine_train/datasets/pubtables1m_structure/extracted \
  --output-dir /root/autodl-tmp/table_recon_engine_train/datasets/yolo_structure_from_json
```

也可以使用旧的直转脚本做快速实验：

```bash
python3 -m table_recon_engine.converters.pubtables_xml_to_yolo \
  --extracted-dir /root/autodl-tmp/table_recon_engine_train/datasets/pubtables1m_structure/extracted \
  --output-dir /root/autodl-tmp/table_recon_engine_train/datasets/yolo_pubtables_structure \
  --train-samples 5000 \
  --val-samples 800
```

ICDAR cTDaR XML 快速转换：

```bash
python3 -m table_recon_engine.converters.ctdar_to_yolo \
  --annotation-root data/ctdar/xml \
  --image-root data/ctdar/images \
  --output-dir data/yolo_cells \
  --split train \
  --copy-images
```

## 4. 数据增强

```bash
python3 -m table_recon_engine.augmentation.albumentations_yolo \
  --dataset-dir data/yolo_cells \
  --split train \
  --repeats 2
```

## 5. YOLO 训练

```bash
python3 -m table_recon_engine.detection.train_yolo \
  --data /root/autodl-tmp/table_recon_engine_train/datasets/yolo_structure_from_json/data.yaml \
  --model yolov8n.yaml \
  --epochs 50 \
  --imgsz 960 \
  --batch 16 \
  --project /root/autodl-tmp/table_recon_engine_train/runs \
  --name yolo_structure_json
```

推荐在 AutoDL 上使用第一层实验入口跑较大子集：

```bash
python3 -m table_recon_engine.experiments.first_layer_pipeline \
  --extracted-dir /root/autodl-tmp/table_recon_engine_train/datasets/pubtables1m_structure/extracted \
  --work-dir /root/autodl-tmp/table_recon_engine_train \
  --name structure_50k_yolov8s \
  --train-samples 50000 \
  --val-samples 5000 \
  --model yolov8s.yaml \
  --epochs 80 \
  --imgsz 960 \
  --batch 16 \
  --workers 8 \
  --no-amp
```

## 6. YOLO 推理 -> 标准 JSON

```bash
python3 -m table_recon_engine.detection.infer_yolo \
  --weights /root/autodl-tmp/table_recon_engine_train/runs/yolo_structure_json/weights/best.pt \
  --source /root/autodl-tmp/table_recon_engine_train/datasets/yolo_structure_from_json/images/val \
  --output-json /root/autodl-tmp/table_recon_engine_train/outputs/pred_structure_val.json \
  --save-visuals
```

## 7. 第一层检测评估

```bash
python3 -m table_recon_engine.evaluation.detection_json \
  --gt-json /root/autodl-tmp/table_recon_engine_train/datasets/structure_json_pubtables/annotations/val.jsonl \
  --pred-json /root/autodl-tmp/table_recon_engine_train/outputs/pred_structure_val.json \
  --output-json /root/autodl-tmp/table_recon_engine_train/outputs/eval_structure_val.json \
  --iou-threshold 0.5
```

评估会输出每一类结构框的 Precision、Recall、F1、平均匹配 IoU，以及每张图的 `row/column/span/header` 数量是否与标注一致。

## 8. 报告重点

论文里少写 YOLO API，多写三个手写模块：

- `topology/cell_cluster.py`：动态阈值行列聚类。
- `topology/grid_builder.py`：虚拟二维网格与空缺单元格填补。
- `topology/latex_generator.py`：LaTeX 表格序列化。
