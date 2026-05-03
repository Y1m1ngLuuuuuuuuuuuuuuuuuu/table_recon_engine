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

这个入口会把 Ultralytics 和 Matplotlib 的运行配置目录固定到 `--work-dir/runtime`，避免训练过程在项目目录外写缓存或字体文件。

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

如果已经准备好了同一份数据，但想换权重或换 run 名字，可以使用 `--dataset-name` 复用数据目录：

```bash
python3 -m table_recon_engine.experiments.first_layer_pipeline \
  --extracted-dir /root/autodl-tmp/table_recon_engine_train/datasets/pubtables1m_structure/extracted \
  --work-dir /root/autodl-tmp/table_recon_engine_train \
  --dataset-name structure_50k_yolov8s \
  --name structure_50k_finetune_5k \
  --model /root/autodl-tmp/table_recon_engine_train/runs/yolo_structure_pubtables_5k_noamp/weights/best.pt \
  --epochs 60 \
  --imgsz 960 \
  --batch 16 \
  --workers 8 \
  --no-amp \
  --skip-prepare
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

## 8. 预测 JSON 后处理

YOLO 的普通 NMS 基于二维 IoU，对横向长条的 `table row` 去重不够敏感。第一层预测 JSON 可以先经过结构后处理：

```bash
python3 -m table_recon_engine.postprocess_structure_json \
  --input-json /root/autodl-tmp/table_recon_engine_train/outputs/structure_50k_finetune_5k_v2/pred_structure_val.json \
  --output-json /root/autodl-tmp/table_recon_engine_train/outputs/structure_50k_finetune_5k_v2/pred_structure_val_post.json
```

默认策略：

- `table` 每张图只保留最高置信度的一个整表框。
- `table row` 使用 Y 方向重叠去重，保留置信度最高的行框。
- `table column` 使用 X 方向重叠去重，并将置信度阈值提高到 `0.65`。
- `table column header` 使用 Y 方向重叠去重，并将置信度阈值提高到 `0.50`。
- `table projected row header` 使用 Y 方向重叠去重。
- `table spanning cell` 投影到去重后的行列网格，只保留覆盖多行或多列的逻辑跨度，并为后续 LaTeX 重建写入 `logical_span` 与 `projected_bbox`。

然后用后处理后的 JSON 重新评估：

```bash
python3 -m table_recon_engine.evaluation.detection_json \
  --gt-json /root/autodl-tmp/table_recon_engine_train/datasets/structure_json_structure_50k_yolov8s/annotations/val.jsonl \
  --pred-json /root/autodl-tmp/table_recon_engine_train/outputs/structure_50k_finetune_5k_v2/pred_structure_val_post.json \
  --output-json /root/autodl-tmp/table_recon_engine_train/outputs/structure_50k_finetune_5k_v2/eval_structure_val_post.json
```

## 9. 标准 JSON -> LaTeX 风格渲染

GT JSON 和预测 JSON 都可以直接渲染成 `.tex` 和表格图。默认使用 booktabs/三线表风格，不画满格竖线：

```bash
python3 -m table_recon_engine.render_structure_json \
  --structure-json /root/autodl-tmp/table_recon_engine_train/datasets/structure_json_pubtables_5k/annotations/val.jsonl \
  --image-root /root/autodl-tmp/table_recon_engine_train/datasets/pubtables1m_structure/extracted \
  --output-dir /root/autodl-tmp/table_recon_engine_train/outputs/gt_latex_renders \
  --limit 12 \
  --style booktabs
```

如果需要调试每个单元格边界，可以改用 `--style grid`。

## 10. 二阶段 spanning cell 图推断

当 `table row` 和 `table column` 已经比较稳定时，可以不再把 `table spanning cell` 当作一个大框目标硬检测，而是把基础网格中的每个 cell 当成图节点，把上下/左右相邻关系当成图边，训练边分类器判断 `merge` 或 `split`。

训练标签可以直接由 PubTables-1M 的 `table spanning cell` 标注投影到 GT 行列网格得到，不需要额外人工标注：

```bash
python3 -m table_recon_engine.graph.train_span_gnn \
  --train-input-json /root/autodl-tmp/table_recon_engine_train/datasets/structure_json_structure_50k_yolov8s/annotations/train.jsonl \
  --train-label-json /root/autodl-tmp/table_recon_engine_train/datasets/structure_json_structure_50k_yolov8s/annotations/train.jsonl \
  --val-input-json /root/autodl-tmp/table_recon_engine_train/outputs/structure_50k_finetune_5k_v2/pred_structure_val_post_v3.json \
  --val-label-json /root/autodl-tmp/table_recon_engine_train/datasets/structure_json_structure_50k_yolov8s/annotations/val.jsonl \
  --output-dir /root/autodl-tmp/table_recon_engine_train/runs/span_gnn_50k
```

推理时，图模型会删除或替换原本的一阶段 spanning cell 框，然后根据边分类结果生成合法矩形跨度，写回标准 JSON 的 `logical_span` 与 `projected_bbox`：

```bash
python3 -m table_recon_engine.graph.infer_span_gnn \
  --input-json /root/autodl-tmp/table_recon_engine_train/outputs/structure_50k_finetune_5k_v2/pred_structure_val_post_v3.json \
  --output-json /root/autodl-tmp/table_recon_engine_train/outputs/structure_50k_finetune_5k_v2/pred_structure_val_graph_spans.json \
  --checkpoint /root/autodl-tmp/table_recon_engine_train/runs/span_gnn_50k/best.pt \
  --threshold 0.50
```

默认使用几何和拓扑特征，训练速度快；如果后续需要利用 cell 内墨迹密度与边界投影特征，可以额外加入 `--image-root ... --use-visual-features`。这个模块不依赖 PyG 或 DGL，`graph/span_gnn.py` 里使用纯 PyTorch 的 `index_add_` 手写 message passing，比较适合在报告中作为“结构拓扑推断”的硬核部分。

## 11. 报告重点

论文里少写 YOLO API，多写三个手写模块：

- `topology/cell_cluster.py`：动态阈值行列聚类。
- `topology/grid_builder.py`：虚拟二维网格与空缺单元格填补。
- `topology/latex_generator.py`：LaTeX 表格序列化。
- `graph/grid_graph.py` 和 `graph/span_gnn.py`：基础网格图构建、spanning cell 边标签生成、纯 PyTorch 图消息传递与合并边分类。
