# 表格结构识别项目最终说明文档

本文档用于说明当前项目的最终版本如何运行、如何演示、输出结果在哪里，以及论文/答辩时应该如何解释项目实现。

## 1. 项目最终目标

本项目最终采用“两层式”表格结构识别路线：

```text
第一层：表格图片 -> YOLO 结构检测 -> 标准结构 JSON
第二层：标准结构 JSON -> 规则后处理/网格推断 -> LaTeX 风格渲染图
```

第一层是本项目的主要训练目标。模型输入一张表格图片，输出 PubTables-1M Structure 标注体系中的六类结构框：

```text
table
table row
table column
table spanning cell
table column header
table projected row header
```

第二层不再训练新模型，而是使用手写规则把检测出的行、列、跨行/跨列区域投影到隐式二维网格中，并渲染为 LaTeX 风格的表格图，方便在论文和演示视频中展示最终效果。

## 2. 最终版本说明

最终版本不包含 GNN。之前讨论过的 GNN/Link Prediction 思路可以作为未来改进方向写进论文，但不放入最终主流程。

原因是：当前 row/column 检测已经比较稳定，项目周期内继续引入 GNN 会增加训练、调参和解释成本；而 spanning cell 召回不足可以作为当前方法的局限性进行分析，反而更适合课程报告中的错误分析部分。

最终主流程为：

```text
YOLOv8s 结构检测
-> post-v3 结构后处理
-> 标准结构 JSON
-> LaTeX 风格渲染
```

## 3. 本地环境

本地使用 conda 管理环境，当前已配置好的环境名为：

```bash
cv_course
```

主要依赖：

```text
torch
ultralytics
Pillow
opencv-python
matplotlib
numpy
PyYAML
```

Apple Silicon Mac 可以使用 PyTorch 的 MPS 后端。如果本地运行时 MPS 出现兼容问题，可以在 demo 命令中增加：

```bash
--device cpu
```

本地演示不需要下载 PubTables-1M 训练数据集，只需要权重和少量演示图片。

## 4. 本地必要文件

最终模型权重位置：

```text
local_outputs/weights/structure_50k_finetune_5k_v2_best.pt
```

演示输入图片位置：

```text
local_outputs/demo_inputs/
```

演示输出位置：

```text
local_outputs/demo_outputs/final_demo/
```

这些文件用于本地展示，不需要全部提交到 GitHub。

## 5. 一键运行本地 Demo

在项目根目录运行：

```bash
conda run -n cv_course python -m table_recon_engine.demo_final \
  --weights local_outputs/weights/structure_50k_finetune_5k_v2_best.pt \
  --source local_outputs/demo_inputs \
  --output-dir local_outputs/demo_outputs/final_demo \
  --imgsz 960 \
  --conf 0.4
```

如果 MPS 不稳定，改用 CPU：

```bash
conda run -n cv_course python -m table_recon_engine.demo_final \
  --weights local_outputs/weights/structure_50k_finetune_5k_v2_best.pt \
  --source local_outputs/demo_inputs \
  --output-dir local_outputs/demo_outputs/final_demo_cpu \
  --imgsz 960 \
  --conf 0.4 \
  --device cpu
```

成功后终端会输出类似：

```text
Wrote 3 final demo result(s) to local_outputs/demo_outputs/final_demo
summary=local_outputs/demo_outputs/final_demo/summary.json
```

## 6. Demo 输出文件解释

运行完成后，主要输出如下：

```text
local_outputs/demo_outputs/final_demo/
├── detections_raw.json
├── detections_post_v3.json
├── summary.json
├── box_overlays/
│   └── *_boxes.png
└── latex_renders/
    ├── latex/
    │   └── *.tex
    ├── renders/
    │   └── *_latex.png
    └── comparisons/
        └── *_comparison.png
```

各文件含义：

```text
detections_raw.json
```

YOLO 直接预测出来的原始结构框。

```text
detections_post_v3.json
```

经过结构后处理之后的标准 JSON，是后续 LaTeX 渲染和论文展示建议使用的结果。

```text
box_overlays/*_boxes.png
```

把模型预测框画回原图上，适合展示“模型识别出了哪些结构区域”。

```text
latex_renders/renders/*_latex.png
```

根据预测 JSON 渲染出的 LaTeX 风格表格。

```text
latex_renders/comparisons/*_comparison.png
```

左侧为原图，右侧为 LaTeX 风格渲染结果，最适合放进论文和演示视频。

```text
summary.json
```

每张演示图的结构摘要，包括网格大小、span 数量、输出文件路径等。

## 7. 演示视频建议流程

演示视频可以按下面顺序录制：

1. 打开项目文件夹，说明最终主线是“图片 -> 结构 JSON -> LaTeX 渲染”。
2. 展示 `local_outputs/demo_inputs/` 中的原始表格图片。
3. 在终端运行本地 demo 命令。
4. 打开 `box_overlays/*_boxes.png`，说明六类结构框的检测效果。
5. 打开 `detections_post_v3.json`，说明模型输出的是结构化 JSON，而不是单纯截图。
6. 打开 `latex_renders/comparisons/*_comparison.png`，展示原图和 LaTeX 渲染结果对比。
7. 最后说明当前方法的局限性：普通行列结构效果较好，但复杂 spanning cell 仍可能出现漏检或跨度不准。

## 8. 论文中建议强调的手写实现

论文中不要把重点放在 `model.predict()` 这种调用上，而应该强调下面这些自己实现的部分：

```text
数据转换：
PubTables-1M XML -> 标准结构 JSON -> YOLO txt
```

```text
结构后处理：
按类别阈值过滤、行/列方向 NMS、table 单目标保留、spanning cell 网格投影
```

```text
LaTeX 风格渲染：
根据 row/column 构建隐式网格，再结合 spanning cell 生成可视化表格
```

```text
测试分析：
混淆矩阵、逐类 Precision/Recall/F1、非几何鲁棒性测试、成功与失败案例对比
```

鲁棒性测试中的 `gaussian_blur` 使用手写高斯卷积实现：先根据 `sigma` 生成二维高斯核，再对 RGB 图像进行 padding 和卷积累加。该部分可以和数字图像处理课程中的均值卷积、高斯卷积核联系起来说明。

这些内容最能体现“不是单纯调包”，而是围绕表格结构识别任务做了完整工程链路。

## 9. 当前实验结果摘要

当前最终后处理版本的主要指标：

```text
all_class_count_exact_rate: 0.6896
table F1: 0.9994
table row F1: 0.9846
table column F1: 0.9971
table spanning cell F1: 0.8259
table column header F1: 0.9681
table projected row header F1: 0.8947
```

解释时要注意：逐类 F1 很高，说明单类结构检测效果较好；但所有类别数量同时完全正确的比例只有约 68.96%，这是因为一张表里任何一类多检/漏检都会导致整图级 exact count 失败。这个指标更严格，也更能反映完整结构重建的难度。

## 10. 报告素材位置

整理好的论文素材位于：

```text
local_outputs/report_ready_assets/
```

压缩包：

```text
local_outputs/report_ready_assets.zip
```

其中包括混淆矩阵、鲁棒性测试、对比图、项目报告草稿等内容。

## 11. GitHub 与 AutoDL 同步状态

当前代码已经同步到 GitHub main 分支，本地和 AutoDL 项目仓库保持一致。

AutoDL 训练数据和完整训练输出仍保存在服务器项目目录中；本地只保留演示所需权重、少量图片和报告素材，避免占用太多本地空间。
