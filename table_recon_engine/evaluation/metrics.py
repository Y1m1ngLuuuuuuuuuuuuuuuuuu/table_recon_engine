from __future__ import annotations

from table_recon_engine.data_structures import DetectionBox


def box_iou(a: DetectionBox, b: DetectionBox) -> float:
    inter_w = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    inter_h = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = inter_w * inter_h
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_boxes(
    preds: list[DetectionBox],
    targets: list[DetectionBox],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    matched_targets: set[int] = set()
    true_positive = 0
    for pred in sorted(preds, key=lambda box: box.score, reverse=True):
        best_idx = -1
        best_iou = 0.0
        for idx, target in enumerate(targets):
            if idx in matched_targets:
                continue
            iou = box_iou(pred, target)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_targets.add(best_idx)
            true_positive += 1

    false_positive = max(0, len(preds) - true_positive)
    false_negative = max(0, len(targets) - true_positive)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
    }
