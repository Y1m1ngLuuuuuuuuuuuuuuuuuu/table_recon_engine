import torch


def giou_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Generalized IoU loss implemented with basic PyTorch tensor ops.

    Args:
        preds: Predicted boxes, shape (N, 4), [x_min, y_min, x_max, y_max].
        targets: Target boxes, shape (N, 4), [x_min, y_min, x_max, y_max].
        eps: Numeric stability constant.

    Returns:
        Mean loss of 1 - GIoU.
    """

    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape}, targets {targets.shape}")
    if preds.dim() != 2 or preds.size(-1) != 4:
        raise ValueError(f"GIoU expects tensors with shape (N, 4), got {preds.shape}")
    if preds.numel() == 0:
        return preds.new_tensor(0.0)

    pred_x0, pred_y0, pred_x1, pred_y1 = preds.unbind(dim=-1)
    tgt_x0, tgt_y0, tgt_x1, tgt_y1 = targets.unbind(dim=-1)

    pred_w = (pred_x1 - pred_x0).clamp(min=0)
    pred_h = (pred_y1 - pred_y0).clamp(min=0)
    tgt_w = (tgt_x1 - tgt_x0).clamp(min=0)
    tgt_h = (tgt_y1 - tgt_y0).clamp(min=0)

    pred_area = pred_w * pred_h
    tgt_area = tgt_w * tgt_h

    inter_x0 = torch.max(pred_x0, tgt_x0)
    inter_y0 = torch.max(pred_y0, tgt_y0)
    inter_x1 = torch.min(pred_x1, tgt_x1)
    inter_y1 = torch.min(pred_y1, tgt_y1)

    inter_w = (inter_x1 - inter_x0).clamp(min=0)
    inter_h = (inter_y1 - inter_y0).clamp(min=0)
    inter_area = inter_w * inter_h

    union_area = pred_area + tgt_area - inter_area
    iou = inter_area / (union_area + eps)

    enc_x0 = torch.min(pred_x0, tgt_x0)
    enc_y0 = torch.min(pred_y0, tgt_y0)
    enc_x1 = torch.max(pred_x1, tgt_x1)
    enc_y1 = torch.max(pred_y1, tgt_y1)

    enc_w = (enc_x1 - enc_x0).clamp(min=0)
    enc_h = (enc_y1 - enc_y0).clamp(min=0)
    enc_area = enc_w * enc_h

    giou = iou - ((enc_area - union_area) / (enc_area + eps))
    return (1.0 - giou).mean()
