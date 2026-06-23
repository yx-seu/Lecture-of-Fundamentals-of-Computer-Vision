from typing import Dict

import numpy as np


class SegmentationMetric:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, labels):
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        valid_mask = labels != self.ignore_index
        preds = preds[valid_mask]
        labels = labels[valid_mask]
        if preds.size == 0:
            return
        indices = self.num_classes * labels.astype(int) + preds.astype(int)
        cm = np.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += cm

    def compute(self) -> Dict[str, float]:
        cm = self.confusion_matrix.astype(np.float64)
        tp = np.diag(cm)
        gt = cm.sum(axis=1)
        pred = cm.sum(axis=0)
        total = cm.sum()

        pixel_accuracy = float(tp.sum() / total) if total > 0 else 0.0
        class_acc = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
        mean_pixel_accuracy = float(class_acc.mean()) if len(class_acc) else 0.0
        union = gt + pred - tp
        iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
        dice = np.divide(2 * tp, gt + pred, out=np.zeros_like(tp), where=(gt + pred) > 0)
        foreground_iou_values = iou[1:] if self.num_classes > 1 else np.array([], dtype=np.float64)
        foreground_dice_values = dice[1:] if self.num_classes > 1 else np.array([], dtype=np.float64)

        return {
            "pixel_accuracy": pixel_accuracy,
            "mean_pixel_accuracy": mean_pixel_accuracy,
            "miou": float(iou.mean()) if len(iou) else 0.0,
            "mean_foreground_iou": float(foreground_iou_values.mean()) if len(foreground_iou_values) else 0.0,
            "foreground_iou": float(iou[1]) if self.num_classes > 1 else 0.0,
            "background_iou": float(iou[0]) if self.num_classes > 0 else 0.0,
            "mean_dice": float(dice.mean()) if len(dice) else 0.0,
            "mean_foreground_dice": float(foreground_dice_values.mean()) if len(foreground_dice_values) else 0.0,
            "foreground_dice": float(dice[1]) if self.num_classes > 1 else 0.0,
            "background_dice": float(dice[0]) if self.num_classes > 0 else 0.0,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }
