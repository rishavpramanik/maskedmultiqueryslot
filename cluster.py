from typing import Any, Dict, Optional

import numpy as np
import scipy.optimize
import torch
import torchmetrics
import torchvision

def masks_to_bboxes(masks: torch.Tensor, empty_value: float = -1.0) -> torch.Tensor:
    """Compute bounding boxes around the provided masks.

    Adapted from DETR: https://github.com/facebookresearch/detr/blob/main/util/box_ops.py

    Args:
        masks: Tensor of shape (N, H, W), where N is the number of masks, H and W are the spatial
            dimensions.
        empty_value: Value bounding boxes should contain for empty masks.

    Returns:
        Tensor of shape (N, 4), containing bounding boxes in (x1, y1, x2, y2) format, where (x1, y1)
        is the coordinate of top-left corner and (x2, y2) is the coordinate of the bottom-right
        corner (inclusive) in pixel coordinates. If mask is empty, all coordinates contain
        `empty_value` instead.
    """
    masks = masks.bool()
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    large_value = 1e8
    inv_mask = ~masks

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x, indexing="ij")

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(inv_mask, large_value).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(inv_mask, large_value).flatten(1).min(-1)[0]

    bboxes = torch.stack((x_min, y_min, x_max, y_max), dim=1)
    bboxes[x_min == large_value] = empty_value

    return bboxes

class UnsupervisedBboxIoUMetric(torchmetrics.Metric):
    """Computes IoU metric for bounding boxes when correspondences to ground truth are not known.

    Currently, assumes segmentation masks as input for both prediction and targets.

    Args:
        target_is_mask: If `True`, assume input is a segmentation mask, in which case the masks are
            converted to bounding boxes before computing IoU. If `False`, assume the input for the
            targets are already bounding boxes.
        use_threshold: If `True`, convert predicted class probabilities to mask using a threshold.
            If `False`, class probabilities are turned into mask using a softmax instead.
        threshold: Value to use for thresholding masks.
        matching: How to match predicted boxes to ground truth boxes. For "hungarian", computes
            assignment that maximizes total IoU between all boxes. For "best_overlap", uses the
            predicted box with maximum overlap for each ground truth box (each predicted box
            can be assigned to multiple ground truth boxes).
        compute_discovery_fraction: Instead of the IoU, compute the fraction of ground truth classes
            that were "discovered", meaning that they have an IoU greater than some threshold. This
            is recall, or sometimes called the detection rate metric.
        correct_localization: Instead of the IoU, compute the fraction of images on which at least
            one ground truth bounding box was correctly localised, meaning that they have an IoU
            greater than some threshold.
        discovery_threshold: Minimum IoU to count a class as discovered/correctly localized.
    """

    def __init__(
        self,
        target_is_mask: bool = True,
        use_threshold: bool = False,
        threshold: float = 0.5,
        matching: str = "hungarian",
        compute_discovery_fraction: bool = False,
        correct_localization: bool = False,
        discovery_threshold: float = 0.5,
        ignore_background=True
    ):
        super().__init__()
        self.target_is_mask = target_is_mask
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.discovery_threshold = discovery_threshold
        self.compute_discovery_fraction = compute_discovery_fraction
        self.correct_localization = correct_localization
        if compute_discovery_fraction and correct_localization:
            raise ValueError(
                "Only one of `compute_discovery_fraction` and `correct_localization` can be enabled."
            )

        matchings = ("hungarian", "best_overlap")
        if matching not in matchings:
            raise ValueError(f"Unknown matching type {matching}. Valid values are {matchings}.")
        self.matching = matching
        self.ignore_background=ignore_background

        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of instances. Assumes class probabilities as inputs.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of instance, if using masks as input, or bounding boxes of shape (B, K, 4)
                or (B, F, K, 4).
        """
        if prediction.ndim == 5:
            # Merge batch and frame dimensions
            prediction = prediction.flatten(0, 1)
            target = target.flatten(0, 1)
        elif prediction.ndim != 4:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        bs, n_pred_classes = prediction.shape[:2]
        n_gt_classes = target.shape[1]

        if self.use_threshold:
            prediction = prediction > self.threshold
        else:
            indices = torch.argmax(prediction, dim=1)
            prediction = torch.nn.functional.one_hot(indices, num_classes=n_pred_classes)
            prediction = prediction.permute(0, 3, 1, 2)

        pred_bboxes = masks_to_bboxes(prediction.flatten(0, 1)).unflatten(0, (bs, n_pred_classes))
        if self.target_is_mask:
            if self.ignore_background:
                target = target[:, 1:]
                target_bboxes = masks_to_bboxes(target.flatten(0, 1)).unflatten(0, (bs, n_gt_classes-1))
            else:
                target_bboxes = masks_to_bboxes(target.flatten(0, 1)).unflatten(0, (bs, n_gt_classes))
        else:
            assert target.shape[-1] == 4
            # Convert all-zero boxes added during padding to invalid boxes
            target[torch.all(target == 0.0, dim=-1)] = -1.0
            target_bboxes = target

        for pred, target in zip(pred_bboxes, target_bboxes):
            valid_pred_bboxes = pred[:, 0] != -1.0
            valid_target_bboxes = target[:, 0] != -1.0
            if valid_target_bboxes.sum() == 0:
                continue  # Skip data points without any target bbox

            pred = pred[valid_pred_bboxes]
            target = target[valid_target_bboxes]

            if valid_pred_bboxes.sum() > 0:
                iou_per_bbox = unsupervised_bbox_iou(
                    pred, target, matching=self.matching, reduction="none"
                )
            else:
                iou_per_bbox = torch.zeros_like(valid_target_bboxes, dtype=torch.float32)

            if self.compute_discovery_fraction:
                discovered = iou_per_bbox > self.discovery_threshold
                self.values += discovered.sum() / len(iou_per_bbox)
            elif self.correct_localization:
                correctly_localized = torch.any(iou_per_bbox > self.discovery_threshold)
                self.values += correctly_localized.sum()
            else:
                self.values += iou_per_bbox.mean()
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.values)
        else:
            return self.values / self.total


class BboxCorLocMetric(UnsupervisedBboxIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", correct_localization=True, **kwargs)


class BboxRecallMetric(UnsupervisedBboxIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", compute_discovery_fraction=True, **kwargs)


def unsupervised_bbox_iou(
    pred_bboxes: torch.Tensor,
    true_bboxes: torch.Tensor,
    matching: str = "best_overlap",
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute IoU between two sets of bounding boxes.

    Args:
        pred_bboxes: Predicted bounding boxes of shape N x 4.
        true_bboxes: True bounding boxes of shape M x 4.
        matching: Method to assign predicted to true bounding boxes.
        reduction: Whether to average the computes IoUs per true box.
    """
    n_gt_bboxes = len(true_bboxes)

    pairwise_iou = torchvision.ops.box_iou(pred_bboxes, true_bboxes)

    if matching == "hungarian":
        pred_idxs, true_idxs = scipy.optimize.linear_sum_assignment(
            pairwise_iou.cpu(), maximize=True
        )
        pred_idxs = torch.as_tensor(pred_idxs, dtype=torch.int64, device=pairwise_iou.device)
        true_idxs = torch.as_tensor(true_idxs, dtype=torch.int64, device=pairwise_iou.device)
    elif matching == "best_overlap":
        pred_idxs = torch.argmax(pairwise_iou, dim=0)
        true_idxs = torch.arange(pairwise_iou.shape[1], device=pairwise_iou.device)
    else:
        raise ValueError(f"Unknown matching {matching}")

    matched_iou = pairwise_iou[pred_idxs, true_idxs]

    iou = torch.zeros(n_gt_bboxes, dtype=torch.float32, device=pairwise_iou.device)
    iou[true_idxs] = matched_iou

    if reduction == "mean":
        return iou.mean()
    else:
        return iou


def unsupervised_mask_iou(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    matching: str = "hungarian",
    reduction: str = "mean",
    iou_empty: float = 0.0,
) -> torch.Tensor:
    """Compute intersection-over-union (IoU) between masks with unknown class correspondences.

    This metric is also known as Jaccard index. Note that this is a non-batched implementation.

    Args:
        pred_mask: Predicted mask of shape (C, N), where C is the number of predicted classes and
            N is the number of points. Masks are assumed to be binary.
        true_mask: Ground truth mask of shape (K, N), where K is the number of ground truth
            classes and N is the number of points. Masks are assumed to be binary.
        matching: How to match predicted classes to ground truth classes. For "hungarian", computes
            assignment that maximizes total IoU between all classes. For "best_overlap", uses the
            predicted class with maximum overlap for each ground truth class (each predicted class
            can be assigned to multiple ground truth classes). Empty ground truth classes are
            assigned IoU of zero.
        reduction: If "mean", return IoU averaged over classes. If "none", return per-class IoU.
        iou_empty: IoU for the case when a class does not occur, but was also not predicted.

    Returns:
        Mean IoU over classes if reduction is `mean`, tensor of shape (K,) containing per-class IoU
        otherwise.
    """
    assert pred_mask.ndim == 2
    assert true_mask.ndim == 2
    n_gt_classes = len(true_mask)
    pred_mask = pred_mask.unsqueeze(1).to(torch.bool)
    true_mask = true_mask.unsqueeze(0).to(torch.bool)

    intersection = torch.sum(pred_mask & true_mask, dim=-1).to(torch.float64)
    union = torch.sum(pred_mask | true_mask, dim=-1).to(torch.float64)
    pairwise_iou = intersection / union

    # Remove NaN from divide-by-zero: class does not occur, and class was not predicted.
    pairwise_iou[union == 0] = iou_empty

    if matching == "hungarian":
        pred_idxs, true_idxs = scipy.optimize.linear_sum_assignment(
            pairwise_iou.cpu(), maximize=True
        )
        pred_idxs = torch.as_tensor(
            pred_idxs, dtype=torch.int64, device=pairwise_iou.device)
        true_idxs = torch.as_tensor(
            true_idxs, dtype=torch.int64, device=pairwise_iou.device)
    elif matching == "best_overlap":
        non_empty_gt = torch.sum(true_mask.squeeze(0), dim=1) > 0
        pred_idxs = torch.argmax(pairwise_iou, dim=0)[non_empty_gt]
        true_idxs = torch.arange(pairwise_iou.shape[1],device=pairwise_iou.device)[non_empty_gt]
    else:
        raise ValueError(f"Unknown matching {matching}")

    matched_iou = pairwise_iou[pred_idxs, true_idxs]
    iou = torch.zeros(n_gt_classes, dtype=torch.float64,
                      device=pairwise_iou.device)
    iou[true_idxs] = matched_iou

    if reduction == "mean":
        return iou.mean()
    else:
        return iou
    
def unsupervised_mask_iou_two(tensor1, tensor2):
    batch_size, K, _, _ = tensor1.size()
    matrix = torch.zeros((batch_size, K, K))

    # Calculate IoU between each pair of dimensions
    for b in range(batch_size):
        for i in range(K):
            for j in range(K):
                intersection = (tensor1[b, i] & tensor2[b, j]).sum()
                union = (tensor1[b, i] | tensor2[b, j]).sum()
                iou = intersection.float() / (union.float() + 1e-6)  # Avoid division by zero
                matrix[b, i, j] = iou
    cost_matrix = 1 - matrix

    # Initialize assignments
    assignments = []
    aligned=torch.zeros_like(tensor1)

    # Use Hungarian algorithm for each batch
    for b in range(batch_size):
        row_indices, col_indices = scipy.optimize.linear_sum_assignment(cost_matrix[b])
        for i,j in zip(row_indices,col_indices):
            aligned[b,i]=tensor2[b,j]
        assignments.append(col_indices)

    # Stack assignments as a tensor
    # optimal_matching = torch.stack(assignments, dim=0)

    return aligned



class UnsupervisedMaskIoUMetric(torchmetrics.Metric):
    """Computes IoU metric for segmentation masks when correspondences to ground truth are not known.

    Uses Hungarian matching to compute the assignment between predicted classes and ground truth
    classes.

    Args:
        use_threshold: If `True`, convert predicted class probabilities to mask using a threshold.
            If `False`, class probabilities are turned into mask using a softmax instead.
        threshold: Value to use for thresholding masks.
        matching: Approach to match predicted to ground truth classes. For "hungarian", computes
            assignment that maximizes total IoU between all classes. For "best_overlap", uses the
            predicted class with maximum overlap for each ground truth class. Using "best_overlap"
            leads to the "average best overlap" metric.
        compute_discovery_fraction: Instead of the IoU, compute the fraction of ground truth classes
            that were "discovered", meaning that they have an IoU greater than some threshold.
        correct_localization: Instead of the IoU, compute the fraction of images on which at least
            one ground truth class was correctly localised, meaning that they have an IoU
            greater than some threshold.
        discovery_threshold: Minimum IoU to count a class as discovered/correctly localized.
        ignore_background: If true, assume class at index 0 of ground truth masks is background class
            that is removed before computing IoU.
        ignore_overlaps: If true, remove points where ground truth masks has overlappign classes from
            predictions and ground truth masks.
    """

    def __init__(
        self,
        use_threshold: bool = False,
        threshold: float = 0.5,
        matching: str = "hungarian",
        compute_discovery_fraction: bool = False,
        correct_localization: bool = False,
        discovery_threshold: float = 0.5,
        ignore_background: bool = True,
        ignore_overlaps: bool = True,
    ):
        super().__init__()
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.discovery_threshold = discovery_threshold
        self.compute_discovery_fraction = compute_discovery_fraction
        self.correct_localization = correct_localization
        if compute_discovery_fraction and correct_localization:
            raise ValueError(
                "Only one of `compute_discovery_fraction` and `correct_localization` can be enabled."
            )

        matchings = ("hungarian", "best_overlap")
        if matching not in matchings:
            raise ValueError(
                f"Unknown matching type {matching}. Valid values are {matchings}.")
        self.matching = matching
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps

        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, ignore: Optional[torch.Tensor] = None
    ):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes. Assumes class probabilities as inputs.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            predictions = prediction.transpose(1, 2).flatten(-3, -1)
            targets = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            predictions = prediction.flatten(-2, -1)
            targets = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.use_threshold:
            predictions = predictions > self.threshold
        else:
            indices = torch.argmax(predictions, dim=1)
            predictions = torch.nn.functional.one_hot(
                indices, num_classes=predictions.shape[1])
            predictions = predictions.transpose(1, 2)

        if self.ignore_background:
            targets = targets[:, 1:]

        targets = targets > 0  # Ensure masks are binary

        if self.ignore_overlaps:
            overlaps = targets.sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            predictions[ignore.expand_as(predictions)] = 0
            targets[ignore.expand_as(targets)] = 0

        # Should be either 0 (empty, padding) or 1 (single object).
        assert torch.all(targets.sum(dim=1) <
                         2), "Issues with target format, mask non-exclusive"

        for pred, target in zip(predictions, targets):
            nonzero_classes = torch.sum(target, dim=-1) > 0
            # Remove empty (e.g. padded) classes
            target = target[nonzero_classes]
            if len(target) == 0:
                continue  # Skip elements without any target mask

            iou_per_class = unsupervised_mask_iou(
                pred, target, matching=self.matching, reduction="none"
            )

            if self.compute_discovery_fraction:
                discovered = iou_per_class > self.discovery_threshold
                self.values += discovered.sum() / len(discovered)
            elif self.correct_localization:
                correctly_localized = torch.any(
                    iou_per_class > self.discovery_threshold)
                self.values += correctly_localized.sum()
            else:
                self.values += iou_per_class.mean()
            self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.values)
        else:
            return self.values / self.total
        
    
class MaskCorLocMetric(UnsupervisedMaskIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", correct_localization=True, **kwargs)


class AverageBestOverlapMetric(UnsupervisedMaskIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", **kwargs)


class BestOverlapObjectRecoveryMetric(UnsupervisedMaskIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", compute_discovery_fraction=True, **kwargs)