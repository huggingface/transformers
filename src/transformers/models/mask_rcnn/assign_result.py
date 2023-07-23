# Copyright (c) OpenMMLab. All rights reserved.
import torch


class AssignResult:
    """Stores assignments between predicted and truth boxes.

    Attributes:
        num_gts (`int`):
            The number of truth boxes considered when computing this assignment
        gt_indices (`torch.LongTensor`):
            For each predicted box indicates the 1-based index of the assigned truth box. 0 means unassigned and -1
            means ignore.
        max_overlaps (`torch.FloatTensor`):
            The iou between the predicted box and its assigned truth box.
        labels (`torch.LongTensor`, *optional*):
            If specified, indicates the category label of the assigned truth box for each predicted box.
    """

    def __init__(self, num_gts, gt_indices, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_indices = gt_indices
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties = {}

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_indices)

    def add_ground_truth(self, gt_labels):
        """Add ground truth as assigned results.

        Args:
            gt_labels (`torch.Tensor`):
                Labels of ground truth boxes.
        """
        self_indices = torch.arange(1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_indices = torch.cat([self_indices, self.gt_indices])

        self.max_overlaps = torch.cat([self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])

        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
