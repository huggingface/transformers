# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch


class SamplingResult:
    """Bounding box sampling result.

    Source: https://github.com/open-mmlab/mmdetection/blob/ecac3a77becc63f23d9f6980b2a36f86acd00a8a/mmdet/models/task_modules/samplers/sampling_result.py#L51.
    """

    def __init__(self, pos_indices, neg_indices, bboxes, gt_bboxes, assign_result, gt_flags):
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.pos_bboxes = bboxes[pos_indices]
        self.neg_bboxes = bboxes[neg_indices]
        self.pos_is_gt = gt_flags[pos_indices]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_indices = assign_result.gt_indices[pos_indices] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            if self.pos_assigned_gt_indices.numel() != 0:
                raise ValueError("No gt bboxes available to choose from")
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_indices.long(), :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_indices]
        else:
            self.pos_gt_labels = None

    def __repr__(self):
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f"<{classname}({nice}) at {hex(id(self))}>"
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f"<{classname}({nice})>"
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    @property
    def bboxes(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        return torch.cat([self.pos_bboxes, self.neg_bboxes])

    def to(self, device):
        """Change the device of the data inplace.
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, torch.Tensor):
                _dict[key] = value.to(device)
        return self

    def __nice__(self):
        data = self.info.copy()
        data["pos_bboxes"] = data.pop("pos_bboxes").shape
        data["neg_bboxes"] = data.pop("neg_bboxes").shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = "    " + ",\n    ".join(parts)
        return "{\n" + body + "\n}"

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            "pos_indices": self.pos_indices,
            "neg_indices": self.neg_indices,
            "pos_bboxes": self.pos_bboxes,
            "neg_bboxes": self.neg_bboxes,
            "pos_is_gt": self.pos_is_gt,
            "num_gts": self.num_gts,
            "pos_assigned_gt_indices": self.pos_assigned_gt_indices,
        }
