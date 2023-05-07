# Copyright (c) OpenMMLab. All rights reserved.
import warnings

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

    def __repr__(self):
        """str: the string of the module"""
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f"<{classname}({nice}) at {hex(id(self))}>"
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        """str: the string of the module"""
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f"<{classname}({nice})>"
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_indices)

    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        if key in self.info:
            raise KeyError(f"Key {key} already exists in the info dict")
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            "num_gts": self.num_gts,
            "num_preds": self.num_preds,
            "gt_indices": self.gt_indices,
            "max_overlaps": self.max_overlaps,
            "labels": self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def __nice__(self):
        """str: a "nice" summary string describing this assign result"""
        parts = []
        parts.append(f"num_gts={self.num_gts!r}")
        if self.gt_indices is None:
            parts.append(f"gt_indices={self.gt_indices!r}")
        else:
            parts.append(f"gt_indices.shape={tuple(self.gt_indices.shape)!r}")
        if self.max_overlaps is None:
            parts.append(f"max_overlaps={self.max_overlaps!r}")
        else:
            parts.append(f"max_overlaps.shape={tuple(self.max_overlaps.shape)!r}")
        if self.labels is None:
            parts.append(f"labels={self.labels!r}")
        else:
            parts.append(f"labels.shape={tuple(self.labels.shape)!r}")
        return ", ".join(parts)

    def add_gt_(self, gt_labels):
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
