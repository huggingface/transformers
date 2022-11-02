# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Adapted from PyTorch Vision (https://github.com/pytorch/vision/blob/master/references/detection/group_by_aspect_ratio.py)
"""
import bisect
import copy
from collections import defaultdict

import numpy as np
from torch.utils.data import BatchSampler, Sampler

from utils import logger


def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def create_lengths_groups(lengths, k=0):
    bins = np.arange(start=3, stop=k, step=4).tolist() if k > 0 else [10]
    groups = _quantize(lengths, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    logger.info("Using {} as bins for aspect lengths quantization".format(fbins))
    logger.info("Count of instances per bin: {}".format(counts))
    return groups


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]  # TODO
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size

        # now we have run out of elements that satisfy
        # the group criteria, let's return the remaining
        # elements so that the size of the sampler is
        # deterministic
        expected_num_batches = len(self)
        num_remaining = expected_num_batches - num_batches
        if num_remaining > 0:
            # for the remaining batches, group the batches by similar lengths
            batch_idx = []
            for group_id, idxs in sorted(buffer_per_group.items(), key=lambda x: x[0]):
                batch_idx.extend(idxs)
                if len(batch_idx) >= self.batch_size:
                    yield batch_idx[: self.batch_size]
                    batch_idx = batch_idx[self.batch_size :]
                    num_remaining -= 1
            if len(batch_idx) > 0:
                yield batch_idx
                num_remaining -= 1
        assert num_remaining == 0

    def __len__(self):
        """
        Return the number of mini-batches rather than the number of samples.
        """
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size
