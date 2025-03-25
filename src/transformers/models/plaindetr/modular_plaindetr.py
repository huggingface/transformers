# coding=utf-8
# Copyright 2025 Facebook AI Research The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch PLAINDETR model."""

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...activations import ACT2FN
from ...integrations import use_kernel_forward_from_hub
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_timm_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from ..deformable_detr.modeling_deformable_detr import DeformableDetrFrozenBatchNorm2d,DeformableDetrLearnedPositionEmbedding,DeformableDetrSinePositionEmbedding






class PlainDetrFrozenBatchNorm2d(DeformableDetrFrozenBatchNorm2d):
    pass

class PlainDetrLearnedPositionEmbedding(DeformableDetrLearnedPositionEmbedding):
    pass


class PlainDetrSinePositionEmbedding(DeformableDetrSinePositionEmbedding):
    pass



def build_position_encoding(config):
    n_steps = config.d_model // 2
    if config.position_embedding_type == "sine":
        # TODO find a better way of exposing other arguments
        position_embedding = PlainDetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        position_embedding = PlainDetrLearnedPositionEmbedding(n_steps)
    elif config.position_embedding_type == "sine":
        position_embedding = PlainDetrSinePositionEmbedding(n_steps, normalize=False)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


