# coding=utf-8
# Copyright 2025 ByteDance and The HuggingFace Team. All rights reserved.
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

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from transformers.models.blip.image_processing_blip import BlipImageProcessor

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...generation import ClassifierFreeGuidanceLogitsProcessor, GenerationMixin, GenerationMode, LogitsProcessorList
from ...generation.utils import GenerateDecoderOnlyOutput
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple, is_torch_available, is_vision_available, logging
from ..auto import AutoModel


if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

if is_vision_available():
    import PIL

from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)

