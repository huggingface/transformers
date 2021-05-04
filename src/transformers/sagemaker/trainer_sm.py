# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

from ..file_utils import WEIGHTS_NAME, is_torch_tpu_available
from ..modeling_utils import PreTrainedModel, unwrap_model
from ..trainer import Trainer
from ..trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    SequentialDistributedSampler,
    nested_detach,
    nested_numpify,
    reissue_pt_warnings,
)
from ..trainer_utils import PREFIX_CHECKPOINT_DIR
from ..utils import logging
from .training_args_sm import is_sagemaker_model_parallel_available


logger = logging.get_logger(__name__)


if is_sagemaker_model_parallel_available():
    import smdistributed.modelparallel.torch as smp

    @smp.step()
    def forward_backward(model, inputs, gradient_accumulation_steps=1):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss /= gradient_accumulation_steps
        model.backward(loss)
        return loss

    @smp.step()
    def forward_only(model, inputs):
        return model(**inputs)

    def smp_gather(tensor):
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(smp_gather(t) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: smp_gather(v) for k, v in tensor.items()})
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
            )
        all_tensors = smp.allgather(tensor, smp.CommGroup.DP_GROUP)
        return torch.cat([t.cpu() for t in all_tensors], dim=0)

    def nested_smp_concat(tensor):
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(nested_smp_concat(t) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: nested_smp_concat(v) for k, v in tensor.items()})
        # It doesn't seem possible to check here if `tensor` is a StepOutput because StepOutput lives in `smp.step`
        # which is also the name of the decorator so Python is confused.
        return tensor.concat().detach().cpu()


class SageMakerTrainer(Trainer):
    def __init__(self, args=None, **kwargs):
        warnings.warn(
            "`SageMakerTrainer` is deprecated and will be removed in v5 of Transformers. You can use `Trainer` "
            "instead.",
            FutureWarning,
        )
        super().__init__(args=args, **kwargs)
