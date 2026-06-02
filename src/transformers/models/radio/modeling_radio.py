# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
from collections import namedtuple
from collections.abc import Callable

import torch
from timm.models import VisionTransformer
from torch import nn

from ...modeling_utils import PreTrainedModel

# Register extra models (side-effect imports — modules execute timm @register_model decorators).
from . import extra_models as _extra_models_register  # noqa: F401
from . import extra_timm_models as _extra_timm_register  # noqa: F401

# Import all required modules.
from .adaptor_generic import GenericAdaptor
from .configuration_radio import RADIOConfig
from .feature_normalizer import FeatureNormalizer, IntermediateFeatureNormalizer
from .input_conditioner import InputConditioner, get_default_conditioner
from .radio_model import RADIOModel as RADIOModelBase
from .radio_model import Resolution, create_model_from_args


__all__ = ["RADIOModel"]


class RADIOModel(PreTrainedModel):
    """Pretrained Hugging Face model for RADIO.

    This class inherits from PreTrainedModel, which provides
    HuggingFace's functionality for loading and saving models.
    """

    config_class = RADIOConfig
    main_input_name = "x"
    _no_split_modules = []
    supports_gradient_checkpointing = False

    def __init__(self, config: RADIOConfig):
        super().__init__(config)

        RADIOArgs = namedtuple("RADIOArgs", config.args.keys())
        args = RADIOArgs(**config.args)

        model = create_model_from_args(args)
        input_conditioner: InputConditioner = get_default_conditioner()

        dtype = getattr(args, "dtype", torch.float32)
        if isinstance(dtype, str):
            # Convert the dtype's string representation back to a dtype.
            dtype = getattr(torch, dtype)
        model.to(dtype=dtype)
        input_conditioner.dtype = dtype

        summary_idxs = torch.tensor(
            [i for i, t in enumerate(args.teachers) if t.get("use_summary", True)],
            dtype=torch.int64,
        )

        adaptor_configs = config.adaptor_configs
        adaptor_names = config.adaptor_names or []

        adaptors = dict()
        for adaptor_name in adaptor_names:
            mlp_config = adaptor_configs[adaptor_name]
            adaptor = GenericAdaptor(args, None, None, mlp_config)
            adaptor.head_idx = mlp_config["head_idx"]
            adaptors[adaptor_name] = adaptor

        feature_normalizer = None
        if config.feature_normalizer_config is not None:
            # Actual normalization values will be restored when loading checkpoint weights.
            feature_normalizer = FeatureNormalizer(config.feature_normalizer_config["embed_dim"])

        inter_feature_normalizer = None
        if config.inter_feature_normalizer_config is not None:
            inter_feature_normalizer = IntermediateFeatureNormalizer(
                config.inter_feature_normalizer_config["num_intermediates"],
                config.inter_feature_normalizer_config["embed_dim"],
                rot_per_layer=config.inter_feature_normalizer_config["rot_per_layer"],
                dtype=dtype,
            )

        self.radio_model = RADIOModelBase(
            model,
            input_conditioner,
            summary_idxs=summary_idxs,
            patch_size=config.patch_size,
            max_resolution=config.max_resolution,
            window_size=config.vitdet_window_size,
            preferred_resolution=config.preferred_resolution,
            adaptors=adaptors,
            feature_normalizer=feature_normalizer,
            inter_feature_normalizer=inter_feature_normalizer,
        )

        self.post_init()

    @property
    def adaptors(self) -> nn.ModuleDict:
        return self.radio_model.adaptors

    @property
    def model(self) -> VisionTransformer:
        return self.radio_model.model

    @property
    def input_conditioner(self) -> InputConditioner:
        return self.radio_model.input_conditioner

    @property
    def num_summary_tokens(self) -> int:
        return self.radio_model.num_summary_tokens

    @property
    def patch_size(self) -> int:
        return self.radio_model.patch_size

    @property
    def max_resolution(self) -> int:
        return self.radio_model.max_resolution

    @property
    def preferred_resolution(self) -> Resolution:
        return self.radio_model.preferred_resolution

    @property
    def window_size(self) -> int:
        return self.radio_model.window_size

    @property
    def min_resolution_step(self) -> int:
        return self.radio_model.min_resolution_step

    def make_preprocessor_external(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.radio_model.make_preprocessor_external()

    def get_nearest_supported_resolution(self, height: int, width: int) -> Resolution:
        return self.radio_model.get_nearest_supported_resolution(height, width)

    def switch_to_deploy(self):
        return self.radio_model.switch_to_deploy()

    def forward(self, x: torch.Tensor):
        return self.radio_model.forward(x)
