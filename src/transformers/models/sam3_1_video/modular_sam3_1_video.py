# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto import CONFIG_MAPPING
from ..sam3.configuration_sam3 import Sam3VisionConfig
from ..sam3.modeling_sam3 import (
    Sam3FPNLayer,
    Sam3SinePositionEmbedding,
    Sam3VisionEncoderOutput,
    Sam3VisionModel,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="facebook/sam3.1")
@strict
class Sam3_1VisionConfig(Sam3VisionConfig):
    r"""
    fpn_hidden_size (`int`, *optional*, defaults to 256):
        The hidden dimension of the FPN.
    backbone_feature_sizes (`List[List[int]]`, *optional*, defaults to `[[288, 288], [144, 144], [72, 72]]`):
        The spatial sizes (height, width) of the feature maps from the backbone at different scales.
    scale_factors (`list[float]`, *optional*, defaults to `[4.0, 2.0, 1.0]`):
        Scale factors for FPN multi-scale features. SAM3.1 uses a three-level pyramid
        (4x, 2x, 1x upsampling) without the 0.5x downsampling level present in SAM3.
    """

    def __post_init__(self, **kwargs):
        self.scale_factors = [4.0, 2.0, 1.0] if self.scale_factors is None else self.scale_factors
        if self.backbone_feature_sizes is None:
            self.backbone_feature_sizes = [[288, 288], [144, 144], [72, 72]]
        if isinstance(self.backbone_config, dict):
            self.backbone_config["model_type"] = self.backbone_config.get("model_type", "sam3_1_vit_model")
            self.backbone_config = CONFIG_MAPPING[self.backbone_config["model_type"]](**self.backbone_config)
        elif self.backbone_config is None:
            self.backbone_config = CONFIG_MAPPING["sam3_vit_model"]()
        super().__post_init__(**kwargs)


@auto_docstring
@dataclass
class Sam3_1VisionEncoderOutput(Sam3VisionEncoderOutput):
    r"""
    sam3_fpn_hidden_states (`tuple[torch.FloatTensor]`):
        Tuple of multi-level FPN feature maps produced by the SAM3 detection head convolutions.
    sam3_fpn_position_encoding (`tuple[torch.FloatTensor]`):
        Tuple of sinusoidal position encodings for each SAM3 FPN level.
    interactive_fpn_hidden_states (`tuple[torch.FloatTensor]`):
        Tuple of multi-level FPN feature maps produced by the interactive head convolutions.
    interactive_fpn_position_encoding (`tuple[torch.FloatTensor]`):
        Tuple of sinusoidal position encodings for each interactive FPN level.
    propagation_fpn_hidden_states (`tuple[torch.FloatTensor]`):
        Tuple of multi-level FPN feature maps produced by the propagation head convolutions.
    propagation_fpn_position_encoding (`tuple[torch.FloatTensor]`):
        Tuple of sinusoidal position encodings for each propagation FPN level.
    """

    fpn_hidden_states = AttributeError()
    fpn_position_encoding = AttributeError()
    sam3_fpn_hidden_states: tuple[torch.FloatTensor, ...] = None
    sam3_fpn_position_encoding: tuple[torch.FloatTensor, ...] = None
    interactive_fpn_hidden_states: tuple[torch.FloatTensor, ...] = None
    interactive_fpn_position_encoding: tuple[torch.FloatTensor, ...] = None
    propagation_fpn_hidden_states: tuple[torch.FloatTensor, ...] = None
    propagation_fpn_position_encoding: tuple[torch.FloatTensor, ...] = None


class Sam3_1FPNLayer(Sam3FPNLayer):
    pass


class Sam3_1VisionNeck(nn.Module):
    def __init__(self, config: Sam3_1VisionConfig):
        super().__init__()
        self.config = config

        self.position_encoding = Sam3SinePositionEmbedding(num_pos_feats=config.fpn_hidden_size // 2, normalize=True)

        self.sam3_fpn_layers = nn.ModuleList(
            [
                Sam3_1FPNLayer(
                    in_channels=config.backbone_config.hidden_size,
                    fpn_dim=config.fpn_hidden_size,
                    scale_factor=scale,
                )
                for scale in config.scale_factors
            ]
        )
        self.interactive_fpn_layers = nn.ModuleList(
            [
                Sam3_1FPNLayer(
                    in_channels=config.backbone_config.hidden_size,
                    fpn_dim=config.fpn_hidden_size,
                    scale_factor=scale,
                )
                for scale in config.scale_factors
            ]
        )
        self.propagation_fpn_layers = nn.ModuleList(
            [
                Sam3_1FPNLayer(
                    in_channels=config.backbone_config.hidden_size,
                    fpn_dim=config.fpn_hidden_size,
                    scale_factor=scale,
                )
                for scale in config.scale_factors
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        sam3_fpn_hidden_states = ()
        sam3_fpn_position_encoding = ()
        interactive_fpn_hidden_states = ()
        interactive_fpn_position_encoding = ()
        propagation_fpn_hidden_states = ()
        propagation_fpn_position_encoding = ()

        for sam3_fpn_layer, interactive_fpn_layer, propagation_fpn_layer in zip(
            self.sam3_fpn_layers, self.interactive_fpn_layers, self.propagation_fpn_layers
        ):
            sam3_out = sam3_fpn_layer(hidden_states)
            sam3_fpn_hidden_states += (sam3_out,)
            sam3_fpn_position_encoding += (self.position_encoding(sam3_out.shape, sam3_out.device, sam3_out.dtype),)

            interactive_out = interactive_fpn_layer(hidden_states)
            interactive_fpn_hidden_states += (interactive_out,)
            interactive_fpn_position_encoding += (
                self.position_encoding(interactive_out.shape, interactive_out.device, interactive_out.dtype),
            )

            propagation_out = propagation_fpn_layer(hidden_states)
            propagation_fpn_hidden_states += (propagation_out,)
            propagation_fpn_position_encoding += (
                self.position_encoding(propagation_out.shape, propagation_out.device, propagation_out.dtype),
            )

        return (
            sam3_fpn_hidden_states,
            sam3_fpn_position_encoding,
            interactive_fpn_hidden_states,
            interactive_fpn_position_encoding,
            propagation_fpn_hidden_states,
            propagation_fpn_position_encoding,
        )


@auto_docstring(
    custom_intro="""
    The SAM3.1 vision encoder: a shared ViT backbone followed by three independent FPN
    necks (SAM3 detection, interactive, and propagation heads), each producing
    multi-scale feature maps used by different parts of the tracker-video pipeline.
    """
)
class Sam3_1VisionModel(Sam3VisionModel):
    config_class = Sam3_1VisionConfig

    def __init__(self, config: Sam3_1VisionConfig):
        super().__init__(config)
        self.neck = Sam3_1VisionNeck(config)

    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Sam3_1VisionEncoderOutput:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        backbone_output = self.backbone(pixel_values, **kwargs)
        hidden_states = backbone_output.last_hidden_state

        batch_size = hidden_states.shape[0]
        height = pixel_values.shape[-2] // self.config.backbone_config.patch_size
        width = pixel_values.shape[-1] // self.config.backbone_config.patch_size
        hidden_states_spatial = hidden_states.view(batch_size, height, width, -1).permute(0, 3, 1, 2)

        (
            sam3_fpn_hidden_states,
            sam3_fpn_position_encoding,
            interactive_fpn_hidden_states,
            interactive_fpn_position_encoding,
            propagation_fpn_hidden_states,
            propagation_fpn_position_encoding,
        ) = self.neck(hidden_states_spatial)

        return Sam3_1VisionEncoderOutput(
            last_hidden_state=hidden_states,
            sam3_fpn_hidden_states=sam3_fpn_hidden_states,
            sam3_fpn_position_encoding=sam3_fpn_position_encoding,
            interactive_fpn_hidden_states=interactive_fpn_hidden_states,
            interactive_fpn_position_encoding=interactive_fpn_position_encoding,
            propagation_fpn_hidden_states=propagation_fpn_hidden_states,
            propagation_fpn_position_encoding=propagation_fpn_position_encoding,
            hidden_states=backbone_output.hidden_states,
            attentions=backbone_output.attentions,
        )


__all__ = ["Sam3_1VisionConfig", "Sam3_1VisionModel"]
