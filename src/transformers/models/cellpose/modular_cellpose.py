# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""PyTorch Cellpose-SAM model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..sam.configuration_sam import SamVisionConfig
from ..sam.modeling_sam import SamPreTrainedModel, SamVisionAttention, SamVisionEncoder, SamVisionNeck


@auto_docstring(checkpoint="mouseland/cellpose-sam")
@strict
class CellposeConfig(SamVisionConfig):
    r"""
    output_channels (`int`, *optional*, defaults to 256):
        Dimensionality of the output channels of the encoder neck.
    use_rel_pos (`bool`, *optional*, defaults to `True`):
        Whether to use relative position embeddings in the attention layers.
    window_size (`int`, *optional*, defaults to 0):
        Window size of the windowed attention layers. `0` makes every layer use global attention.
    global_attn_indexes (`list[int]`, *optional*, defaults to `()`):
        The indexes of the global attention layers when `window_size > 0`.
    mlp_dim (`int`, *optional*, defaults to 4096):
        The dimensionality of the MLP layer in the Transformer encoder.
    num_outputs (`int`, *optional*, defaults to 3):
        Number of predicted maps per pixel: vertical flow, horizontal flow and cell probability.

    Example:

    ```python
    >>> from transformers import CellposeConfig, CellposeForInstanceSegmentation

    >>> configuration = CellposeConfig()
    >>> model = CellposeForInstanceSegmentation(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "cellpose"
    base_config_key = AttributeError()

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    image_size: int | list[int] | tuple[int, int] = 256
    patch_size: int | list[int] | tuple[int, int] = 8
    initializer_range: float = 0.02
    window_size: int = 0
    global_attn_indexes: list[int] | tuple[int, ...] = ()
    mlp_dim: int = 4096
    num_outputs: int = 3

    num_pos_feats = AttributeError()
    mlp_ratio = AttributeError()

    def __post_init__(self, **kwargs):
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(
    custom_intro="""
    Output type of [`CellposeForInstanceSegmentation`].
    """
)
@dataclass
class CellposeInstanceSegmentationOutput(ModelOutput):
    r"""
    logits (`torch.FloatTensor` of shape `(batch_size, 3, height, width)`):
        Per-pixel vertical flow, horizontal flow and cell probability logits. Use
        [`CellposeImageProcessor.post_process_instance_segmentation`] to turn them into instance masks.
    """

    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class CellposeVisionAttention(SamVisionAttention):
    pass


class CellposeVisionNeck(SamVisionNeck):
    def __init__(self, config: CellposeConfig):
        super().__init__(config)


@auto_docstring
class CellposePreTrainedModel(SamPreTrainedModel):
    config: CellposeConfig
    base_model_prefix = "encoder"
    _no_split_modules = ["CellposeVisionAttention"]
    _keys_to_ignore_on_load_unexpected = [r"W2"]

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, CellposeVisionAttention):
            if module.use_rel_pos:
                init.zeros_(module.rel_pos_h)
                init.zeros_(module.rel_pos_w)
        elif isinstance(module, CellposeModel):
            if self.config.use_abs_pos:
                init.zeros_(module.pos_embed)


@auto_docstring(
    custom_intro="""
    The Cellpose-SAM image encoder: a SAM ViT encoder with global attention in every layer, returning the neck
    features of shape `(batch_size, output_channels, height // patch_size, width // patch_size)`.
    """
)
class CellposeModel(SamVisionEncoder):
    def __init__(self, config: CellposeConfig):
        super().__init__(config)

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]) -> BaseModelOutput:
        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        hidden_states = self.neck(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring(
    custom_intro="""
    Cellpose-SAM: predicts per-pixel flows and cell probabilities, turned into instance masks by
    [`CellposeImageProcessor.post_process_instance_segmentation`].
    """
)
class CellposeForInstanceSegmentation(CellposePreTrainedModel):
    def __init__(self, config: CellposeConfig):
        super().__init__(config)
        self.encoder = CellposeModel(config)
        self.readout = nn.Conv2d(config.output_channels, config.num_outputs * config.patch_size**2, kernel_size=1)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> CellposeInstanceSegmentationOutput:
        outputs = self.encoder(pixel_values, **kwargs)
        # The original model unpatchifies with a `conv_transpose2d` on an identity kernel, i.e. a pixel shuffle
        logits = F.pixel_shuffle(self.readout(outputs.last_hidden_state), self.config.patch_size)
        return CellposeInstanceSegmentationOutput(
            logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


__all__ = ["CellposeConfig", "CellposePreTrainedModel", "CellposeModel", "CellposeForInstanceSegmentation"]
