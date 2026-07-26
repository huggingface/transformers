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

from ..lightglue.configuration_lightglue import LightGlueConfig
from ..lightglue.image_processing_lightglue import LightGlueImageProcessor, LightGlueImageProcessorKwargs
from ..lightglue.image_processing_pil_lightglue import LightGlueImageProcessorPil
from ..lightglue.modeling_lightglue import (
    LightGlueAttention,
    LightGlueForKeypointMatching,
    LightGlueKeypointMatchingOutput,
    LightGlueMatchAssignmentLayer,
    LightGlueMLP,
    LightGluePositionalEncoder,
    LightGluePreTrainedModel,
    LightGlueTokenConfidenceLayer,
    LightGlueTransformerLayer,
)


class LoMaConfig(LightGlueConfig):
    r"""
    keypoint_detector_config (`Union[AutoConfig, dict]`, *optional*, defaults to `SuperPointConfig`):
        Configuration of the keypoint detector. The initial LoMa integration supports SuperPoint.
    input_descriptor_dim (`int`, *optional*, defaults to 256):
        Dimension of the local descriptors supplied by the descriptor network.
    descriptor_dim (`int`, *optional*, defaults to 256):
        Dimension of the descriptors used by the matching transformer.
    attention_head_dim (`int`, *optional*, defaults to 64):
        Dimension of each attention head. The number of attention heads is derived from `descriptor_dim` when it is
        not specified.
    num_hidden_layers (`int`, *optional*, defaults to 9):
        Number of self- and cross-attention layers in the matching transformer.
    filter_threshold (`float`, *optional*, defaults to 0.1):
        Confidence threshold used to retain mutual matches.
    depth_confidence (`float`, *optional*, defaults to -1.0):
        Compatibility setting for the inherited model skeleton. LoMa does not use adaptive early stopping.
    width_confidence (`float`, *optional*, defaults to -1.0):
        Compatibility setting for the inherited model skeleton. LoMa does not use adaptive keypoint pruning.
    positional_encoding_type (`str`, *optional*, defaults to `"learnable"`):
        Type of Fourier positional encoding applied in self-attention. Supported values are `"learnable"` and
        `"fixed"`.
    positional_encoding_gamma (`float`, *optional*, defaults to 1.0):
        Frequency scale used by the Fourier positional encoding.

    Examples:
        ```python
        >>> from transformers import LoMaConfig

        >>> config = LoMaConfig()
        >>> config.num_attention_heads
        4
        ```
    """

    model_type = "loma"

    input_descriptor_dim: int = 256
    descriptor_dim: int = 256
    attention_head_dim: int | None = None
    num_hidden_layers: int = 9
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    filter_threshold: float = 0.1
    positional_encoding_type: str = "learnable"
    positional_encoding_gamma: float = 1.0

    # LoMa does not use LightGlue's adaptive early stopping or point pruning. These fields remain temporarily so the
    # inherited skeleton stays executable until the LoMa matching transformer replaces it.
    depth_confidence: float = -1.0
    width_confidence: float = -1.0

    def __post_init__(self, **kwargs):
        if self.num_attention_heads is None:
            if self.attention_head_dim is None:
                self.attention_head_dim = 64
            if self.descriptor_dim % self.attention_head_dim != 0:
                raise ValueError("descriptor_dim must be divisible by attention_head_dim")
            self.num_attention_heads = self.descriptor_dim // self.attention_head_dim
        elif self.attention_head_dim is None:
            self.attention_head_dim = self.descriptor_dim // self.num_attention_heads
        elif self.descriptor_dim // self.num_attention_heads != self.attention_head_dim:
            raise ValueError("descriptor_dim / num_attention_heads must equal attention_head_dim")

        if self.positional_encoding_type not in {"learnable", "fixed"}:
            raise ValueError("positional_encoding_type must be either 'learnable' or 'fixed'")

        super().__post_init__(**kwargs)


class LoMaKeypointMatchingOutput(LightGlueKeypointMatchingOutput):
    pass


class LoMaPositionalEncoder(LightGluePositionalEncoder):
    pass


class LoMaAttention(LightGlueAttention):
    pass


class LoMaMLP(LightGlueMLP):
    pass


class LoMaTransformerLayer(LightGlueTransformerLayer):
    pass


class LoMaMatchAssignmentLayer(LightGlueMatchAssignmentLayer):
    pass


class LoMaTokenConfidenceLayer(LightGlueTokenConfidenceLayer):
    pass


class LoMaPreTrainedModel(LightGluePreTrainedModel):
    pass


class LoMaForKeypointMatching(LightGlueForKeypointMatching):
    pass


class LoMaImageProcessorKwargs(LightGlueImageProcessorKwargs):
    pass


class LoMaImageProcessor(LightGlueImageProcessor):
    pass


class LoMaImageProcessorPil(LightGlueImageProcessorPil):
    pass


__all__ = [
    "LoMaConfig",
    "LoMaPreTrainedModel",
    "LoMaForKeypointMatching",
    "LoMaImageProcessor",
    "LoMaImageProcessorPil",
]
