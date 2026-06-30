# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Idefics model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="HuggingFaceM4/idefics-9b")
@strict
class IdeficsVisionConfig(PreTrainedConfig):
    model_type = "idefics_vision"
    attribute_map = {"hidden_size": "embed_dim"}

    embed_dim: int = 768
    image_size: int | list[int] | tuple[int, int] = 224
    intermediate_size: int = 5120
    patch_size: int | list[int] | tuple[int, int] = 14
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_channels: int = 3
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0


@auto_docstring(checkpoint="HuggingFaceM4/idefics-9b")
@strict
class IdeficsPerceiverConfig(PreTrainedConfig):
    r"""
    use_resampler (`bool`, *optional*, defaults to `False`):
        Whether or not to use the resampler
    resampler_n_latents (`int`, *optional*, defaults to 64):
        Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
    resampler_depth (`int`, *optional*, defaults to 6):
        Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (< 3).
    resampler_n_heads (`int`, *optional*, defaults to 16):
        Number of heads in each Transformer block (for multi-headed self-attention).
    resampler_head_dim (`int`, *optional*, defaults to 96):
        Dimensionality of each head projection in the Transformer block.
    qk_layer_norms_perceiver (`bool`, *optional*, defaults to `False`):
        Whether or not to use qk layer norms in perceiver
    """

    model_type = "idefics_perciever"

    use_resampler: bool = False
    resampler_n_latents: int = 64
    resampler_depth: int = 6
    resampler_n_heads: int = 16
    resampler_head_dim: int = 96
    qk_layer_norms_perceiver: bool = False


@auto_docstring(checkpoint="HuggingFaceM4/idefics-9b")
@strict
class IdeficsConfig(PreTrainedConfig):
    r"""
    additional_vocab_size (`int`, *optional*, defaults to 0):
        Additional vocabulary size of the model, typically for the special "<img>" token. Additional vocab tokens
        are always trainable whereas regular vocab tokens can be frozen or not.
    alpha_initializer (`str`, *optional*, defaults to `"zeros"`):
        Initialization type for the alphas.
    alphas_initializer_range (`float`, *optional*, defaults to 0.0):
        The standard deviation of the truncated_normal_initializer for initializing the alphas in the Gated Cross
        Attention.
    alpha_type (`str`, *optional*, defaults to `"float"`):
        Whether the gating alphas should be vectors or single floats.
    cross_layer_interval (`int`, *optional*, default to 1):
        Interval for cross attention (from text to image) layers.
    qk_layer_norms (`bool`, *optional*, defaults to `False`):
        Whether to add layer norm after q and k
    freeze_text_layers (`bool`, *optional*, defaults to `True`):
        Whether to freeze text layers
    freeze_text_module_exceptions (`bool`, *optional*, defaults to `[]`):
        Exceptions to freezing text layers when `freeze_text_layers` is `True`
    freeze_lm_head (`bool`, *optional*, defaults to `False`):
        Whether to freeze lm head
    freeze_vision_layers (`bool`, *optional*, defaults to `True`):
        Whether to freeze vision layers
    freeze_vision_module_exceptions (`bool`, *optional*, defaults to `[]`):
        Exceptions to freezing vision layers when `freeze_vision_layers` is `True`
    use_resampler (`bool`, *optional*, defaults to `False`):
        Whether to use the Resampler
    perceiver_config (`IdeficsPerceiverConfig`,  *optional*):
        Custom perceiver config or dict

    Example:

    ```python
    >>> from transformers import IdeficsModel, IdeficsConfig

    >>> # Initializing a Idefics idefics-9b style configuration
    >>> configuration = IdeficsConfig()

    >>> # Initializing a model from the idefics-9b style configuration
    >>> model = IdeficsModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "idefics"
    sub_configs = {"perceiver_config": IdeficsPerceiverConfig, "vision_config": IdeficsVisionConfig}

    vocab_size: int = 32000
    additional_vocab_size: int = 0
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    dropout: float | int = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    alpha_initializer: str = "zeros"
    alphas_initializer_range: float = 0.0
    alpha_type: str = "float"
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    cross_layer_interval: int = 1
    qk_layer_norms: bool = False
    freeze_text_layers: bool = True
    freeze_text_module_exceptions: list | tuple = ()
    freeze_lm_head: bool = False
    freeze_vision_layers: bool = True
    freeze_vision_module_exceptions: list | tuple = ()
    use_resampler: bool = False
    vision_config: dict | PreTrainedConfig | None = None
    perceiver_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.perceiver_config is None:
            self.perceiver_config = IdeficsPerceiverConfig()
        elif isinstance(self.perceiver_config, dict):
            self.perceiver_config = IdeficsPerceiverConfig(**self.perceiver_config)

        if self.vision_config is None:
            self.vision_config = IdeficsVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = IdeficsVisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["IdeficsConfig", "IdeficsPerceiverConfig", "IdeficsVisionConfig"]
