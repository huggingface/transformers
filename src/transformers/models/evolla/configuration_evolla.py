# Copyright 2025 Westlake Representational Learning Lab (Fajie Yuan Lab) team and the HuggingFace Inc. team. All rights reserved.
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
"""Evolla model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="westlake-repl/Evolla-10B-hf")
@strict
class SaProtConfig(PreTrainedConfig):
    r"""
    mask_token_id (`int`, *optional*, defaults to 4):
        The id of the *mask* token in the protein sequence model.
    position_embedding_type (`str`, *optional*, defaults to `"rotary"`):
        The type of position embedding to use in the protein sequence model. Currently only `"rotary"` is supported.
    emb_layer_norm_before (`bool`, *optional*, defaults to `False`):
        Whether to apply layer normalization before the position embedding in the protein sequence model.
    token_dropout (`bool`, *optional*, defaults to `True`):
        Whether to apply dropout to the tokens in the protein sequence model.
    """

    vocab_size: int = 446
    mask_token_id: int = 4
    pad_token_id: int = 1
    hidden_size: int = 1280
    num_hidden_layers: int = 33
    num_attention_heads: int = 20
    intermediate_size: int = 5120
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 1026
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-05
    position_embedding_type: str = "rotary"
    emb_layer_norm_before: bool = False
    token_dropout: bool = True
    is_decoder: bool = False
    add_cross_attention: bool = False


@auto_docstring(checkpoint="westlake-repl/Evolla-10B-hf")
@strict
class EvollaConfig(PreTrainedConfig):
    r"""
    protein_encoder_config (`dict`, *optional*):
        Dictionary of configuration options used to initialize [`SaProtConfig`].
    aligner_ffn_mult (`int`, *optional*, defaults to 4):
        The FFN multiplier for the aligner layer.
    aligner_enable_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in the aligner layer.
    aligner_attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities in the aligner layer.
    aligner_num_add_layers (`int`, *optional*, defaults to 8):
        The number of additional layers for the aligner layer.
    resampler_depth (`int`, *optional*, defaults to 6):
        The depth of the resampler layer in the llama model.
    resampler_dim_head (`int`, *optional*, defaults to 64):
        The dimension of the heads in the resampler layer in the llama model.
    resampler_heads (`int`, *optional*, defaults to 8):
        The number of heads in the resampler layer in the llama model.
    resampler_num_latents (`int`, *optional*, defaults to 64):
        The number of latents in the resampler layer in the llama model.
    resampler_ff_mult (`int`, *optional*, defaults to 4):
        The FFN multiplier for the resampler layer.

    Example:

    ```python
    >>> from transformers import EvollaModel, EvollaConfig

    >>> # Initializing a Evolla evolla-10b style configuration
    >>> configuration = EvollaConfig()

    >>> # Initializing a model from the evolla-10b style configuration
    >>> model = EvollaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "EvollaModel"
    sub_configs = {"protein_encoder_config": SaProtConfig}
    default_theta = 500000.0

    protein_encoder_config: dict | PreTrainedConfig | None = None
    vocab_size: int = 128256  # llama vocab size
    hidden_size: int = 4096  # llama hidden size
    intermediate_size: int = 14336  # llama intermediate size
    num_hidden_layers: int = 32  # llama num layers
    num_attention_heads: int = 32  # llama num heads
    num_key_value_heads: int | None = 8  # llama num key-value heads
    hidden_act: str = "silu"  # llama activation function
    max_position_embeddings: int = 8192  # llama rope max length
    rms_norm_eps: float = 1e-05
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    mlp_bias: bool = False
    aligner_ffn_mult: int | None = 4
    aligner_enable_bias: bool | None = True
    aligner_attention_probs_dropout_prob: float | None = 0.1
    aligner_num_add_layers: int | None = 8
    resampler_depth: int | None = 6
    resampler_dim_head: int | None = 64
    resampler_heads: int | None = 8
    resampler_num_latents: int | None = 64
    resampler_ff_mult: int | None = 4
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    bos_token_id: int | None = 128000
    eos_token_id: int | list[int] | None = 128009
    use_cache: bool = False
    tie_word_embeddings: bool = False
    is_decoder: bool | None = False
    add_cross_attention: bool | None = False

    def __post_init__(self, **kwargs):
        if self.protein_encoder_config is None:
            self.protein_encoder_config = SaProtConfig()
            logger.info("`protein_encoder_config` is `None`. Initializing the `SaProtConfig` with default values.")
        elif isinstance(self.protein_encoder_config, dict):
            self.protein_encoder_config = SaProtConfig(**self.protein_encoder_config)
        super().__post_init__(**kwargs)


__all__ = ["EvollaConfig"]
