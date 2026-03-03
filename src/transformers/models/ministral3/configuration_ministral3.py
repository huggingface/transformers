# Copyright 2025 Mistral AI and the HuggingFace Inc. team. All rights reserved.
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
"""Ministral model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import logging


logger = logging.get_logger(__name__)


class Ministral3Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Ministral3Model`]. It is used to instantiate an
    Mistral model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the mistralai/Ministral-3-8B-Base-2512, mistralai/Ministral-3-8B-Instruct-2512 or mistralai/Ministral-3-8B-Reasoning-2512.

    [mistralai/Ministral-3-8B-Base-2512](https://huggingface.co/mistralai/Ministral-3-8B-Base-2512)
    [mistralai/Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512)
    [mistralai/Ministral-3-8B-Reasoning-2512](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`Optional`, *optional*, defaults to 131072):
            Vocabulary size of the Ministral3 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`Ministral3Model`].
        hidden_size (`Optional`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        intermediate_size (`Optional`, *optional*, defaults to 14336):
            Dimensionality of the intermediate (feed-forward) layer.
        num_hidden_layers (`Optional`, *optional*, defaults to 34):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`Optional`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`Optional`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA); if
            `num_key_value_heads=1`, the model will use Multi Query Attention (MQA); otherwise GQA is used.
        head_dim (`Optional`, *optional*, defaults to 128):
            The attention head dimension. If not specified, will default to `hidden_size // num_attention_heads`.
        hidden_act (`Optional`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`Optional`, *optional*, defaults to 262144):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`Optional`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`Optional`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`Optional`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`Optional`, *optional*, defaults to 11):
            The id of the padding token.
        bos_token_id (`Optional`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`Optional`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`Optional`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`Union`, *optional*, defaults to `{'type': 'yarn', 'rope_theta': 1000000.0, 'factor': 16.0, 'original_max_position_embeddings': 16384, 'beta_fast': 32.0, 'beta_slow': 1.0, 'mscale_all_dim': 1.0, 'mscale': 1.0, 'llama_4_scaling_beta': 0.1}`):
            Dictionary containing the configuration parameters for the RoPE embeddings, including optional Yarn scaling
            settings such as `factor`, `original_max_position_embeddings`, `mscale`, and `llama_4_scaling_beta`.
        sliding_window (`Optional`, *optional*):
            Sliding window attention window size. If `None`, full attention is used.
        attention_dropout (`Optional`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    Example:

    ```python
    >>> from transformers import Ministral3Config, Ministral3ForCausalLM, Mistral3Config, Mistral3ForConditionalGeneration, PixtralVisionConfig

    >>> # Initializing a Pixtral-vision config
    >>> vision_config = PixtralVisionConfig()

    >>> # Initializing a Ministral3 config
    >>> text_config = Ministral3Config()

    >>> # Initializing a Mistral3 configuration
    >>> configuration = Mistral3Config(vision_config, text_config)

    >>> # Initializing a model from the Ministral3 configuration
    >>> text_model = Ministral3ForCausalLM(text_config)

    >>> # Initializing a model from the Mistral3 configuration
    >>> model = Mistral3ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ministral3"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `MistralModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 131072,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 14336,
        num_hidden_layers: int | None = 34,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 262144,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = 11,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        sliding_window: int | None = None,
        attention_dropout: float | None = 0.0,
        **kwargs,
    ):
        if rope_parameters is None:
            rope_parameters = {
                "type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 16384,
                "max_position_embeddings": max_position_embeddings,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale_all_dim": 1.0,
                "mscale": 1.0,
                "llama_4_scaling_beta": 0.1,
            }

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout

        if "layer_types" in kwargs:
            logger.warning_once(
                "Detected Mistral model with layer_types. Consider using AutoModel or Ministral classes instead to enable alternating attention compatibility."
            )

        self.rope_parameters = rope_parameters
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(
            ignore_keys_at_rope_validation={"llama_4_scaling_beta", "max_position_embeddings"},
            **kwargs,
        )


__all__ = ["Ministral3Config"]
