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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="mistralai/Ministral-3-8B-Base-2512")
@strict
class Ministral3Config(PreTrainedConfig):
    r"""
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
    ignore_keys_at_rope_validation = {"llama_4_scaling_beta", "max_position_embeddings"}

    vocab_size: int = 131072
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 34
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 11
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    sliding_window: int | None = None
    attention_dropout: float | int = 0.0

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 16384,
                "max_position_embeddings": self.max_position_embeddings,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale_all_dim": 1.0,
                "mscale": 1.0,
                "llama_4_scaling_beta": 0.1,
            }

        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if "layer_types" in kwargs:
            logger.warning_once(
                "Detected Mistral model with layer_types. Consider using AutoModel or Ministral classes instead to enable alternating attention compatibility."
            )

        super().__post_init__(**kwargs)


__all__ = ["Ministral3Config"]
