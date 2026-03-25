# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
"""GPTNeoX model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="EleutherAI/gpt-neox-20b")
@strict
class GPTNeoXConfig(PreTrainedConfig):
    r"""
    use_parallel_residual (`bool`, *optional*, defaults to `True`):
        Whether to use a "parallel" formulation in each Transformer layer, which can provide a slight training
        speedup at large scales (e.g. 20B).

    Example:

    ```python
    >>> from transformers import GPTNeoXConfig, GPTNeoXModel

    >>> # Initializing a GPTNeoX gpt-neox-20b style configuration
    >>> configuration = GPTNeoXConfig()

    >>> # Initializing a model (with random weights) from the gpt-neox-20b style configuration
    >>> model = GPTNeoXModel(configuration)  # doctest: +SKIP

    >>> # Accessing the model configuration
    >>> configuration = model.config  # doctest: +SKIP
    ```"""

    model_type = "gpt_neox"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.attention.query_key_value": "colwise",
        "layers.*.attention.dense": "rowwise",
        "layers.*.mlp.dense_h_to_4h": "colwise",
        "layers.*.mlp.dense_4h_to_h": "rowwise",
    }
    base_model_pp_plan = {
        "embed_in": (["input_ids"], ["inputs_embeds"]),
        "emb_dropout": (["inputs_embeds"], ["hidden_states"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "final_layer_norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 50432
    hidden_size: int = 6144
    num_hidden_layers: int = 44
    num_attention_heads: int = 64
    intermediate_size: int = 24576
    hidden_act: str = "gelu"
    attention_dropout: float | int = 0.0
    hidden_dropout: float | int = 0.0
    classifier_dropout: float | int = 0.1
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    pad_token_id: int | None = None
    tie_word_embeddings: bool = False
    use_parallel_residual: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = True
    is_decoder: bool = False

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisible by the number of attention heads! Make sure to update them!"
            )

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        # Standardize and validate the correctness of rotary position embeddings parameters
        # Model uses non-standard naming for rope params, overwrite!
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rotary_emb_base", self.default_theta))
        self.rope_parameters["partial_rotary_factor"] = kwargs.pop("rotary_pct", 0.25)
        self.standardize_rope_params()
        return kwargs


__all__ = ["GPTNeoXConfig"]
