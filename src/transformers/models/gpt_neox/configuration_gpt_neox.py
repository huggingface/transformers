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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="EleutherAI/gpt-neox-20b")
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

    def __init__(
        self,
        vocab_size: int | None = 50432,
        hidden_size: int | None = 6144,
        num_hidden_layers: int | None = 44,
        num_attention_heads: int | None = 64,
        intermediate_size: int | None = 24576,
        hidden_act: str | None = "gelu",
        attention_dropout: float | None = 0.0,
        hidden_dropout: float | None = 0.0,
        classifier_dropout: float | None = 0.1,
        max_position_embeddings: int | None = 2048,
        initializer_range: float | None = 0.02,
        layer_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        bos_token_id: int | None = 0,
        eos_token_id: int | None = 2,
        pad_token_id: int | None = None,
        tie_word_embeddings: bool | None = False,
        use_parallel_residual: bool | None = True,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = True,
        is_decoder: bool | None = False,
        **kwargs,
    ):
        self.is_decoder = is_decoder
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.use_parallel_residual = use_parallel_residual
        self.attention_bias = attention_bias
        self.rope_parameters = rope_parameters
        self.tie_word_embeddings = tie_word_embeddings

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisible by the number of attention heads! Make sure to update them!"
            )
        super().__init__(**kwargs)

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation=None, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        # Standardize and validate the correctness of rotary position embeddings parameters
        # Model uses non-standard naming for rope params, overwrite!
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rotary_emb_base", self.default_theta))
        self.rope_parameters["partial_rotary_factor"] = kwargs.pop("rotary_pct", 0.25)
        self.standardize_rope_params()
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)
        return kwargs


__all__ = ["GPTNeoXConfig"]
