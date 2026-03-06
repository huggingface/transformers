# Copyright 2022 ABEJA, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""GPTNeoX Japanese model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="EleutherAI/gpt-neox-japanese-2.7b")
class GPTNeoXJapaneseConfig(PreTrainedConfig):
    r"""
    intermediate_multiple_size (`int`, *optional*, defaults to 4):
        Dimension of the "intermediate" layer in the Transformer encoder is calculated by hidden_size *
        intermediate_multiple_size.

    Example:

    ```python
    >>> from transformers import GPTNeoXJapaneseConfig, GPTNeoXJapaneseModel

    >>> # Initializing a GPTNeoXJapanese gpt-neox-japanese-2.7b style configuration
    >>> configuration = GPTNeoXJapaneseConfig()

    >>> # Initializing a model (with random weights) from the gpt-neox-japanese-2.7b style configuration
    >>> model = GPTNeoXJapaneseModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gpt_neox_japanese"

    def __init__(
        self,
        vocab_size: int | None = 32000,
        hidden_size: int | None = 2560,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        intermediate_multiple_size: int | None = 4,
        hidden_act: str | None = "gelu",
        max_position_embeddings: int | None = 2048,
        initializer_range: float | None = 0.02,
        layer_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        bos_token_id: int | None = 31996,
        eos_token_id: int | None = 31999,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_dropout: float | None = 0.1,
        hidden_dropout: float | None = 0.0,
        is_decoder: bool | None = False,
        pad_token_id: int | None = None,
        tie_word_embeddings: bool | None = True,
        **kwargs,
    ):
        self.is_decoder = is_decoder
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_multiple_size = intermediate_multiple_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation=None, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        # Standardize and validate the correctness of rotary position embeddings parameters
        # Model uses non-standard naming for rope params, overwrite!
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rotary_emb_base", self.default_theta))
        self.rope_parameters["partial_rotary_factor"] = kwargs.pop("rotary_pct", 1.0)
        self.standardize_rope_params()
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)
        return kwargs


__all__ = ["GPTNeoXJapaneseConfig"]
