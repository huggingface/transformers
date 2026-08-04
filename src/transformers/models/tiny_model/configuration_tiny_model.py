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
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="noanabeshima/tiny_model")
@strict
class TinyModelConfig(PreTrainedConfig):
    r"""
    attention_output_bias (`bool`, *optional*, defaults to `True`):
        Whether to use a bias in the attention output projection.
    mlp_bias (`bool`, *optional*, defaults to `True`):
        Whether to use biases in the feed-forward projections.
    lm_head_bias (`bool`, *optional*, defaults to `True`):
        Whether to use a bias in the language-modeling head.
    embedding_initializer_range (`float`, *optional*, defaults to 0.0001):
        Standard deviation used to initialize token and position embeddings.
    """

    model_type = "tiny_model"

    vocab_size: int = 10_000
    hidden_size: int = 768
    intermediate_size: int = 3_072
    num_hidden_layers: int = 4
    num_attention_heads: int = 16
    max_position_embeddings: int = 256
    hidden_act: str = "relu"
    attention_bias: bool = False
    attention_output_bias: bool = True
    mlp_bias: bool = True
    lm_head_bias: bool = True
    initializer_range: float = 0.02
    embedding_initializer_range: float = 1e-4
    bos_token_id: int | None = 9_996
    eos_token_id: int | list[int] | None = 9_997
    pad_token_id: int | None = 9_998
    tie_word_embeddings: bool = False

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )


__all__ = ["TinyModelConfig"]
