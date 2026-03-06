# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Pix2Struct model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/pix2struct-base")
class Pix2StructTextConfig(PreTrainedConfig):
    r"""
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.
    relative_attention_max_distance (`int`, *optional*, defaults to 128):
        The maximum distance of the longer sequences for the bucket separation.
    dense_act_fn (`Union[Callable, str]`, *optional*, defaults to `"gelu_new"`):
        The non-linear activation function (function or string).

    Example:

    ```python
    >>> from transformers import Pix2StructTextConfig, Pix2StructTextModel

    >>> # Initializing a Pix2StructTextConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructTextConfig()

    >>> # Initializing a Pix2StructTextModel (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pix2struct_text_model"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "decoder_attention_heads": "num_heads",
        "encoder_attention_heads": "num_heads",
        "encoder_layers": "num_layers",
        "decoder_layers": "num_layers",
    }

    def __init__(
        self,
        vocab_size=50244,
        hidden_size=768,
        d_kv=64,
        d_ff=2048,
        num_layers=12,
        num_heads=12,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        dense_act_fn="gelu_new",
        decoder_start_token_id=0,
        use_cache=False,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=None,
        tie_word_embeddings=False,
        is_decoder=True,
        add_cross_attention=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.use_cache = use_cache

        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.decoder_start_token_id = decoder_start_token_id

        # for backwards compatibility
        self.dense_act_fn = dense_act_fn

        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        super().__init__(**kwargs)


@auto_docstring(checkpoint="google/pix2struct-base")
class Pix2StructVisionConfig(PreTrainedConfig):
    r"""
    dense_act_fn (`Union[Callable, str]`, *optional*, defaults to `"gelu_new"`):
        The non-linear activation function (function or string).
    patch_embed_hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the input patch_embedding layer in the Transformer encoder.
    d_ff (`int`, *optional*, defaults to 2048):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    d_kv (`int`, *optional*, defaults to 64):
        Dimensionality of the key, query, value projections per attention head.
    The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported.
    seq_len (`int`, *optional*, defaults to 4096):
        Maximum sequence length (here number of patches) supported by the model.
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.
    relative_attention_max_distance (`int`, *optional*, defaults to 128):
        The maximum distance (in tokens) to use for each attention layer.

    Example:

    ```python
    >>> from transformers import Pix2StructVisionConfig, Pix2StructVisionModel

    >>> # Initializing a Pix2StructVisionConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructVisionConfig()

    >>> # Initializing a Pix2StructVisionModel (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pix2struct_vision_model"

    def __init__(
        self,
        hidden_size=768,
        patch_embed_hidden_size=768,
        d_ff=2048,
        d_kv=64,
        num_hidden_layers=12,
        num_attention_heads=12,
        dense_act_fn="gelu_new",
        layer_norm_eps=1e-6,
        dropout_rate=0.0,
        attention_dropout=0.0,
        initializer_range=1e-10,
        initializer_factor=1.0,
        seq_len=4096,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.patch_embed_hidden_size = patch_embed_hidden_size
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.dense_act_fn = dense_act_fn
        self.seq_len = seq_len
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.d_kv = d_kv


@auto_docstring(checkpoint="google/pix2struct-base")
class Pix2StructConfig(PreTrainedConfig):
    r"""
    is_vqa (`bool`, *optional*, defaults to `False`):
        Whether the model has been fine-tuned for VQA or not.

    Example:

    ```python
    >>> from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration

    >>> # Initializing a Pix2StructConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructConfig()

    >>> # Initializing a Pix2StructForConditionalGeneration (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Pix2StructConfig from a Pix2StructTextConfig and a Pix2StructVisionConfig

    >>> # Initializing a Pix2Struct text and Pix2Struct vision configuration
    >>> config_text = Pix2StructTextConfig()
    >>> config_vision = Pix2StructVisionConfig()

    >>> config = Pix2StructConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    model_type = "pix2struct"
    sub_configs = {"text_config": Pix2StructTextConfig, "vision_config": Pix2StructVisionConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        initializer_factor=1.0,
        initializer_range=0.02,
        is_vqa=False,
        tie_word_embeddings=False,
        is_encoder_decoder=True,
        **kwargs,
    ):
        if text_config is None:
            text_config = Pix2StructTextConfig(
                {"is_encoder_decoder": is_encoder_decoder, "tie_word_embeddings": tie_word_embeddings}
            )
            logger.info("`text_config` is `None`. initializing the `Pix2StructTextConfig` with default values.")
        elif isinstance(text_config, dict):
            text_config["is_encoder_decoder"] = is_encoder_decoder
            text_config["tie_word_embeddings"] = tie_word_embeddings
            text_config = Pix2StructTextConfig(**text_config)

        if vision_config is None:
            vision_config = Pix2StructVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `Pix2StructVisionConfig` with default values.")
        elif isinstance(vision_config, dict):
            vision_config = Pix2StructVisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config

        self.decoder_start_token_id = self.text_config.decoder_start_token_id
        self.pad_token_id = self.text_config.pad_token_id
        self.eos_token_id = self.text_config.eos_token_id

        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range

        self.text_config.initializer_range = self.initializer_range
        self.vision_config.initializer_range = self.initializer_range

        self.is_vqa = is_vqa
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)


__all__ = ["Pix2StructConfig", "Pix2StructTextConfig", "Pix2StructVisionConfig"]
