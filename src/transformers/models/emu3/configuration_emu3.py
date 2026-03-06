# Copyright 2024 HuggingFace Inc. team. All rights reserved.
#
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


from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="Emu3-community/Emu3-Chat-hf")
class Emu3VQVAEConfig(PreTrainedConfig):
    r"""
    out_channels (`int`, *optional*, defaults to 3):
        Output channel of decoder.
    temporal_downsample_factor (`int`, *optional*, defaults to 4):
        Temporal downsample factor.
    base_channels (`int`, *optional*, defaults to 256):
        Basic channel number of the intermediate blocks.
    channel_multiplier (`list[int]`, *optional*, defaults to `[1, 2, 2, 4]`):
        Channel scaling factor of the intermediate blocks.
    num_res_blocks (`int`, *optional*, defaults to 2):
        Residual block number in each stage.
    embed_dim (`int`, *optional*, defaults to 4):
        Dimension of the quantized vector in codebook.
    attn_resolutions (`list[int]`, *optional*, defaults to `[3]`):
        Stage indices to apply attention.

    ```python
    >>> from transformers import Emu3VQVAE, Emu3VQVAEConfig

    >>> # Initializing a video VQ model of Emu3 configuration
    >>> configuration = Emu3VQVAEConfig()

    >>> # Initializing a model from the Emu3 VQ model style configuration
    >>> model = Emu3VQVAE(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "emu3_vqgan"
    base_config_key = "vq_config"

    def __init__(
        self,
        codebook_size: int = 32768,
        embed_dim: int = 4,
        latent_channels: int = 4,
        double_latent: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        temporal_downsample_factor: int = 4,
        base_channels: int = 256,
        channel_multiplier: list[int] = [1, 2, 2, 4],
        num_res_blocks: int = 2,
        attn_resolutions: list[int] = [3],
        hidden_size: int = 1024,
        num_attention_heads: int = 1,
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.latent_channels = latent_channels
        self.double_latent = double_latent
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_downsample_factor = temporal_downsample_factor
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout


@auto_docstring(checkpoint="Emu3-community/Emu3-Chat-hf")
class Emu3TextConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Emu3Model, Emu3Config

    >>> # Initializing a Emu3-community/Emu3-Chat-hf style configuration
    >>> configuration = Emu3Config()

    >>> # Initializing a model from the Emu3-community/Emu3-Chat-hf style configuration
    >>> model = Emu3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "emu3_text_model"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    def __init__(
        self,
        vocab_size: int = 184622,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 9216,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 151643,
        bos_token_id: int = 151849,
        eos_token_id: int = 151850,
        rope_parameters: RopeParameters | None = None,
        mlp_bias=False,
        attention_bias=False,
        attention_dropout: float = 0.1,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool | None = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.mlp_bias = mlp_bias
        self.attention_bias = attention_bias
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


@auto_docstring(checkpoint="Emu3-community/Emu3-Chat-hf")
class Emu3Config(PreTrainedConfig):
    r"""
    vocabulary_map (`dict`, *optional*):
        A dictionary containing the vocabulary map from the tokenizer. Used to obtain tokens from the image inputs.
    """

    model_type = "emu3"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"text_config": Emu3TextConfig, "vq_config": Emu3VQVAEConfig}

    def __init__(
        self,
        vq_config: dict | Emu3VQVAEConfig = None,
        text_config: dict | Emu3TextConfig = None,
        vocabulary_map: dict[int, int] | None = None,
        tie_word_embeddings: bool | None = False,
        **kwargs,
    ):
        if vq_config is None:
            vq_config = Emu3VQVAEConfig()
        elif isinstance(vq_config, dict):
            vq_config = Emu3VQVAEConfig(**vq_config)

        if text_config is None:
            text_config = Emu3TextConfig()
        elif isinstance(text_config, dict):
            text_config = Emu3TextConfig(**text_config)

        self.vq_config = vq_config
        self.text_config = text_config
        self.vocabulary_map = vocabulary_map
        self.image_token_id = vocabulary_map.get("<image>") if vocabulary_map is not None else None
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(**kwargs)


__all__ = ["Emu3Config", "Emu3TextConfig", "Emu3VQVAEConfig"]
