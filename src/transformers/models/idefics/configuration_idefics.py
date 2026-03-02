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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="HuggingFaceM4/idefics-9b")
class IdeficsVisionConfig(PreTrainedConfig):
    model_type = "idefics_vision"
    attribute_map = {
        "hidden_size": "embed_dim",
    }

    def __init__(
        self,
        embed_dim=768,
        image_size=224,
        intermediate_size=5120,
        patch_size=14,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_channels=3,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.hidden_act = hidden_act

        super().__init__(**kwargs)


@auto_docstring(checkpoint="HuggingFaceM4/idefics-9b")
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

    def __init__(
        self,
        use_resampler=False,
        resampler_n_latents=64,
        resampler_depth=6,
        resampler_n_heads=16,
        resampler_head_dim=96,
        qk_layer_norms_perceiver=False,
        **kwargs,
    ):
        self.use_resampler = use_resampler
        self.resampler_n_latents = resampler_n_latents
        self.resampler_depth = resampler_depth
        self.resampler_n_heads = resampler_n_heads
        self.resampler_head_dim = resampler_head_dim
        self.qk_layer_norms_perceiver = qk_layer_norms_perceiver

        super().__init__(**kwargs)


@auto_docstring(checkpoint="HuggingFaceM4/idefics-9b")
class IdeficsConfig(PreTrainedConfig):
    r"""
    alpha_initializer (`str`, *optional*, defaults to `"zeros"`):
        Initialization type for the alphas.
    alphas_initializer_range (`float`, *optional*, defaults to 0.0):
        The standard deviation of the truncated_normal_initializer for initializing the alphas in the Gated Cross
        Attention.
    alpha_type (`str`, *optional*, defaults to `"float"`):
        Whether the gating alphas should be vectors or single floats.
    additional_vocab_size (`int`, *optional*, defaults to 0):
        Additional vocabulary size of the model, typically for the special "<img>" token. Additional vocab tokens
        are always trainable whereas regular vocab tokens can be frozen or not.
    cross_layer_interval (`int`, *optional*, default to 1):
        Interval for cross attention (from text to image) layers.
    qk_layer_norms (`bool`, *optional*, defaults to `False`): Whether to add layer norm after q and k
    freeze_text_layers (`bool`, *optional*, defaults to `True`): Whether to freeze text layers
    freeze_text_module_exceptions (`bool`, *optional*, defaults to `[]`):
        Exceptions to freezing text layers when `freeze_text_layers` is `True`
    freeze_lm_head (`bool`, *optional*, defaults to `False`): Whether to freeze lm head
    freeze_vision_layers (`bool`, *optional*, defaults to `True`):  Whether to freeze vision layers
    freeze_vision_module_exceptions (`bool`, *optional*, defaults to `[]`):
        Exceptions to freezing vision layers when `freeze_vision_layers` is `True`
    use_resampler (`bool`, *optional*, defaults to `False`): Whether to use the Resampler
    perceiver_config (`IdeficsPerceiverConfig`,  *optional*): Custom perceiver config or dict

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

    def __init__(
        self,
        vocab_size=32000,
        additional_vocab_size=0,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        dropout=0.0,
        hidden_act="silu",
        initializer_range=0.02,
        alpha_initializer="zeros",
        alphas_initializer_range=0.0,
        alpha_type="float",
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        cross_layer_interval=1,
        qk_layer_norms=False,
        freeze_text_layers=True,
        freeze_text_module_exceptions=[],
        freeze_lm_head=False,
        freeze_vision_layers=True,
        freeze_vision_module_exceptions=[],
        use_resampler=False,
        vision_config=None,
        perceiver_config=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.alpha_initializer = alpha_initializer
        self.alphas_initializer_range = alphas_initializer_range
        self.alpha_type = alpha_type
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.cross_layer_interval = cross_layer_interval
        self.qk_layer_norms = qk_layer_norms
        self.freeze_vision_layers = freeze_vision_layers

        self.freeze_text_layers = freeze_text_layers
        self.freeze_text_module_exceptions = freeze_text_module_exceptions
        self.freeze_vision_module_exceptions = freeze_vision_module_exceptions
        self.freeze_lm_head = freeze_lm_head

        self.use_resampler = use_resampler

        if perceiver_config is None:
            self.perceiver_config = IdeficsPerceiverConfig()
        elif isinstance(perceiver_config, dict):
            self.perceiver_config = IdeficsPerceiverConfig(**perceiver_config)
        elif isinstance(perceiver_config, IdeficsPerceiverConfig):
            self.perceiver_config = perceiver_config

        if vision_config is None:
            self.vision_config = IdeficsVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = IdeficsVisionConfig(**vision_config)
        elif isinstance(vision_config, IdeficsVisionConfig):
            self.vision_config = vision_config

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)

        # IMPORTANT: Do not do any __init__ args-based checks in the constructor, since
        # PreTrainedConfig.from_dict first instantiates the class with the config dict and only then
        # updates the config object with `kwargs` from from_pretrained, so during the instantiation
        # of this object many attributes have default values and haven't yet been overridden.
        # Do any required checks inside `from_pretrained` once the superclass' `from_pretrained` was run.


__all__ = ["IdeficsConfig"]
