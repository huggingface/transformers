# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
import math
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out
from torchvision.ops import roi_align

from transformers.models.siglip2.configuration_siglip2 import (
    Siglip2Config,
    Siglip2TextConfig,
    Siglip2VisionConfig,
)
from transformers.models.siglip2.modeling_siglip2 import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    Siglip2Attention,
    Siglip2EncoderLayer,
    Siglip2MLP,
    Siglip2Model,
    Siglip2MultiheadAttentionPoolingHead,
    Siglip2Output,
    Siglip2PreTrainedModel,
    Siglip2TextEmbeddings,
    Siglip2TextModel,
    Siglip2TextOutput,
    Siglip2TextTransformer,
    Siglip2VisionEmbeddings,
    Siglip2VisionModel,
    Siglip2VisionOutput,
    Siglip2VisionTransformer,
)

from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    filter_out_non_signature_kwargs,
)
from ...utils.generic import check_model_inputs


class Fgclip2TextConfig(Siglip2TextConfig):
    r"""
    This is the configuration class to store the configuration of a [`Fgclip2TextModel`]. It is used to instantiate a
    Fgclip2 text encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text encoder of the Fgclip2
    [qihoo360/fg-clip2-base](https://huggingface.co/qihoo360/fg-clip2-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Fgclip2 text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`Fgclip2Model`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 64):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        pad_token_id (`int`, *optional*, defaults to 1):
            The id of the padding token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 49406):
            The id of the beginning-of-sequence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 49407):
            The id of the end-of-sequence token in the vocabulary.
        projection_size (`int`, *optional*, defaults to `hidden_size`):
            The size of the projection head.
        keep_len (`int`, *optional*, defaults to 20):
            When processing long texts, the retained tokens are used for handling short text lengths.
            For details, please refer to the FG-CLIP 'https://arxiv.org/abs/2505.05071' paper.
        longtext_len (`int`, *optional*, defaults to 196):
            The maximum number of tokens in long texts that can be processed

    Example:

    ```python
    >>> from transformers import Fgclip2TextConfig, Fgclip2TextModel

    >>> # Initializing a Fgclip2TextConfig with qihoo360/fg-clip2-base style configuration
    >>> configuration = Fgclip2TextConfig()

    >>> # Initializing a Fgclip2TextModel (with random weights) from the qihoo360/fg-clip2-base style configuration
    >>> model = Fgclip2TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fgclip2_text_model"
    base_config_key = "text_config"

    # Update: add `keep_len` and `longtext_len`
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=64,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        projection_size=None,
        keep_len=20,
        longtext_len=196,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.projection_size = projection_size if projection_size is not None else hidden_size
        self.keep_len = keep_len
        self.longtext_len = longtext_len


class Fgclip2VisionConfig(Siglip2VisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`Fgclip2VisionModel`]. It is used to instantiate a
    Fgclip2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Fgclip2
    [qihoo360/fg-clip2-base](https://huggingface.co/qihoo360/fg-clip2-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        num_patches (`int`, *optional*, defaults to 256):
            The number of patches in the image with the size of (`patch_size`, `patch_size`).
            The image is resized to fill maximum of this number of patches, and to preserve
            the aspect ratio. In case the resulted number of patches is lower, the image is
            padded in "patch" dimension.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    Example:

    ```python
    >>> from transformers import Fgclip2VisionConfig, Fgclip2VisionModel

    >>> # Initializing a Fgclip2VisionConfig with qihoo360/fg-clip2-base style configuration
    >>> configuration = Fgclip2VisionConfig()

    >>> # Initializing a Fgclip2VisionModel (with random weights) from the qihoo360/fg-clip2-base style configuration
    >>> model = Fgclip2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    pass


class Fgclip2Config(Siglip2Config):
    r"""
    [`Fgclip2Config`] is the configuration class to store the configuration of a [`Fgclip2Model`]. It is used to
    instantiate a Fgclip2 model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Fgclip2
    [qihoo360/fg-clip2-base](https://huggingface.co/qihoo360/fg-clip2-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Fgclip2TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Fgclip2VisionConfig`].
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import Fgclip2Config, Fgclip2Model

    >>> # Initializing a Fgclip2Config with qihoo360/fg-clip2-base style configuration
    >>> configuration = Fgclip2Config()

    >>> # Initializing a Fgclip2Model (with random weights) from the qihoo360/fg-clip2-base style configuration
    >>> model = Fgclip2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Fgclip2Config from a Fgclip2TextConfig and a Fgclip2VisionConfig
    >>> from transformers import Fgclip2TextConfig, Fgclip2VisionConfig

    >>> # Initializing a Fgclip2Text and Fgclip2Vision configuration
    >>> config_text = Fgclip2TextConfig()
    >>> config_vision = Fgclip2VisionConfig()

    >>> config = Fgclip2Config.from_text_vision_configs(config_text, config_vision)
    ```"""

    pass


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)


def trunc_normal_tf_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \text{mean} \\leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsequently scaled and shifted by the mean and std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


class Fgclip2VisionOutput(Siglip2VisionOutput):
    pass


class Fgclip2TextOutput(Siglip2TextOutput):
    pass


class Fgclip2Attention(Siglip2Attention):
    pass


class Fgclip2MLP(Siglip2MLP):
    pass


class Fgclip2EncoderLayer(Siglip2EncoderLayer):
    pass


class Fgclip2Output(Siglip2Output):
    pass


class Fgclip2VisionEmbeddings(Siglip2VisionEmbeddings):
    pass


class Fgclip2VisionTransformer(Siglip2VisionTransformer):
    pass


class Fgclip2PreTrainedModel(Siglip2PreTrainedModel):
    config: Fgclip2Config
    base_model_prefix = "fgclip2"
    input_modalities = ["image", "text"]
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "Fgclip2TextEmbeddings",
        "Fgclip2VisionEmbeddings",
        "Fgclip2EncoderLayer",
        "Fgclip2MultiheadAttentionPoolingHead",
    ]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": Fgclip2EncoderLayer,
        "attentions": Fgclip2Attention,
    }

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, Fgclip2VisionEmbeddings):
            width = (
                self.config.vision_config.hidden_size
                if isinstance(self.config, Fgclip2Config)
                else self.config.hidden_size
            )
            nn.init.normal_(module.position_embedding.weight, std=1 / np.sqrt(width))
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, Fgclip2Attention):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, Fgclip2MLP):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, Fgclip2MultiheadAttentionPoolingHead):
            nn.init.xavier_uniform_(module.probe.data)
            nn.init.xavier_uniform_(module.attention.in_proj_weight.data)
            nn.init.zeros_(module.attention.in_proj_bias.data)
        elif isinstance(module, Fgclip2Model):
            logit_scale_init = torch.log(torch.tensor(1.0))
            module.logit_scale.data.fill_(logit_scale_init)
            module.logit_bias.data.zero_()
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Fgclip2TextEmbeddings(Siglip2TextEmbeddings):
    # Update: add `position_embedding_res`, `position_embedding_ori`, `mask1` and `mask2`
    # Enable the model to support long-text retrieval
    def __init__(self, config: Fgclip2TextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        keep_len = config.keep_len
        longtext_len = config.longtext_len

        self.position_embedding_res = nn.Embedding(longtext_len, embed_dim)
        self.position_embedding_ori = nn.Embedding(longtext_len, embed_dim)

        self.mask1 = torch.zeros([longtext_len, 1])
        self.mask1[:keep_len, :] = 1
        self.mask2 = torch.zeros([longtext_len, 1])
        self.mask2[keep_len:, :] = 1

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(longtext_len).expand((1, -1)), persistent=False)

    # Update: add `use_short_position_ids`
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_short_position_ids: Optional[bool] = True,
    ) -> torch.Tensor:
        r"""
        Args:
        use_short_position_ids (`bool`, optional, defaults to `True`):
            If `True`, applies a positional encoding scheme optimized for **short-text processing** and **local-region description processing**,
            such as phrases or simple sentences. Corresponds to the `"short"` and `"box"` walk type.
            Assumes compact semantic structure and local dependency dominance.
        """

        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        if use_short_position_ids:
            position_embeddings = self.position_embedding(position_ids)
            embeddings = inputs_embeds + position_embeddings
        else:
            position_embeddings_res = self.position_embedding_res(position_ids)
            position_embeddings_ori = self.position_embedding_ori(position_ids)
            embeddings = (
                inputs_embeds
                + (position_embeddings_ori * self.mask1.to(inputs_embeds.device))
                .type(inputs_embeds.dtype)
                .to(inputs_embeds.device)
                + (position_embeddings_res * self.mask2.to(inputs_embeds.device))
                .type(inputs_embeds.dtype)
                .to(inputs_embeds.device)
            )

        return embeddings


class Fgclip2TextTransformer(Siglip2TextTransformer):
    # Update: add `walk_type`
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        walk_type: str = "short",  # Modified: Single parameter
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Args:
        walk_type (`str`, optional, defaults to `"short"`):
            The traversal strategy used during feature extraction. Must be one of
            `"short"`, `"box"`, or `"long"`. This controls how contextual information
            is aggregated across the input:
            - `"short"`: Optimized for short-text understanding, focusing on tight semantic coherence
            and direct word interactions. Suitable when the input is a phrase or brief sentence.
            - `"box"`: Designed for local-region description processing, such as grounding in vision-language
            models or processing localized textual descriptions (e.g., object regions or segments).
            Emphasizes dense features within bounded semantic units.
            - `"long"`: Tailored for long-form text processing, enabling modeling of extended dependencies
            and discourse structure. Uses strategies like chunking or hierarchical attention to handle
            longer sequences effectively.
        """
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        # Validate walk_type
        walk_type = walk_type.lower()
        if walk_type not in ["short", "box", "long"]:
            raise ValueError(f"Invalid `walk_type`: {walk_type}. Must be one of 'short', 'box', 'long'.")

        # Convert walk_type to boolean flags for internal logic
        walk_short = walk_type == "short"
        walk_box = walk_type == "box"
        walk_long = walk_type == "long"

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            use_short_position_ids=(not walk_long),
        )
        # note: fgclip2's text model does not use a causal mask, unlike the original CLIP model.
        # expand attention_mask
        uses_flash_attention = "flash" in self.config._attn_implementation
        if uses_flash_attention:
            attention_mask = None
        elif attention_mask is not None and not uses_flash_attention:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        # The model uses the last token's hidden state, which may be padding.
        pooled_output = last_hidden_state[:, -1, :]
        if walk_short:
            assert not walk_box
            assert not walk_long
            temp_pool_out = []
            for i in range(pooled_output.shape[0]):
                temp_pool_out.append(self.head(pooled_output[i : i + 1]))
            pooled_output = torch.cat(temp_pool_out, dim=0)
            # pooled_output = self.head(pooled_output)
        if walk_box:
            assert not walk_short
            assert not walk_long
            pooled_output = pooled_output
        if walk_long:
            assert not walk_short
            assert not walk_box
            pooled_output = pooled_output
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )


class Fgclip2TextModel(Siglip2TextModel):
    # Update: add `walk_type`
    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        walk_type: str = "short",  # Modified: Single parameter
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        Args:
        walk_type (`str`, optional, defaults to `"short"`):
            The traversal strategy used during feature extraction. Must be one of
            `"short"`, `"box"`, or `"long"`. This controls how contextual information
            is aggregated across the input:
            - `"short"`: Optimized for short-text understanding, focusing on tight semantic coherence
            and direct word interactions. Suitable when the input is a phrase or brief sentence.
            - `"box"`: Designed for local-region description processing, such as grounding in vision-language
            models or processing localized textual descriptions (e.g., object regions or segments).
            Emphasizes dense features within bounded semantic units.
            - `"long"`: Tailored for long-form text processing, enabling modeling of extended dependencies
            and discourse structure. Uses strategies like chunking or hierarchical attention to handle
            longer sequences effectively.
        """
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            walk_type=walk_type,  # Modified: Pass single parameter
            **kwargs,
        )


class Fgclip2MultiheadAttentionPoolingHead(Siglip2MultiheadAttentionPoolingHead):
    # Update: The following improvements have been made, ensuring that the precision difference
    # between the output results of batch inference and individual inference remains stable below 1e-5.
    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class Fgclip2VisionModel(Siglip2VisionModel):
    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Fgclip2VisionModel

        >>> model = Fgclip2VisionModel.from_pretrained("qihoo360/fg-clip2-base")
        >>> processor = AutoProcessor.from_pretrained("qihoo360/fg-clip2-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class Fgclip2Model(Siglip2Model):
    # Update: add `dense_feature_head`, `longtext_head` and `boxtext_head`
    def __init__(self, config: Fgclip2Config):
        super().__init__(config)

        if not isinstance(config.text_config, Fgclip2TextConfig):
            raise TypeError(
                "config.text_config is expected to be of type Fgclip2TextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, Fgclip2VisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type Fgclip2VisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        # First, initialize the text and vision models with proper attention implementation
        text_model = Fgclip2TextModel._from_config(text_config)
        vision_model = Fgclip2VisionModel._from_config(vision_config)

        # Second, get the text and vision submodules (for backward compatibility)
        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model
        self.dense_feature_head = Fgclip2MultiheadAttentionPoolingHead(vision_config)
        self.embed_dim = text_config.hidden_size
        self.longtext_head = nn.Linear(self.embed_dim, self.embed_dim)
        self.boxtext_head = nn.Linear(self.embed_dim, self.embed_dim)

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        # Initialize weights and apply final processing
        self.post_init()

    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.

        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`Fgclip2VisionModel`].

        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, AutoModel
        >>> from transformers.image_utils import load_image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> model = AutoModel.from_pretrained("qihoo360/fg-clip2-base")
        >>> processor = AutoProcessor.from_pretrained("qihoo360/fg-clip2-base")

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     image_features = model.get_image_features(**inputs)
        ```
        """
        super().get_image_features(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

    # NOTE: Fgclip2Model uses Pretrained backbones, so we don't need to add `check_model_inputs` here
    # Update: add `walk_type`
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        walk_type: str = "short",
    ) -> Fgclip2Output:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
            Tensor containing the spatial dimensions (height, width) of the input images.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        walk_type (`str`, optional, defaults to `"short"`):
                The traversal strategy used during feature extraction. Must be one of
                `"short"`, `"box"`, or `"long"`. This controls how contextual information
                is aggregated across the input:
                - `"short"`: Optimized for short-text understanding, focusing on tight semantic coherence
                and direct word interactions. Suitable when the input is a phrase or brief sentence.
                - `"box"`: Designed for local-region description processing, such as grounding in vision-language
                models or processing localized textual descriptions (e.g., object regions or segments).
                Emphasizes dense features within bounded semantic units.
                - `"long"`: Tailored for long-form text processing, enabling modeling of extended dependencies
                and discourse structure. Uses strategies like chunking or hierarchical attention to handle
                longer sequences effectively.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("qihoo360/fg-clip2-base")
        >>> processor = AutoProcessor.from_pretrained("qihoo360/fg-clip2-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> texts = ["a photo of two cats", "a photo of a cat"]
        >>> # important: we pass `padding=max_length` since the model was trained with this
        >>> inputs = processor(text=texts, images=image, padding="max_length", max_length=64, truncation=True, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> logits_per_image = outputs.logits_per_image
        >>> probs = torch.sigmoid(logits_per_image) # these are the probabilities
        >>> print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
        53.2% that image 0 is 'a photo of two cats'
        ```
        """
        walk_type = walk_type.lower()

        if walk_type not in ["short", "box", "long"]:
            raise ValueError(f"Invalid `walk_type`: {walk_type}. Must be one of 'short', 'box', 'long'.")

        walk_short = walk_type == "short"
        walk_box = walk_type == "box"
        walk_long = walk_type == "long"

        # Use Fgclip2 model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            walk_type=walk_type,
        )

        image_embeds = vision_outputs.pooler_output

        if walk_short:
            text_embeds = text_outputs.pooler_output

        if walk_box:
            text_embeds = self.boxtext_head(text_outputs.pooler_output)

        if walk_long:
            text_embeds = self.longtext_head(text_outputs.pooler_output)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))

        logit_scale, logit_bias = self.logit_scale.to(text_embeds.device), self.logit_bias.to(text_embeds.device)
        logits_per_text = logits_per_text * logit_scale.exp() + logit_bias

        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

        return Fgclip2Output(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    # Update: add `walk_type`
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        walk_type: str = "short",
    ) -> torch.FloatTensor:
        r"""
        Extracts feature representations from the input text.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The token IDs of the input sequence, as generated by the tokenizer.
            attention_mask (`torch.Tensor`, optional, of shape `(batch_size, sequence_length)`):
                A mask indicating which tokens are valid (1) and which are padding (0).
                If not provided, all tokens are assumed to be valid.
            position_ids (`torch.Tensor`, optional, of shape `(batch_size, sequence_length)`):
                Position indices for each token in the sequence. If not provided,
                positions are automatically constructed based on `input_ids`.
            walk_type (`str`, optional, defaults to `"short"`):
                The traversal strategy used during feature extraction. Must be one of
                `"short"`, `"box"`, or `"long"`. This controls how contextual information
                is aggregated across the input:
                - `"short"`: Optimized for short-text understanding, focusing on tight semantic coherence
                and direct word interactions. Suitable when the input is a phrase or brief sentence.
                - `"box"`: Designed for local-region description processing, such as grounding in vision-language
                models or processing localized textual descriptions (e.g., object regions or segments).
                Emphasizes dense features within bounded semantic units.
                - `"long"`: Tailored for long-form text processing, enabling modeling of extended dependencies
                and discourse structure. Uses strategies like chunking or hierarchical attention to handle
                longer sequences effectively.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, hidden_size)` or `(batch_size, sequence_length, hidden_size)`:
                The extracted feature tensor representing the input text. The output shape depends on
                whether a pooled representation or per-token embeddings are returned.
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("qihoo360/fg-clip2-base")
        >>> tokenizer = AutoTokenizer.from_pretrained("qihoo360/fg-clip2-base")

        >>> # important: make sure to set padding="max_length" as that's how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")
        >>> with torch.no_grad():
        ...     text_features = model.get_text_features(**inputs, walk_type="short")
        ```"""

        walk_type = walk_type.lower()

        if walk_type not in ["short", "box", "long"]:
            raise ValueError(f"Invalid `walk_type`: {walk_type}. Must be one of 'short', 'box', 'long'.")

        walk_short = walk_type == "short"
        walk_box = walk_type == "box"
        walk_long = walk_type == "long"

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            walk_type=walk_type,
        )

        if walk_short:
            pooled_output = text_outputs.pooler_output

        if walk_box:
            pooled_output = self.boxtext_head(text_outputs.pooler_output)

        if walk_long:
            pooled_output = self.longtext_head(text_outputs.pooler_output)

        return pooled_output

    # New function: Acquire dense visual features of images with support for dynamic resolution
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_image_dense_feature(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        r"""
        Extract dense visual features from input images by forwarding through the vision backbone.

        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, max_num_patches, num_channels * patch_size * patch_size)
            pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
                Mask to avoid performing attention on padding pixel indices.
            spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
                Tensor containing the spatial dimensions (height, width) of the input images.

        Returns:
            `torch.FloatTensor` of shape  `(batch_size, max_num_patches, hidden_size)`:
        """

        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

        probe = vision_outputs.last_hidden_state
        hidden_state = vision_outputs.last_hidden_state
        attention_mask = pixel_attention_mask

        if attention_mask is not None:
            target_len, source_len = probe.shape[1], hidden_state.shape[1]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype, target_len)
            attention_mask = attention_mask.repeat(1, self.dense_feature_head.num_heads, 1, 1)
            attention_mask = attention_mask.reshape(-1, target_len, source_len)

        hidden_state = self.dense_feature_head.attention(probe, hidden_state, hidden_state, attn_mask=attention_mask)[
            0
        ]
        residual = hidden_state
        hidden_state = self.dense_feature_head.layernorm(hidden_state)
        hidden_state = residual + self.dense_feature_head.mlp(hidden_state)
        feature_map = hidden_state

        return feature_map

    # New function: Acquire local features of images, applicable to retrieval, classification, and localization, with support for dynamic resolution
    @filter_out_non_signature_kwargs()
    @auto_docstring
    def get_image_region_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        image_sizes: Optional[list[tuple]] = None,
        region_infos: Optional[list[list[list[float]]]] = None,
    ) -> list[torch.FloatTensor]:
        r"""
        Extract region-of-interest (RoI) features from images using RoI Align.
        This method supports batched processing of variable-sized images and allows feature extraction
        from user-specified image regions.

        The input can be either a full image with corresponding region coordinates.
        Features are extracted per region (e.g., bounding boxes), making this function suitable for tasks such as
        object detection, referring expression grounding, or vision-language alignment.

        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, max_num_patches, num_channels * patch_size * patch_size)
            pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
                Mask to avoid performing attention on padding pixel indices.
            spatial_shapes (`torch.LongTensor` of shape `(batch_size, 2)`):
                Tensor containing the spatial dimensions (height, width) of the input images.
            image_sizes (`List[tuple]`, optional, each tuple of form `(int, int)`):
                Original size (height, width) of each image in the batch before padding or resizing.
                Required for accurate coordinate projection when region_infos are defined in original image space.
            region_infos (`List[List[List[float]]]`, optional):
                Bounding box coordinates for regions of interest in each image. Format:
                - Outer list: length `batch_size`
                - Middle list: number of regions per image
                - Inner list: each contains `[x_min, y_min, x_max, y_max]` in **absolute pixel coordinates**
                relative to the original image size (as specified in `image_sizes`).
                These boxes are projected to feature map space using `image_sizes` and `spatial_shapes`,
                then used to pool features via RoI Align or equivalent.

        Returns:
            `List[torch.FloatTensor]`:
                A list of length `batch_size`, where each element is a tensor of shape
                `(num_boxes, hidden_dim)` containing the extracted visual features for each region
                in the corresponding image.
        Example::
            >>> # For a batch of 2 images
            >>> region_features = model.get_image_region_features(
            >>>     pixel_values=pixel_values,
            >>>     image_sizes=[(640, 480), (480, 640)],
            >>>     region_infos=[
            >>>         [[100, 100, 200, 200], [300, 300, 400, 400]],  # 2 boxes in first image
            >>>         [[50, 50, 150, 150]]                          # 1 box in second image
            >>>     ]
            >>> )
            >>> print(region_features[0].shape)  # torch.Size([2, hidden_dim])
            >>> print(region_features[1].shape)  # torch.Size([1, hidden_dim])

        """
        if region_infos is None or len(region_infos) == 0:
            return []

        # Get dense feature maps: (B, N, D)
        dense_feature_map = self.get_image_dense_feature(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        bs, _, hidden_dim = dense_feature_map.shape

        all_region_features = []

        for i in range(bs):
            h, w = spatial_shapes[i].tolist()
            img_h, img_w = image_sizes[i]
            bboxes = region_infos[i]

            if not bboxes:
                all_region_features.append(torch.empty(0, hidden_dim, device=dense_feature_map.device))
                continue

            # Reshape to (1, C, H', W')
            num_valid = h * w
            feat_seq = dense_feature_map[i, :num_valid]  # (num_valid, D)
            feat_map = feat_seq.view(h, w, hidden_dim).permute(2, 0, 1).unsqueeze(0)  # (1, D, H', W')

            # Normalize bboxes to feature map coordinates
            rois = []
            for x1, y1, x2, y2 in bboxes:
                nx1 = (x1 / img_w) * w
                ny1 = (y1 / img_h) * h
                nx2 = (x2 / img_w) * w
                ny2 = (y2 / img_h) * h
                rois.append([0, nx1, ny1, nx2, ny2])  #
            rois_tensor = torch.tensor(rois, dtype=torch.float32, device=feat_map.device)  # (N, 5)

            # RoI Align on single image
            pooled = roi_align(
                input=feat_map,
                boxes=rois_tensor,
                output_size=(1, 1),
                spatial_scale=1.0,
                sampling_ratio=-1,
                aligned=True,
            )  # (N, D, 1, 1)
            region_feats = pooled.squeeze(-1).squeeze(-1)  # (N, D)

            all_region_features.append(region_feats)

        return all_region_features


__all__ = [
    "Fgclip2Config",
    "Fgclip2TextConfig",
    "Fgclip2VisionConfig",
    "Fgclip2Model",
    "Fgclip2PreTrainedModel",
    "Fgclip2TextModel",
    "Fgclip2VisionModel",
]
