# Copyright 2022 The OFA-Sys Team Authors and The HuggingFace Team. All rights reserved.
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
"""PyTorch Chinese-CLIP model."""

from collections.abc import Callable

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..align.modeling_align import (
    AlignTextEmbeddings,
    AlignTextEncoder,
    AlignTextLayer,
    AlignTextSelfAttention,
    eager_attention_forward,
)
from ..bert.modeling_bert import BertMLP, BertPooler
from ..clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from ..clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPModel,
    CLIPOutput,
    CLIPPreTrainedModel,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    _get_vector_norm,
    image_text_contrastive_loss,
)


@auto_docstring(checkpoint="OFA-Sys/chinese-clip-vit-base-patch16")
@strict
class ChineseCLIPTextConfig(CLIPTextConfig):
    r"""
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling [`ChineseCLIPModel`].

    Example:

    ```python
    >>> from transformers import ChineseCLIPTextConfig, ChineseCLIPTextModel

    >>> # Initializing a ChineseCLIPTextConfig with OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> configuration = ChineseCLIPTextConfig()

    >>> # Initializing a ChineseCLIPTextModel (with random weights) from the OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> model = ChineseCLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    vocab_size: int = 30522
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    bos_token_id: int | None = 0
    eos_token_id: int | None = None
    attention_dropout = AttributeError()
    projection_dim = AttributeError()


@auto_docstring(checkpoint="OFA-Sys/chinese-clip-vit-base-patch16")
@strict
class ChineseCLIPVisionConfig(CLIPVisionConfig):
    r"""
    Example:
    ```python
    >>> from transformers import ChineseCLIPVisionConfig, ChineseCLIPVisionModel

    >>> # Initializing a ChineseCLIPVisionConfig with OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> configuration = ChineseCLIPVisionConfig()

    >>> # Initializing a ChineseCLIPVisionModel (with random weights) from the OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> model = ChineseCLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""


@auto_docstring(checkpoint="OFA-Sys/chinese-clip-vit-base-patch16")
@strict
class ChineseCLIPConfig(CLIPConfig):
    r"""
    Example:

    ```python
    >>> from transformers import ChineseCLIPConfig, ChineseCLIPModel

    >>> # Initializing a ChineseCLIPConfig with OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> configuration = ChineseCLIPConfig()

    >>> # Initializing a ChineseCLIPModel (with random weights) from the OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> model = ChineseCLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a ChineseCLIPConfig from a ChineseCLIPTextConfig and a ChineseCLIPVisionConfig

    >>> # Initializing a ChineseCLIPTextConfig and ChineseCLIPVisionConfig configuration
    >>> config_text = ChineseCLIPTextConfig()
    >>> config_vision = ChineseCLIPVisionConfig()

    >>> config = ChineseCLIPConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    initializer_range: float = 0.02


class ChineseCLIPOutput(CLIPOutput):
    pass


class ChineseCLIPTextEmbeddings(AlignTextEmbeddings):
    pass


class ChineseCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    pass


class ChineseCLIPTextAttention(AlignTextSelfAttention):
    """Self-attention with its output projection, in the shape used across the library.

    The text tower used to spread this over `ChineseCLIPTextSelfAttention` (q/k/v) and
    `ChineseCLIPTextSelfOutput` (the output projection plus the residual LayerNorm). The projection
    lives here as `o_proj`; the LayerNorm and residual belong to `ChineseCLIPTextLayer`.

    It stays a reduced copy of ALIGN's text attention rather than a `BertAttention` subclass for two
    reasons: `ChineseCLIPTextConfig` descends from `CLIPTextConfig` and has no `is_decoder`, and this
    file's `eager_attention_forward` must remain the CLIP/ALIGN one -- it upcasts the softmax to
    float32, which `BertAttention`'s does not, and the same helper is shared with the vision tower.
    """

    def __init__(self, config):
        super().__init__(config)

        del self.query
        del self.key
        del self.value
        # ALIGN keeps the dropout probability twice, once as a module and once as a bare float.
        # Read it off the module, like `BertAttention` does, so `attention_dropout` means the same
        # thing here as it does on `ChineseCLIPTextLayer` (an `nn.Dropout`) rather than two things.
        del self.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.o_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout.p,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class ChineseCLIPVisionAttention(CLIPAttention):
    pass


class ChineseCLIPTextMLP(BertMLP):
    pass


class ChineseCLIPVisionMLP(CLIPMLP):
    pass


class ChineseCLIPTextLayer(AlignTextLayer):
    """Post-norm transformer layer: `LayerNorm(x + sublayer(x))` for each sublayer.

    The norms and residuals live here rather than inside the sublayers, so the layer's topology is
    stated in one place -- the same arrangement `BertLayer` now uses, minus the decoder and
    cross-attention branches the text tower never takes. The arithmetic is unchanged from the
    previous five-class arrangement.
    """

    def __init__(self, config):
        super().__init__(config)

        del self.attention
        del self.intermediate
        del self.output

        self.self_attn = ChineseCLIPTextAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlp = ChineseCLIPTextMLP(config)
        self.post_feedforward_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states + self.attention_dropout(attn_output))

        return apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, hidden_states
        )

    def feed_forward_chunk(self, hidden_states):
        return self.post_feedforward_layernorm(hidden_states + self.mlp(hidden_states))


class ChineseCLIPVisionLayer(CLIPEncoderLayer):
    def __init__(self, config: ChineseCLIPConfig):
        super().__init__()
        self.self_attn = ChineseCLIPVisionAttention(config)
        self.mlp = ChineseCLIPVisionMLP(config)


class ChineseCLIPTextPooler(BertPooler):
    pass


@auto_docstring
class ChineseCLIPPreTrainedModel(CLIPPreTrainedModel):
    _no_split_modules = [
        "ChineseCLIPVisionEmbeddings",
        "ChineseCLIPTextEmbeddings",
        "ChineseCLIPTextLayer",
        "ChineseCLIPVisionLayer",
    ]
    _can_record_outputs = {
        "hidden_states": ChineseCLIPVisionLayer,
        "attentions": ChineseCLIPVisionAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        PreTrainedModel._init_weights(self, module)
        factor = self.config.initializer_factor
        if isinstance(module, ChineseCLIPVisionEmbeddings):
            init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
            init.copy_(module.position_ids, torch.arange(module.num_positions).expand((1, -1)))
        elif isinstance(module, ChineseCLIPTextEmbeddings):
            init.normal_(module.word_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.position_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.token_type_embeddings.weight, mean=0.0, std=self.config.initializer_range)
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
            init.zeros_(module.token_type_ids)
            for embedding in [module.word_embeddings, module.position_embeddings, module.token_type_embeddings]:
                if embedding.padding_idx is not None:
                    init.zeros_(embedding.weight[embedding.padding_idx])
        elif isinstance(module, ChineseCLIPVisionAttention):
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            init.normal_(module.q_proj.weight, std=in_proj_std)
            init.normal_(module.k_proj.weight, std=in_proj_std)
            init.normal_(module.v_proj.weight, std=in_proj_std)
            init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, ChineseCLIPVisionMLP):
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            init.normal_(module.fc1.weight, std=fc_std)
            init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, ChineseCLIPModel):
            init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * factor,
            )
            init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * factor,
            )


class ChineseCLIPTextEncoder(AlignTextEncoder):
    pass


class ChineseCLIPVisionEncoder(CLIPEncoder):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`ChineseCLIPVisionEncoderLayer`].

    Args:
        config: ChineseCLIPConfig
    """

    def __init__(self, config: ChineseCLIPConfig):
        super().__init__()
        self.layers = nn.ModuleList([ChineseCLIPVisionLayer(config) for _ in range(config.num_hidden_layers)])


@auto_docstring(
    custom_intro="""
    The vision model from CHINESE_CLIP without any head or projection on top.
    """
)
class ChineseCLIPVisionModel(CLIPVisionModel):
    def __init__(self, config: ChineseCLIPVisionConfig):
        super().__init__(config)
        self.encoder = ChineseCLIPVisionEncoder(config)

    def forward(self, **super_kwargs):
        r"""
        Examples:

        ```python
        >>> import httpx
        >>> from io import BytesIO
        >>> from PIL import Image
        >>> from transformers import CLIPProcessor, ChineseCLIPVisionModel

        >>> model = ChineseCLIPVisionModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> processor = CLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        super().forward(**super_kwargs)


# Dont copy from AltCLIP if you don't want to get into infinite recursion!
@auto_docstring(
    custom_intro="""
    The text model from CHINESE_CLIP without any head or projection on top.
    """
)
class ChineseCLIPTextModel(ChineseCLIPPreTrainedModel):
    config: ChineseCLIPTextConfig
    input_modalities = ("text",)

    _input_embed_layer = "word_embeddings"
    _can_record_outputs = {
        "hidden_states": ChineseCLIPTextLayer,
        "attentions": ChineseCLIPTextAttention,
    }

    def __init__(self, config, add_pooling_layer=True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)

        self.embeddings = ChineseCLIPTextEmbeddings(config)
        self.encoder = ChineseCLIPTextEncoder(config)
        self.pooler = ChineseCLIPTextPooler(config) if add_pooling_layer else None

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ChineseCLIPTextModel

        >>> model = ChineseCLIPTextModel.from_pretrained("openai/chinese_clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/chinese_clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        inputs_embeds = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


@auto_docstring
class ChineseCLIPModel(CLIPModel):
    def __init__(self, config: ChineseCLIPConfig):
        super().__init__(config)
        self.text_model = ChineseCLIPTextModel(self.config.text_config, add_pooling_layer=False)

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, ChineseCLIPModel

        >>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> inputs = tokenizer(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], padding=True, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     text_features = model.get_text_features(**inputs)
        >>> text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        ```"""
        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            **kwargs,
        )
        pooled_output = text_outputs.last_hidden_state[:, 0, :]
        text_outputs.pooler_output = self.text_projection(pooled_output)

        return text_outputs

    def get_image_features(self, **super_kwargs):
        r"""
        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, ChineseCLIPModel
        >>> from transformers.image_utils import load_image

        >>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> processor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
        >>> image = load_image(url)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.inference_mode():
        ...     image_features = model.get_image_features(**inputs)
        >>> image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        ```"""
        return super().get_image_features(**super_kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_loss: bool | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | ChineseCLIPOutput:
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, ChineseCLIPModel
        >>> from transformers.image_utils import load_image

        >>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> processor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
        >>> image = load_image(url)

        >>> inputs = processor(text=["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], images=image, return_tensors="pt", padding=True)

        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # As CLIP with `token_type_ids`
        vision_outputs = self.get_image_features(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        text_outputs = self.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            **kwargs,
        )

        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        # normalized features
        image_embeds = image_embeds / _get_vector_norm(image_embeds)
        text_embeds = text_embeds / _get_vector_norm(text_embeds)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))
        logits_per_text = logits_per_text * self.logit_scale.exp().to(text_embeds.device)
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = image_text_contrastive_loss(logits_per_text)

        return ChineseCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


__all__ = [
    "ChineseCLIPModel",
    "ChineseCLIPPreTrainedModel",
    "ChineseCLIPTextModel",
    "ChineseCLIPVisionModel",
    "ChineseCLIPConfig",
    "ChineseCLIPTextConfig",
    "ChineseCLIPVisionConfig",
]
