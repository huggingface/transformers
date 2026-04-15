# Copyright 2022 The BAAI Teams Authors and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch AltCLIP model."""

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndProjection,
)
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..chinese_clip.modeling_chinese_clip import (
    ChineseCLIPModel,
    ChineseCLIPTextAttention,
    ChineseCLIPTextLayer,
    ChineseCLIPTextSelfAttention,
)
from ..clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from ..clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPOutput,
    CLIPPreTrainedModel,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    _get_vector_norm,
    image_text_contrastive_loss,
)
from ..roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaIntermediate,
    RobertaOutput,
    RobertaPooler,
    RobertaSelfOutput,
)


@auto_docstring(checkpoint="BAAI/AltCLIP")
@strict
class AltCLIPTextConfig(CLIPTextConfig):
    r"""
    project_dim (`int`, *optional*, defaults to 768):
        The dimensions of the teacher model before the mapping layer.

    Examples:

    ```python
    >>> from transformers import AltCLIPTextModel, AltCLIPTextConfig

    >>> # Initializing a AltCLIPTextConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPTextConfig()

    >>> # Initializing a AltCLIPTextModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    vocab_size: int = 250002
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_act: str = "gelu"
    hidden_dropout_prob: int | float = 0.1
    attention_probs_dropout_prob: int | float = 0
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    initializer_factor: float = 0.02
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | None = 2
    project_dim: int = 768
    projection_dim = AttributeError()
    attention_dropout = AttributeError()


@auto_docstring(checkpoint="BAAI/AltCLIP")
@strict
class AltCLIPVisionConfig(CLIPVisionConfig):
    r"""
    Example:

    ```python
    >>> from transformers import AltCLIPVisionConfig, AltCLIPVisionModel

    >>> # Initializing a AltCLIPVisionConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPVisionConfig()

    >>> # Initializing a AltCLIPVisionModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""


@auto_docstring(checkpoint="BAAI/AltCLIP")
@strict
class AltCLIPConfig(CLIPConfig):
    r"""
    Example:

    ```python
    >>> from transformers import AltCLIPConfig, AltCLIPModel

    >>> # Initializing a AltCLIPConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPConfig()

    >>> # Initializing a AltCLIPModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a AltCLIPConfig from a AltCLIPTextConfig and a AltCLIPVisionConfig

    >>> # Initializing a AltCLIPText and AltCLIPVision configuration
    >>> config_text = AltCLIPTextConfig()
    >>> config_vision = AltCLIPVisionConfig()

    >>> config = AltCLIPConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    projection_dim: int = 768


class AltCLIPOutput(CLIPOutput):
    pass


class AltRobertaEmbeddings(RobertaEmbeddings):
    pass


class AltRobertaSelfAttention(ChineseCLIPTextSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.is_causal = False


class AltRobertaSelfOutput(RobertaSelfOutput):
    pass


class AltRobertaAttention(ChineseCLIPTextAttention):
    def __init__(self, config):
        super().__init__()
        self.self = AltRobertaSelfAttention(config)
        self.output = AltRobertaSelfOutput(config)


class AltRobertaIntermediate(RobertaIntermediate):
    pass


class AltRobertaOutput(RobertaOutput):
    pass


class AltRobertaLayer(ChineseCLIPTextLayer):
    def __init__(self, config):
        super().__init__()
        self.attention = AltRobertaAttention(config)
        self.intermediate = AltRobertaIntermediate(config)
        self.output = AltRobertaOutput(config)


class AltRobertaEncoder(CLIPEncoder):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`AltRobertaEncoderLayer`].

    Args:
        config: AltCLIPTextConfig
    """

    def __init__(self, config: AltCLIPTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([AltRobertaLayer(config) for _ in range(config.num_hidden_layers)])


class AltRobertaPooler(RobertaPooler):
    pass


class AltCLIPAttention(CLIPAttention):
    pass


class AltCLIPMLP(CLIPMLP):
    pass


class AltCLIPEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config: AltCLIPVisionConfig):
        super().__init__(config)


class AltCLIPEncoder(CLIPEncoder):
    pass


class AltCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    pass


class AltCLIPPreTrainedModel(CLIPPreTrainedModel):
    _can_record_outputs = {
        "hidden_states": AltCLIPEncoderLayer,
        "attentions": AltCLIPAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, AltCLIPVisionEmbeddings):
            init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
            init.copy_(module.position_ids, torch.arange(module.num_positions).expand((1, -1)))
        elif isinstance(module, AltCLIPAttention):
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            init.normal_(module.q_proj.weight, std=in_proj_std)
            init.normal_(module.k_proj.weight, std=in_proj_std)
            init.normal_(module.v_proj.weight, std=in_proj_std)
            init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, AltCLIPMLP):
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            init.normal_(module.fc1.weight, std=fc_std)
            init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, AltCLIPModel):
            init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * factor,
            )
            init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * factor,
            )
        elif isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=factor)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=factor)
            # Here we need the check explicitly, as we slice the weight in the `zeros_` call, so it looses the flag
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, AltRobertaEmbeddings):
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
            init.zeros_(module.token_type_ids)


class AltCLIPVisionModel(CLIPVisionModel):
    def forward(self, **super_kwargs):
        r"""
        Examples:

        ```python
        >>> import httpx
        >>> from io import BytesIO
        >>> from PIL import Image
        >>> from transformers import AutoProcessor, AltCLIPVisionModel

        >>> model = AltCLIPVisionModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return super().forward(**super_kwargs)


@auto_docstring(
    custom_intro="""
    The model behaves as an encoder following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    .. _*Attention is all you need*: https://huggingface.co/papers/1706.03762
    """
)
class AltRobertaModel(AltCLIPPreTrainedModel):
    config: AltCLIPTextConfig
    input_modalities = ("text",)

    _input_embed_layer = "word_embeddings"
    _can_record_outputs = {
        "hidden_states": AltRobertaLayer,
        "attentions": AltRobertaSelfAttention,
    }

    def __init__(self, config, add_pooling_layer=True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.embeddings = AltRobertaEmbeddings(config)
        self.encoder = AltRobertaEncoder(config)
        self.pooler = AltRobertaPooler(config) if add_pooling_layer else None

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
        >>> from transformers import AutoTokenizer, AltRobertaModel

        >>> model = AltRobertaModel.from_pretrained("openai/alt_roberta-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/alt_roberta-vit-base-patch32")

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


class AltCLIPTextModel(AltCLIPPreTrainedModel):
    config: AltCLIPTextConfig
    input_modalities = ("text",)
    _input_embed_layer = "word_embedding"
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.roberta = AltRobertaModel(config, add_pooling_layer=False)
        self.transformation = nn.Linear(config.hidden_size, config.project_dim)
        self.pre_LN = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPoolingAndProjection:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoProcessor, AltCLIPTextModel

        >>> model = AltCLIPTextModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

        >>> texts = ["it's a cat", "it's a dog"]

        >>> inputs = processor(text=texts, padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # last module outputs
        sequence_output = outputs[0]

        # project every module
        sequence_output = self.pre_LN(sequence_output)

        # pooler
        projection_state = self.transformation(sequence_output)
        pooler_output = projection_state[:, 0]

        return BaseModelOutputWithPoolingAndProjection(
            last_hidden_state=projection_state,
            pooler_output=pooler_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AltCLIPModel(ChineseCLIPModel, AltCLIPPreTrainedModel):
    config: AltCLIPConfig

    def __init__(self, config: AltCLIPConfig):
        super().__init__(config)
        text_config = config.text_config
        self.text_embed_dim = text_config.project_dim
        self.text_model = AltCLIPTextModel._from_config(self.config.text_config)
        self.vision_model = AltCLIPVisionModel._from_config(self.config.vision_config)

    def get_text_features(self, **super_kwargs):
        r"""
        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, AltCLIPModel

        >>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     text_features = model.get_text_features(**inputs)
        ```"""
        return super().get_text_features(**super_kwargs)

    def get_image_features(self, **super_kwargs):
        r"""
        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, AltCLIPModel
        >>> from transformers.image_utils import load_image

        >>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> inputs = processor(images=image, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     image_features = model.get_image_features(**inputs)
        ```"""
        return super().get_image_features(**super_kwargs)

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
    ) -> tuple | AltCLIPOutput:
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, AltCLIPModel
        >>> from transformers.image_utils import load_image

        >>> model = AltCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        >>> processor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

        >>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
        >>> image = load_image(url)

        >>> inputs = processor(text=["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], images=image, return_tensors="pt", padding=True)

        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            **kwargs,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

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

        return AltCLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


__all__ = [
    "AltCLIPPreTrainedModel",
    "AltCLIPVisionModel",
    "AltCLIPTextModel",
    "AltCLIPModel",
    "AltCLIPTextConfig",
    "AltCLIPVisionConfig",
    "AltCLIPConfig",
]
