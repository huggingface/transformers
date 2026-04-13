# Copyright 2022 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
"""PyTorch CLIPSeg model."""

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from ..clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPEncoder,
    CLIPEncoderLayer,
    CLIPModel,
    CLIPOutput,
    CLIPPreTrainedModel,
    CLIPTextEmbeddings,
    CLIPTextModel,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
)


@auto_docstring(checkpoint="CIDAS/clipseg-rd64")
@strict
class CLIPSegTextConfig(CLIPTextConfig):
    r"""
    Example:

    ```python
    >>> from transformers import CLIPSegTextConfig, CLIPSegTextModel

    >>> # Initializing a CLIPSegTextConfig with CIDAS/clipseg-rd64 style configuration
    >>> configuration = CLIPSegTextConfig()

    >>> # Initializing a CLIPSegTextModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
    >>> model = CLIPSegTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    projection_dim = AttributeError()


@auto_docstring(checkpoint="CIDAS/clipseg-rd64")
@strict
class CLIPSegVisionConfig(CLIPVisionConfig):
    r"""
    Example:

    ```python
    >>> from transformers import CLIPSegVisionConfig, CLIPSegVisionModel

    >>> # Initializing a CLIPSegVisionConfig with CIDAS/clipseg-rd64 style configuration
    >>> configuration = CLIPSegVisionConfig()

    >>> # Initializing a CLIPSegVisionModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
    >>> model = CLIPSegVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    projection_dim = AttributeError()


@auto_docstring(checkpoint="CIDAS/clipseg-rd64")
@strict
class CLIPSegConfig(CLIPConfig):
    r"""
    extract_layers (`list[int]`, *optional*, defaults to `[3, 6, 9]`):
        Layers to extract when forwarding the query image through the frozen visual backbone of CLIP.
    reduce_dim (`int`, *optional*, defaults to 64):
        Dimensionality to reduce the CLIP vision embedding.
    conditional_layer (`int`, *optional*, defaults to 0):
        The layer to use of the Transformer encoder whose activations will be combined with the condition
        embeddings using FiLM (Feature-wise Linear Modulation). If 0, the last layer is used.
    use_complex_transposed_convolution (`bool`, *optional*, defaults to `False`):
        Whether to use a more complex transposed convolution in the decoder, enabling more fine-grained
        segmentation..

    Example:

    ```python
    >>> from transformers import CLIPSegConfig, CLIPSegModel

    >>> # Initializing a CLIPSegConfig with CIDAS/clipseg-rd64 style configuration
    >>> configuration = CLIPSegConfig()

    >>> # Initializing a CLIPSegModel (with random weights) from the CIDAS/clipseg-rd64 style configuration
    >>> model = CLIPSegModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a CLIPSegConfig from a CLIPSegTextConfig and a CLIPSegVisionConfig

    >>> # Initializing a CLIPSegText and CLIPSegVision configuration
    >>> config_text = CLIPSegTextConfig()
    >>> config_vision = CLIPSegVisionConfig()

    >>> config = CLIPSegConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    extract_layers: list[int] | tuple[int, ...] = (3, 6, 9)
    reduce_dim: int = 64
    decoder_num_attention_heads: int = 4
    decoder_attention_dropout: float | int = 0.0
    decoder_hidden_act: str = "quick_gelu"
    decoder_intermediate_size: int = 2048
    conditional_layer: int = 0
    use_complex_transposed_convolution: bool = False


class CLIPSegOutput(CLIPOutput):
    pass


@dataclass
@auto_docstring
class CLIPSegDecoderOutput(ModelOutput):
    r"""
    logits (`torch.FloatTensor` of shape `(batch_size, height, width)`):
        Classification scores for each pixel.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*,):
        Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        Rreturned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`
    attentions (`tuple(torch.FloatTensor)`, *optional*):
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads. Returned when `output_attentions=True` is passed or when `config.output_attentions=True`
    """

    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
@auto_docstring
class CLIPSegImageSegmentationOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Binary cross entropy loss for segmentation.
    logits (`torch.FloatTensor` of shape `(batch_size, height, width)`):
        Classification scores for each pixel.
    conditional_embeddings (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
        Conditional embeddings used for segmentation.
    pooled_output (`torch.FloatTensor` of shape `(batch_size, embed_dim)`):
        Pooled output of the [`CLIPSegVisionModel`].
    vision_model_output (`BaseModelOutputWithPooling`):
        The output of the [`CLIPSegVisionModel`].
    decoder_output (`CLIPSegDecoderOutput`):
        The output of the [`CLIPSegDecoder`].
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    conditional_embeddings: torch.FloatTensor | None = None
    pooled_output: torch.FloatTensor | None = None
    vision_model_output: BaseModelOutputWithPooling = None
    decoder_output: CLIPSegDecoderOutput = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(v.to_tuple() if isinstance(v, ModelOutput) else v for v in self.values())


class CLIPSegVisionEmbeddings(CLIPVisionEmbeddings):
    # Different default for `interpolate_pos_encoding` from CLIP
    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=True) -> torch.Tensor:
        super().forward(pixel_values, interpolate_pos_encoding)


class CLIPSegTextEmbeddings(CLIPTextEmbeddings):
    pass


class CLIPSegAttention(CLIPAttention):
    pass


class CLIPSegMLP(CLIPMLP):
    pass


class CLIPSegEncoderLayer(CLIPEncoderLayer):
    pass


class CLIPSegDecoderLayer(CLIPEncoderLayer):
    """
    CLIPSeg decoder layer, which is identical to `CLIPSegEncoderLayer`, except that normalization is applied after
    self-attention/MLP, rather than before.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        return hidden_states


@auto_docstring
class CLIPSegPreTrainedModel(CLIPPreTrainedModel):
    _can_record_outputs = {
        "hidden_states": [CLIPSegEncoderLayer, CLIPSegDecoderLayer],
        "attentions": CLIPSegAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, CLIPSegTextEmbeddings):
            init.normal_(module.token_embedding.weight, mean=0.0, std=factor * 0.02)
            init.normal_(module.position_embedding.weight, mean=0.0, std=factor * 0.02)
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        elif isinstance(module, CLIPSegVisionEmbeddings):
            init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
            init.copy_(module.position_ids, torch.arange(module.num_positions).expand((1, -1)))
        elif isinstance(module, CLIPSegAttention):
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            init.normal_(module.q_proj.weight, std=in_proj_std)
            init.normal_(module.k_proj.weight, std=in_proj_std)
            init.normal_(module.v_proj.weight, std=in_proj_std)
            init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, CLIPSegMLP):
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            init.normal_(module.fc1.weight, std=fc_std)
            init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, CLIPSegModel):
            init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * factor,
            )
            init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * factor,
            )

        if isinstance(module, nn.LayerNorm):
            init.zeros_(module.bias)
            init.ones_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            init.zeros_(module.bias)


class CLIPSegEncoder(CLIPEncoder):
    pass


class CLIPSegDecoder(CLIPSegPreTrainedModel):
    def __init__(self, config: CLIPSegConfig):
        super().__init__(config)

        self.conditional_layer = config.conditional_layer

        self.film_mul = nn.Linear(config.projection_dim, config.reduce_dim)
        self.film_add = nn.Linear(config.projection_dim, config.reduce_dim)

        if config.use_complex_transposed_convolution:
            transposed_kernels = (config.vision_config.patch_size // 4, config.vision_config.patch_size // 4)

            self.transposed_convolution = nn.Sequential(
                nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    config.reduce_dim,
                    config.reduce_dim // 2,
                    kernel_size=transposed_kernels[0],
                    stride=transposed_kernels[0],
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    config.reduce_dim // 2, 1, kernel_size=transposed_kernels[1], stride=transposed_kernels[1]
                ),
            )
        else:
            self.transposed_convolution = nn.ConvTranspose2d(
                config.reduce_dim, 1, config.vision_config.patch_size, stride=config.vision_config.patch_size
            )

        depth = len(config.extract_layers)
        self.reduces = nn.ModuleList(
            [nn.Linear(config.vision_config.hidden_size, config.reduce_dim) for _ in range(depth)]
        )

        decoder_config = copy.deepcopy(config.vision_config)
        decoder_config.hidden_size = config.reduce_dim
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config.hidden_act = "relu"
        self.layers = nn.ModuleList([CLIPSegDecoderLayer(decoder_config) for _ in range(len(config.extract_layers))])

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        hidden_states: tuple[torch.Tensor],
        conditional_embeddings: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CLIPSegDecoderOutput:
        r"""
        conditional_embeddings (`torch.FloatTensor` of shape `(batch_size, config.projection_dim)`, *optional*):
            The conditional embeddings for the query images. If provided, the model will use this instead of computing
            the embeddings from the conditional_pixel_values.
        """
        activations = hidden_states[::-1]

        output = None
        for i, (activation, layer, reduce) in enumerate(zip(activations, self.layers, self.reduces)):
            if output is not None:
                output = reduce(activation) + output
            else:
                output = reduce(activation)

            if i == self.conditional_layer:
                output = self.film_mul(conditional_embeddings) * output.permute(1, 0, 2) + self.film_add(
                    conditional_embeddings
                )
                output = output.permute(1, 0, 2)

            output = layer(output, attention_mask=None, **kwargs)

        output = output[:, 1:, :].transpose(1, 2)  # remove cls token and reshape to [batch_size, reduce_dim, seq_len]

        size = int(math.sqrt(output.shape[2]))

        batch_size = conditional_embeddings.shape[0]
        output = output.view(batch_size, output.shape[1], size, size)

        logits = self.transposed_convolution(output).squeeze(1)

        return CLIPSegDecoderOutput(logits=logits)


class CLIPSegTextModel(CLIPTextModel):
    def forward(self, **super_kwargs) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPSegTextModel

        >>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegTextModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return super().forward(**super_kwargs)


class CLIPSegVisionModel(CLIPVisionModel):
    def forward(
        self,
        pixel_values: torch.FloatTensor | None,
        interpolate_pos_encoding: bool | None = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> import httpx
        >>> from io import BytesIO
        >>> from PIL import Image
        >>> from transformers import AutoProcessor, CLIPSegVisionModel

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegVisionModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return super().forward(pixel_values, interpolate_pos_encoding, **kwargs)


class CLIPSegModel(CLIPModel):
    def get_text_features(self, **super_kwargs):
        r"""
        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, CLIPSegModel

        >>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> with torch.inference_mode():
        ...     text_features = model.get_text_features(**inputs)
        ```"""
        return super().get_text_features(**super_kwargs)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, CLIPSegModel
        >>> from transformers.image_utils import load_image

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.inference_mode():
        ...     image_features = model.get_image_features(**inputs)
        ```"""
        return super().get_image_features(pixel_values, interpolate_pos_encoding, **kwargs)

    def forward(self, interpolate_pos_encoding: bool = True, **super_kwargs):
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, CLIPSegModel
        >>> from transformers.image_utils import load_image

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        super().forward(interpolate_pos_encoding=interpolate_pos_encoding, **super_kwargs)


@auto_docstring(
    custom_intro="""
    CLIPSeg model with a Transformer-based decoder on top for zero-shot and one-shot image segmentation.
    """
)
class CLIPSegForImageSegmentation(CLIPSegPreTrainedModel):
    config: CLIPSegConfig

    def __init__(self, config: CLIPSegConfig):
        super().__init__(config)
        self.clip = CLIPSegModel(config)
        self.extract_layers = config.extract_layers
        self.decoder = CLIPSegDecoder(config)

        self.post_init()

    def get_conditional_embeddings(
        self,
        batch_size: int | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        conditional_pixel_values: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        if input_ids is not None:
            # compute conditional embeddings from texts
            if len(input_ids) != batch_size:
                raise ValueError("Make sure to pass as many prompt texts as there are query images")
            with torch.no_grad():
                conditional_embeddings = self.clip.get_text_features(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids
                ).pooler_output
        elif conditional_pixel_values is not None:
            # compute conditional embeddings from images
            if len(conditional_pixel_values) != batch_size:
                raise ValueError("Make sure to pass as many prompt images as there are query images")
            with torch.no_grad():
                conditional_embeddings = self.clip.get_image_features(conditional_pixel_values).pooler_output
        else:
            raise ValueError(
                "Invalid conditional, should be either provided as `input_ids` or `conditional_pixel_values`"
            )

        return conditional_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        conditional_pixel_values: torch.FloatTensor | None = None,
        conditional_embeddings: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        interpolate_pos_encoding: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CLIPSegOutput:
        r"""
        conditional_pixel_values (`torch.FloatTensor`, *optional*):
            The pixel values of the conditional images.
        conditional_embeddings (`torch.FloatTensor` of shape `(batch_size, config.projection_dim)`, *optional*):
            The conditional embeddings for the query images. If provided, the model will use this instead of computing
            the embeddings from the conditional_pixel_values.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, CLIPSegForImageSegmentation
        >>> from transformers.image_utils import load_image

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> texts = ["a cat", "a remote", "a blanket"]
        >>> inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> print(logits.shape)
        torch.Size([3, 352, 352])
        ```"""
        # step 1: forward the query images through the frozen CLIP vision encoder
        with torch.no_grad():
            kwargs["output_hidden_states"] = True  # required to extract layers for the stages
            vision_outputs = self.clip.get_image_features(
                pixel_values=pixel_values,
                interpolate_pos_encoding=interpolate_pos_encoding,
                **kwargs,
            )
            pooled_output = vision_outputs.pooler_output

            hidden_states = vision_outputs.hidden_states
            # we add +1 here as the hidden states also include the initial embeddings
            activations = [hidden_states[i + 1] for i in self.extract_layers]

            # update vision_outputs
            vision_outputs = BaseModelOutputWithPooling(
                last_hidden_state=vision_outputs.last_hidden_state,
                pooler_output=vision_outputs.pooler_output,
                hidden_states=vision_outputs.hidden_states,
                attentions=vision_outputs.attentions,
            )

        # step 2: compute conditional embeddings, either from text, images or an own provided embedding
        if conditional_embeddings is None:
            conditional_embeddings = self.get_conditional_embeddings(
                batch_size=pixel_values.shape[0],
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                conditional_pixel_values=conditional_pixel_values,
            )
        else:
            if conditional_embeddings.shape[0] != pixel_values.shape[0]:
                raise ValueError(
                    "Make sure to pass as many conditional embeddings as there are query images in the batch"
                )
            if conditional_embeddings.shape[1] != self.config.projection_dim:
                raise ValueError(
                    "Make sure that the feature dimension of the conditional embeddings matches"
                    " `config.projection_dim`."
                )

        # step 3: forward both the pooled output and the activations through the lightweight decoder to predict masks
        decoder_outputs = self.decoder(
            activations,
            conditional_embeddings,
            **kwargs,
        )
        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            # move labels to the correct device to enable PP
            labels = labels.to(logits.device)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return CLIPSegImageSegmentationOutput(
            loss=loss,
            logits=logits,
            conditional_embeddings=conditional_embeddings,
            pooled_output=pooled_output,
            vision_model_output=vision_outputs,
            decoder_output=decoder_outputs,
        )


__all__ = [
    "CLIPSegConfig",
    "CLIPSegTextConfig",
    "CLIPSegVisionConfig",
    "CLIPSegModel",
    "CLIPSegPreTrainedModel",
    "CLIPSegTextModel",
    "CLIPSegVisionModel",
    "CLIPSegForImageSegmentation",
]
