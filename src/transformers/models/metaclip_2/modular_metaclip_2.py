# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from ..clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPForImageClassification,
    CLIPModel,
    CLIPPreTrainedModel,
    CLIPTextEmbeddings,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
    _init_child_weight,
)


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "facebook/metaclip-2-worldwide-huge-quickgelu"
_CONFIG_FOR_DOC = "MetaClip2Config"


@auto_docstring(checkpoint="facebook/metaclip-2-worldwide-huge-quickgelu")
@strict
class MetaClip2TextConfig(CLIPTextConfig):
    r"""
    Example:

    ```python
    >>> from transformers import MetaClip2TextConfig, MetaClip2TextModel

    >>> # Initializing a MetaClip2TextConfig with facebook/metaclip-2-worldwide-huge-quickgelu style configuration
    >>> configuration = MetaClip2TextConfig()

    >>> # Initializing a MetaClip2TextModel (with random weights) from the facebook/metaclip-2-worldwide-huge-quickgelu style configuration
    >>> model = MetaClip2TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""


@auto_docstring(checkpoint="facebook/metaclip-2-worldwide-huge-quickgelu")
@strict
class MetaClip2VisionConfig(CLIPVisionConfig):
    r"""
    Example:

    ```python
    >>> from transformers import MetaClip2VisionConfig, MetaClip2VisionModel

    >>> # Initializing a MetaClip2VisionConfig with facebook/metaclip-2-worldwide-huge-quickgelu style configuration
    >>> configuration = MetaClip2VisionConfig()

    >>> # Initializing a MetaClip2VisionModel (with random weights) from the facebook/metaclip-2-worldwide-huge-quickgelu style configuration
    >>> model = MetaClip2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""


@auto_docstring(checkpoint="facebook/metaclip-2-worldwide-huge-quickgelu")
@strict
class MetaClip2Config(CLIPConfig):
    r"""
    Example:

    ```python
    >>> from transformers import MetaClip2Config, MetaClip2Model

    >>> # Initializing a MetaClip2Config with facebook/metaclip-2-worldwide-huge-quickgelu style configuration
    >>> configuration = MetaClip2Config()

    >>> # Initializing a MetaClip2Model (with random weights) from the facebook/metaclip-2-worldwide-huge-quickgelu style configuration
    >>> model = MetaClip2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a MetaClip2Config from a MetaClip2TextConfig and a MetaClip2VisionConfig
    >>> from transformers import MetaClip2TextConfig, MetaClip2VisionConfig

    >>> # Initializing a MetaClip2Text and MetaClip2Vision configuration
    >>> config_text = MetaClip2TextConfig()
    >>> config_vision = MetaClip2VisionConfig()

    >>> config = MetaClip2Config(text_config=config_text, vision_config=config_vision)
    ```"""

    model_type = "metaclip_2"
    sub_configs_defaults = {
        "vision_config": MetaClip2VisionConfig,
        "text_config": MetaClip2TextConfig,
    }

    def __post_init__(self, **kwargs):
        # For backward compatibility check keyword args
        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if (text_config_dict := kwargs.pop("text_config_dict", None)) is not None:
            # This is the complete result when using `text_config_dict`.
            _text_config_dict = MetaClip2TextConfig(**text_config_dict).to_dict()
            text_config = (
                self.text_config.to_dict() if isinstance(self.text_config, PreTrainedConfig) else self.text_config
            )
            text_config = text_config or {}

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key != "transformers_version":
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The "
                            f'value `text_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)
            self.text_config = text_config

        if (vision_config_dict := kwargs.pop("vision_config_dict", None)) is not None:
            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = MetaClip2VisionConfig(**vision_config_dict).to_dict()
            vision_config = (
                self.vision_config.to_dict()
                if isinstance(self.vision_config, PreTrainedConfig)
                else self.vision_config
            )
            vision_config = vision_config or {}

            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict` and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key != "transformers_version":
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `CLIPVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overridden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)
            self.vision_config = vision_config

        PreTrainedConfig.__post_init__(self, **kwargs)


class MetaClip2TextEmbeddings(CLIPTextEmbeddings):
    pass


class MetaClip2VisionEmbeddings(CLIPVisionEmbeddings):
    pass


class MetaClip2Attention(CLIPAttention):
    pass


class MetaClip2MLP(CLIPMLP):
    pass


@auto_docstring
class MetaClip2PreTrainedModel(CLIPPreTrainedModel):
    base_model_prefix = "metaclip_2"

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        PreTrainedModel._init_weights(self, module)
        factor = self.config.initializer_factor
        if isinstance(module, MetaClip2TextEmbeddings):
            _init_child_weight(module.token_embedding, factor * 0.02)
            _init_child_weight(module.position_embedding, factor * 0.02)
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))
        elif isinstance(module, MetaClip2VisionEmbeddings):
            init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            _init_child_weight(module.patch_embedding, module.config.initializer_range * factor)
            _init_child_weight(module.position_embedding, module.config.initializer_range * factor)
            init.copy_(module.position_ids, torch.arange(module.num_positions).expand((1, -1)))
        elif isinstance(module, MetaClip2Attention):
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            _init_child_weight(module.q_proj, in_proj_std)
            _init_child_weight(module.k_proj, in_proj_std)
            _init_child_weight(module.v_proj, in_proj_std)
            _init_child_weight(module.out_proj, out_proj_std)
        elif isinstance(module, MetaClip2MLP):
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            _init_child_weight(module.fc1, fc_std)
            _init_child_weight(module.fc2, in_proj_std)
        elif isinstance(module, MetaClip2Model):
            _init_child_weight(module.text_projection, module.text_embed_dim**-0.5 * factor)
            _init_child_weight(module.visual_projection, module.vision_embed_dim**-0.5 * factor)
        elif isinstance(module, MetaClip2VisionModelWithProjection):
            _init_child_weight(module.visual_projection, self.config.hidden_size**-0.5 * factor)
        elif isinstance(module, MetaClip2TextModelWithProjection):
            _init_child_weight(module.text_projection, self.config.hidden_size**-0.5 * factor)
        elif isinstance(module, MetaClip2ForImageClassification):
            _init_child_weight(module.classifier, self.config.vision_config.hidden_size**-0.5 * factor)
        if hasattr(module, "logit_scale"):
            init.constant_(module.logit_scale, self.config.logit_scale_init_value)


class MetaClip2TextModel(CLIPTextModel):
    """
    The text model from MetaClip2 without any head or projection on top.
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Args:
        config ([`MetaClip2TextConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, MetaClip2TextModel

    >>> model = MetaClip2TextModel.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

    >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
    ```"""

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MetaClip2TextModel

        >>> model = MetaClip2TextModel.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        # Unlike CLIP, this model doesn't handle bc for `self.eos_token_id`, even if the EOS
        # token is 2 (it is set to 2 in released weights)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
        )

        kwargs.pop("is_causal", None)
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            is_causal=True,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Use robust pooling like CLIP - finds the first EOS token position per sequence
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id).int().argmax(dim=-1),
        ]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MetaClip2TextModelWithProjection(CLIPTextModelWithProjection):
    """
    MetaClip2 text model with a projection layer on top (a linear layer on top of the pooled output).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Args:
        config ([`MetaClip2TextConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, MetaClip2TextModelWithProjection

    >>> model = MetaClip2TextModelWithProjection.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

    >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> text_embeds = outputs.text_embeds
    ```"""

    def __init__(self, config: MetaClip2TextConfig):
        super().__init__(config)

        self.text_model = MetaClip2TextModel._from_config(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MetaClip2TextModelWithProjection

        >>> model = MetaClip2TextModelWithProjection.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


class MetaClip2Model(CLIPModel):
    """
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Args:
        config ([`MetaClip2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Examples:

    ```python
    >>> from PIL import Image
    >>> from huggingface_hub.utils import httpx
        >>> from io import BytesIO
    >>> from transformers import AutoProcessor, MetaClip2Model

    >>> model = MetaClip2Model.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
    >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

    >>> inputs = processor(
    ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
    ... )

    >>> outputs = model(**inputs)
    >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    ```"""

    def __init__(self, config: MetaClip2Config):
        super().__init__(config)

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = MetaClip2TextModel._from_config(text_config)
        self.vision_model = MetaClip2VisionModel._from_config(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_loss: bool | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> from PIL import Image
        >>> from huggingface_hub.utils import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, MetaClip2Model

        >>> model = MetaClip2Model.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MetaClip2Model

        >>> model = MetaClip2Model.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        return super().get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from PIL import Image
        >>> from huggingface_hub.utils import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, MetaClip2Model

        >>> model = MetaClip2Model.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        return super().get_image_features(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
            **kwargs,
        )


class MetaClip2VisionModel(CLIPVisionModel):
    """
    The vision model from MetaClip2 without any head or projection on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Args:
        config ([`MetaClip2VisionConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Examples:

    ```python
    >>> from PIL import Image
    >>> from huggingface_hub.utils import httpx
    >>> from io import BytesIO
    >>> from transformers import AutoProcessor, MetaClip2VisionModel

    >>> model = MetaClip2VisionModel.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
    >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> with httpx.stream("GET", url) as response:
    ...     image = Image.open(BytesIO(response.read()))

    >>> inputs = processor(images=image, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooled_output = outputs.pooler_output  # pooled CLS states
    ```"""

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        Examples:

        ```python
        >>> from PIL import Image
        >>> from huggingface_hub.utils import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, MetaClip2VisionModel

        >>> model = MetaClip2VisionModel.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return super().forward(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )


class MetaClip2VisionModelWithProjection(CLIPVisionModelWithProjection):
    """
    MetaClip2 vision model with a projection layer on top (a linear layer on top of the pooled output).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Args:
        config ([`MetaClip2VisionConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.

    Examples:

    ```python
    >>> from PIL import Image
    >>> from huggingface_hub.utils import httpx
    >>> from io import BytesIO
    >>> from transformers import AutoProcessor, MetaClip2VisionModelWithProjection

    >>> model = MetaClip2VisionModelWithProjection.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
    >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> with httpx.stream("GET", url) as response:
    ...     image = Image.open(BytesIO(response.read()))

    >>> inputs = processor(images=image, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> image_embeds = outputs.image_embeds
    ```"""

    def __init__(self, config: MetaClip2VisionConfig):
        super().__init__(config)

        self.vision_model = MetaClip2VisionModel._from_config(config)

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        Examples:

        ```python
        >>> from PIL import Image
        >>> from huggingface_hub.utils import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, MetaClip2VisionModelWithProjection

        >>> model = MetaClip2VisionModelWithProjection.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> image_embeds = outputs.image_embeds
        ```"""
        return super().forward(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )


class MetaClip2ForImageClassification(CLIPForImageClassification):
    def __init__(self, config: MetaClip2Config) -> None:
        super().__init__(config)
        self.vision_model = MetaClip2VisionModel._from_config(config.vision_config)


__all__ = [
    "MetaClip2Config",
    "MetaClip2TextConfig",
    "MetaClip2VisionConfig",
    "MetaClip2Model",
    "MetaClip2PreTrainedModel",
    "MetaClip2TextModel",
    "MetaClip2TextModelWithProjection",
    "MetaClip2VisionModel",
    "MetaClip2VisionModelWithProjection",
    "MetaClip2ForImageClassification",
]
