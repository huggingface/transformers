from typing import Optional

import torch
from torch import nn

from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import check_model_inputs
from ..clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from ..clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPForImageClassification,
    CLIPModel,
    CLIPTextEmbeddings,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTextTransformer,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "facebook/metaclip-2-worldwide-huge-quickgelu"
_CONFIG_FOR_DOC = "MetaClip2Config"


class MetaClip2TextConfig(CLIPTextConfig):
    r"""
    This is the configuration class to store the configuration of a [`MetaClip2TextModel`]. It is used to instantiate
    a MetaClip2 text encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MetaClip2
    [facebook/metaclip-2-worldwide-huge-quickgelu](https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the MetaClip2 text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`MetaClip2TextModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 49406):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 49407):
            End of stream token id.

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

    pass


class MetaClip2VisionConfig(CLIPVisionConfig):
    r"""
    This is the configuration class to store the configuration of a [`MetaClip2VisionModel`]. It is used to instantiate a MetaClip2
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the vision encoder of the MetaClip2
    [facebook/metaclip-2-worldwide-huge-quickgelu](https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

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

    pass


class MetaClip2Config(CLIPConfig):
    r"""
    [`MetaClip2Config`] is the configuration class to store the configuration of a [`MetaClip2Model`]. It is used to
    instantiate a MetaClip2 model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the MetaClip2
    [facebook/metaclip-2-worldwide-huge-quickgelu](https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`MetaClip2TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`MetaClip2VisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimensionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The initial value of the *logit_scale* parameter. Default is used as per the original MetaClip2 implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

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

    >>> config = MetaClip2Config.from_text_vision_configs(config_text, config_vision)
    ```"""

    pass


class MetaClip2TextEmbeddings(CLIPTextEmbeddings):
    pass


class MetaClip2VisionEmbeddings(CLIPVisionEmbeddings):
    pass


class MetaClip2Attention(CLIPAttention):
    pass


class MetaClip2MLP(CLIPMLP):
    pass


@auto_docstring
class MetaClip2PreTrainedModel(PreTrainedModel):
    config: MetaClip2Config
    base_model_prefix = "metaclip_2"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, MetaClip2TextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, MetaClip2VisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, MetaClip2Attention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, MetaClip2MLP):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, MetaClip2Model):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, MetaClip2VisionModelWithProjection):
            nn.init.normal_(
                module.visual_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, MetaClip2TextModelWithProjection):
            nn.init.normal_(
                module.text_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        elif isinstance(module, MetaClip2ForImageClassification):
            nn.init.normal_(
                module.classifier.weight,
                std=self.config.vision_config.hidden_size**-0.5 * self.config.initializer_factor,
            )

        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class MetaClip2TextTransformer(CLIPTextTransformer):
    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None and self.config._attn_implementation != "flash_attention_2":
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
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

    def __init__(self, config: MetaClip2TextConfig):
        super().__init__(config)
        self.text_model = MetaClip2TextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
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
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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

        text_model = MetaClip2TextModel._from_config(config)
        self.text_model = text_model.text_model

        self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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
    >>> import requests
    >>> from transformers import AutoProcessor, MetaClip2Model

    >>> model = MetaClip2Model.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
    >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

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

        text_model = MetaClip2TextModel._from_config(text_config)
        self.text_model = text_model.text_model

        vision_model = MetaClip2VisionModel._from_config(vision_config)
        self.vision_model = vision_model.vision_model

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MetaClip2Model

        >>> model = MetaClip2Model.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`MetaClip2TextModel`].

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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`MetaClip2VisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MetaClip2Model

        >>> model = MetaClip2Model.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        return super().get_image_features(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
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
    >>> import requests
    >>> from transformers import AutoProcessor, MetaClip2VisionModel

    >>> model = MetaClip2VisionModel.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
    >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooled_output = outputs.pooler_output  # pooled CLS states
    ```"""

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        r"""
        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MetaClip2VisionModel

        >>> model = MetaClip2VisionModel.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return super().forward(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
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
    >>> import requests
    >>> from transformers import AutoProcessor, MetaClip2VisionModelWithProjection

    >>> model = MetaClip2VisionModelWithProjection.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
    >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> image_embeds = outputs.image_embeds
    ```"""

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        r"""
        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MetaClip2VisionModelWithProjection

        >>> model = MetaClip2VisionModelWithProjection.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> image_embeds = outputs.image_embeds
        ```"""
        return super().forward(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )


class MetaClip2ForImageClassification(CLIPForImageClassification):
    pass


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
