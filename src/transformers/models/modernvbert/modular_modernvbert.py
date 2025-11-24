import copy
import os
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple, logging
from ..modernbert import ModernBertConfig, ModernBertForMaskedLM, ModernBertModel
from ..siglip import SiglipConfig, SiglipVisionConfig, SiglipVisionModel
from ..idefics3 import Idefics3ImageProcessor, Idefics3ImageProcessorFast, Idefics3Processor

logger = logging.get_logger(__name__)

class ModernVBertImageProcessor(Idefics3ImageProcessor):
    pass

class ModernVBertImageProcessorFast(Idefics3ImageProcessorFast):
    pass

DEFAULT_CHAT_TEMPLATE = "<|begin_of_text|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
class ModernVBertProcessor(Idefics3Processor):
    image_processor_class = "ModernVBertImageProcessor"

    def apply_chat_template(self, conversation, chat_template = None, **kwargs):
        if chat_template is None:
            chat_template = DEFAULT_CHAT_TEMPLATE
        return super().apply_chat_template(conversation, chat_template, **kwargs)

class ModernVBertTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ModernBERT`]. It is used to instantiate an ModernBERT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [jhu-clsp/ettin-encoder-150m](https://huggingface.co/jhu-clsp/ettin-encoder-150m) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "modernvbert_text"

    def __init__(
        self,
        text_model_name="jhu-clsp/ettin-encoder-150m",
        hidden_size=768,
        num_hidden_layers=22,
        intermediate_size=1152,
        mlp_bias=False,
        vocab_size=50368,
        **kwargs,
    ):
        super().__init__(
            text_model_name=text_model_name,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            mlp_bias=mlp_bias,
            vocab_size=vocab_size,
            **kwargs,
        )

    @classmethod
    def from_base_model(
        cls,
        text_model_name,
        **kwargs,
    ):
        text_config = ModernBertConfig.from_pretrained(text_model_name)
        if hasattr(text_config, "text_config"):
            text_config = text_config.text_config

        return cls(
            text_model_name=text_model_name,
            hidden_size=text_config.hidden_size,
            num_hidden_layers=text_config.num_hidden_layers,
            intermediate_size=text_config.intermediate_size,
            mlp_bias=text_config.mlp_bias,
            vocab_size=text_config.vocab_size,
            **kwargs,
        )


class ModernVBertVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SigLIP`]. It is used to instantiate the vision encoder part of the ModernVBERT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the SigLIP.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "modernvbert_vision"

    attribute_map = {
        "hidden_size": "embed_dim",
    }

    def __init__(
        self,
        vision_model_name="google/siglip2-base-patch16-512",
        embed_dim=768,
        image_size=512,
        patch_size=16,
        num_hidden_layers=12,
        intermediate_size=3072,
        **kwargs,
    ):
        super().__init__(
            vision_model_name=vision_model_name,
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            **kwargs,
        )

    @classmethod
    def from_base_model(
        cls,
        vision_model_name,
        **kwargs,
    ):
        vision_config = SiglipConfig.from_pretrained(vision_model_name)
        if hasattr(vision_config, "vision_config"):
            vision_config = vision_config.vision_config

        return cls(
            vision_model_name=vision_model_name,
            embed_dim=vision_config.hidden_size,
            image_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            num_hidden_layers=vision_config.num_hidden_layers,
            intermediate_size=vision_config.intermediate_size,
            **kwargs,
        )


class ModernVBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a `ModernVBert` model. It is used to
    instantiate a ModernVBert model according to the specified arguments and defines the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    See the documentation for [`PretrainedConfig`] for more details.

    Args:
        text_config (`PretrainedConfig` or `dict`, optional):
            Custom text config or a dict with a `text_model_name` key for the text encoder. If `None`, the
            default text backbone defined by `DEFAULT_TEXT_MODEL_NAME` is used.
        vision_config (`PretrainedConfig` or `dict`, optional):
            Custom vision config or a dict with a `vision_model_name` key for the vision encoder. If `None`, the
            default vision backbone defined by `DEFAULT_VISION_MODEL_NAME` is used.
        image_token_id (`int`, optional, defaults to 128257):
            Token id reserved for image tokens inserted into the text stream.
        vocab_size (`int`, optional, defaults to 128256):
            Vocabulary size used by the text embeddings.
        tie_word_embeddings (`bool`, optional, defaults to `False`):
            Whether to tie input token embeddings and output token embeddings.
        pixel_shuffle_factor (`int`, optional, defaults to 4):
            Scale factor used by any pixel-shuffle / upsampling operations in the vision head.
        additional_vocab_size (`int`, optional, defaults to 0):
            Number of extra tokens appended to the base vocabulary (useful for adapters / special tokens).
        pad_token_id (`int`, optional):
            Padding token id.
        initializer_range (`float`, optional, defaults to 0.02):
            Stddev used for weight initialization.

    Example:
    ```python
    >>> from modernvbert import ModernVBertConfig

    >>> # Initializing configuration
    >>> configuration = ModernVBertConfig()

    >>> # Initializing a model from the configuration (model class is implemented in
    >>> # `modernvbert.modeling_modernvbert`)

    >>> from modernvbert import ModernVBertModel
    >>> model = ModernVBertModel(configuration)

    >>> # Accessing the model configuration
    >>> cfg = model.config
    ```"""

    model_type = "modernvbert"
    sub_configs: dict[str, Any] = {"text_config": ModernVBertTextConfig, "vision_config": ModernVBertVisionConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 50407,
        initializer_range=0.02,
        vocab_size=50368,
        pad_token_id=None,
        pixel_shuffle_factor=4,
        additional_vocab_size=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = self.sub_configs["text_config"].from_base_model("jhu-clsp/ettin-encoder-150m")
        elif isinstance(text_config, dict):
            text_config = self.sub_configs["text_config"].from_dict(text_config)
        self.text_config = text_config

        if vision_config is None:
            vision_config = self.sub_configs["vision_config"].from_base_model("google/siglip2-base-patch16-512")
        elif isinstance(vision_config, dict):
            vision_config = self.sub_configs["vision_config"].from_dict(vision_config)
        self.vision_config = vision_config

        self.initializer_range = initializer_range
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.vocab_size = vocab_size
        self.additional_vocab_size = additional_vocab_size
        self.hidden_size = kwargs.pop("hidden_size", self.text_config.hidden_size)

    @classmethod
    def from_pretrained_models(
        cls,
        text_model_name: Union[str, os.PathLike],
        vision_model_name: Union[str, os.PathLike],
        **kwargs,
    ) -> "PretrainedConfig":
        text_model_config = ModernVBertTextConfig.from_base_model(text_model_name)
        vision_model_config = ModernVBertVisionConfig.from_base_model(vision_model_name)
        return cls(
            text_config=text_model_config,
            vision_config=vision_model_config,
            **kwargs,
        )


class DecoupledEmbedding(nn.Embedding):
    # Derived from https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the embeddings.
    In practise, the regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `num_additional_embeddings` > 0, then it will create `num_additional_embeddings` additional parameters that are always trained.
    If `num_additional_embeddings=0`, then the module defaults back to the regular behavior of `nn.Embedding`.
    """

    def __init__(
        self,
        num_embeddings,
        num_additional_embeddings,
        embedding_dim,
        partially_freeze=False,
        device=None,
        dtype=None,
        padding_idx=None,
        **kwargs,
    ) -> None:
        """
        num_additional_embeddings: int. Number of additional embeddings. Only useful when you `partially_freeze=True`.
        partially_freeze: bool. If True, the regular `weight` will be frozen. `additional_weight` is never frozen.

        Note: there are a lot of other parameters to initialize a standard `nn.Embedding` such as `padding_idx`, `max_norm` or `norm_type`. We are not supporting these.
        """
        if padding_idx is not None and padding_idx > num_embeddings:
            raise ValueError(f"padding_idx must be within num_embeddings. Got {padding_idx} and {num_embeddings}")

        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            padding_idx=padding_idx,
            **kwargs,
        )
        self.num_embeddings = num_embeddings
        self.num_additional_embeddings = num_additional_embeddings
        self.partially_freeze = partially_freeze

        if partially_freeze:
            self.weight.requires_grad_(False)

        if self.num_additional_embeddings > 0:
            self.additional_embedding = nn.Embedding(
                num_embeddings=num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )

    def forward(self, input_ids):
        """
        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.

        in order to make a lookup of the input ids, we:
        1. find out the indices of the entries belonging to the 2nd embedding
        2. extract those values while subtracting the size of the first embedding (num_embeddings),
           since the 2nd embedding starts from 0 and not num_embeddings
        3. perform the 2nd embedding lookup
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        5. perform the 1st embedding lookup
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup

        note: for the 1st embedding lookup we could have looked up only the low indices and not do
        the padding, but then we have to create a new tensor and populate it with 2 tensors that are
        spread out across various indices - i.e. not a simple concat - I haven't benchmarked the
        complex case if it's any faster, given that seqlens are usually relatively short it's
        probably not faster or if faster not by much - but might be a good idea to measure.

        """
        if self.num_additional_embeddings == 0:
            return super().forward(input_ids)

        input_ids = input_ids.clone()
        additional_vocab_indices = torch.where(input_ids >= self.num_embeddings)
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        additional_embeddings = self.additional_embedding(input_ids_additional_vocab - self.num_embeddings)

        # for successful lookup replace input_ids with 0, the results of these will be discarded anyway
        input_ids[additional_vocab_indices] = 0
        full_vector = F.embedding(input_ids, self.weight)
        full_vector[additional_vocab_indices] = additional_embeddings  # overwrite the records with high indices
        return full_vector


@dataclass
class ModernVBertBaseModelOutput(BaseModelOutput):
    """
    Base class for ModernVBERT model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[tuple[torch.FloatTensor]] = None


@dataclass
class ModernVBertMaskedLMOutput(MaskedLMOutput):
    """
    Base class for ModernVBERT model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class ModernVBertSimpleMLP(nn.Module):
    """A simple linear projection layer to project the vision hidden states to the text hidden states."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.proj = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.proj(x)


class ModernVBertConnector(nn.Module):
    """
    Connector module for ModernVBERT. It performs a pixel shuffle operation followed by a linear projection to match the text model's hidden size.
    Based on https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
    """

    def __init__(self, config):
        super().__init__()
        self.pixel_shuffle_factor = config.pixel_shuffle_factor
        self.modality_projection = ModernVBertSimpleMLP(
            input_size=config.vision_config.hidden_size * (config.pixel_shuffle_factor**2),
            output_size=config.text_config.hidden_size,
        )

    def pixel_shuffle(self, x, pixel_shuffle_factor):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / pixel_shuffle_factor), embed_dim * pixel_shuffle_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / pixel_shuffle_factor), int(height / pixel_shuffle_factor), embed_dim * (pixel_shuffle_factor**2))
        x = x.permute(0, 2, 1, 3)
        return x.reshape(bsz, int(seq / (pixel_shuffle_factor**2)), embed_dim * (pixel_shuffle_factor**2))

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.pixel_shuffle_factor)
        return self.modality_projection(image_hidden_states)


class ModernVBertPreTrainedModel(PreTrainedModel):
    config_class = ModernVBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@auto_docstring
class ModernVBertModel(ModernVBertPreTrainedModel):
    def __init__(self, config: ModernVBertConfig):
        super().__init__(config)

        # init components
        self.vision_model = ModernVBertModel.init_vision_model(config)
        self.connector = ModernVBertConnector(config)
        self.text_model = ModernVBertModel.init_language_model(config)

        # set the correct dtype for vision and text models
        self.vision_model.to(self.dtype)
        self.text_model.to(self.dtype)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        
        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2) / (config.pixel_shuffle_factor**2)
        )

        self.post_init()

    @staticmethod
    def init_vision_model(config: ModernVBertConfig):
        vision_model_config = SiglipVisionConfig.from_pretrained(
            config.vision_config.vision_model_name,
            _attn_implementation=config._attn_implementation,
        )
        vision_model = SiglipVisionModel(vision_model_config).vision_model
        return vision_model

    @staticmethod
    def init_language_model(config: ModernVBertConfig):
        text_model_config = ModernBertConfig.from_pretrained(
            config.text_config.text_model_name,
            _attn_implementation=config._attn_implementation,
        )
        text_model = ModernBertModel(text_model_config)
        embed_layer = DecoupledEmbedding(
            num_embeddings=text_model_config.vocab_size,
            num_additional_embeddings=config.additional_vocab_size,
            embedding_dim=config.hidden_size,
            partially_freeze=getattr(config, "freeze_config", {"freeze_text_layers": False})["freeze_text_layers"],
            padding_idx=config.pad_token_id,
        )
        text_model.set_input_embeddings(embed_layer)
        return text_model

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2Model.enable_input_require_grads
    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings.

        This is useful for lora when using gradient checkpointing.
        c.f. https://github.com/huggingface/peft/issues/1402#issuecomment-1913675032

        Override to set output.requires_grad = True for both the decoder's and vision model's embeddings.
        """

        def get_lowest_module(module):
            if len(list(module.children())) == 0:
                # If the module has no children, it is a leaf module (e.g., Linear, Conv2d, etc.)
                return module
            else:
                # Recursively call the function on each child module
                return get_lowest_module(list(module.children())[0])

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = get_lowest_module(self.vision_model).register_forward_hook(
            make_inputs_require_grads
        )

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2Model.disable_input_require_grads
    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def get_image_features(
        self, pixel_values: torch.FloatTensor, pixel_attention_mask: Optional[torch.LongTensor] = None
    ):
        """
        Derived from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/modeling_smolvlm.py
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            pixel_attention_mask (`torch.LongTensor`, *optional*):
                The attention mask indicating padded regions in the image.
        """
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

        if not any(real_images_inds):
            real_images_inds[0] = True

        pixel_values = pixel_values[real_images_inds].contiguous()
        # Handle the vision attention mask
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=[pixel_values.shape[i] for i in (0, 2, 3)],
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            # Remove padding images from the mask
            pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Get sequence from the vision encoder
        image_hidden_states = self.vision_model(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)
        image_hidden_states = image_hidden_states.last_hidden_state

        return image_hidden_states

    def inputs_merger(self, input_ids, inputs_embeds, image_hidden_states):
        """Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/smolvlm/modeling_smolvlm.py

        This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
        The merging happens as follows:
        - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
        - We get the image hidden states for the image through the vision encoder and that hidden state, after a pixel shuffle operation, is then projected into the text embedding space.
        We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
        - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
        - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
        """

        _, patch_size, _ = image_hidden_states.shape

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask[..., 0]  # slice off the hidden dim
        else:
            image_mask = input_ids == self.config.image_token_id

        # Assert that the input <image> tokens are valid (i.e. multiple of patch_size)
        num_image_tokens = image_mask.sum(dim=1)
        if not torch.all(num_image_tokens % patch_size == 0):
            raise ValueError("Number of <image> tokens not divisible by patch_size.")
        
        blocks_per_sample = num_image_tokens // patch_size

        offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
        block_offset = offsets[:-1]
        row_cum = image_mask.cumsum(dim=-1)
        chunk_idx = (row_cum - 1) // patch_size
        local_idx = (row_cum - 1) % patch_size
        block_idx = block_offset.unsqueeze(1) + chunk_idx
        
        image_embeds = torch.zeros_like(inputs_embeds)
        image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]
        
        return torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds)

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        checkpoint="modernvbert/ModernVBert",
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(input_ids.device)

        # Images processing
        if pixel_values is not None:
            # Vision encoder pass
            image_hidden_states = self.get_image_features(
                pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask
            )
            # Modality projection & resampling
            image_hidden_states = self.connector(image_hidden_states)

        # Merge image and text embeddings
        if image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=inputs_embeds.device)
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids, inputs_embeds=inputs_embeds, image_hidden_states=image_hidden_states
            )

        # Language model pass
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return ModernVBertBaseModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


class ModernVBertLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        pretrained_config = ModernBertConfig.from_pretrained(config.text_config.text_model_name)
        pretrained_model = ModernBertForMaskedLM(pretrained_config)
        self.head = pretrained_model.head
        self.decoder = pretrained_model.decoder

    def forward(self, hidden_states):
        return self.decoder(self.head(hidden_states))


@auto_docstring
class ModernVBertForMaskedLM(ModernVBertPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "model.text_model.embeddings.word_embeddings.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.in_features = config.hidden_size
        self.out_additional_features = config.additional_vocab_size
        self.vocab_size = config.vocab_size
        self.model = ModernVBertModel(config)
        self.lm_head = ModernVBertLMHead(config)
        if self.out_additional_features > 0:
            self.additional_fc = nn.Linear(self.in_features, self.out_additional_features, bias=False)
        self.lm_head.to(self.dtype)
        self.post_init()

    # Copied from transformers.models.idefics2.modeling_idefics2.Idefics2ForConditionalGeneration.disable_input_require_grads
    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        checkpoint="modernvbert/ModernVBert",
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, ModernVBertMaskedLMOutput]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id`. Tokens with indices set to `model.image_token_id` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        if self.out_additional_features > 0:
            proj_states = self.lm_head.head(hidden_states)
            additional_features = self.additional_fc(proj_states)
            logits = torch.cat((logits, additional_features), -1)

        loss = None
        if labels is not None:
            loss = CrossEntropyLoss()(logits.view(-1, self.vocab_size + self.out_additional_features), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ModernVBertMaskedLMOutput(
            loss=loss,
            logits=logits.float(),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

__all__ = [
    "ModernVBertImageProcessor",
    "ModernVBertImageProcessorFast",
    "ModernVBertProcessor",
    "ModernVBertConfig",
    "ModernVBertTextConfig",
    "ModernVBertVisionConfig",
    "ModernVBertPreTrainedModel",
    "ModernVBertModel",
    "ModernVBertForMaskedLM",
]
