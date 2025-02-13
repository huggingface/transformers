# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers.models.llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaPreTrainedModel,
)
from transformers.models.sam.modeling_sam import SamMLPBlock, SamVisionAttention, SamVisionEncoder, SamVisionLayer

from ...configuration_utils import PretrainedConfig
from ...utils import (
    add_start_docstrings_to_model_forward,
    is_vision_available,
    logging,
    replace_return_docstrings,
)
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModelForCausalLM


if is_vision_available():
    pass

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GotOcr2Config"


class GotOcr2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GotOcr2VisionModel`]. It is used to instantiate a GOT_OCR2
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of the SAM ViT-h
    [facebook/sam-vit-huge](https://huggingface.co/facebook/sam-vit-huge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        output_channels (`int`, *optional*, defaults to 256):
            Dimensionality of the output channels in the Patch Encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input image.
        image_size (`int`, *optional*, defaults to 1024):
            Expected resolution. Target size of the resized input image.
        patch_size (`int`, *optional*, defaults to 16):
            Size of the patches to be extracted from the input image.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string)
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to query, key, value projections.
        use_abs_pos (`bool`, *optional*, defaults to `True`):
            Whether to use absolute position embedding.
        use_rel_pos (`bool`, *optional*, defaults to `True`):
            Whether to use relative position embedding.
        window_size (`int`, *optional*, defaults to 14):
            Window size for relative position.
        global_attn_indexes (`List[int]`, *optional*, defaults to `[2, 5, 8, 11]`):
            The indexes of the global attention layers.
        mlp_dim (`int`, *optional*, defaults to 3072):
            The dimensionality of the MLP layer in the Transformer encoder.
    """

    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        output_channels=256,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=1024,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-06,
        attention_dropout=0.0,
        initializer_range=1e-10,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        global_attn_indexes=[2, 5, 8, 11],
        mlp_dim=3072,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.qkv_bias = qkv_bias
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.mlp_dim = mlp_dim


class GotOcr2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GotOcr2ForConditionalGeneration`]. It is used to instantiate a
    GotOcr2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of GOT-OCR-2.0.

    e.g [stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 151859):
            The image token index to encode the image prompt.
        image_seq_length (`int`, *optional*, defaults to 576):
            Sequence length of one image embedding.
        pad_token_id (`int`, *optional*, defaults to -1):
            Padding token id.

    ```python
    >>> from transformers import GotOcr2ForConditionalGeneration, GotOcr2Config

    >>> # Initializing a GotOcr2 style configuration
    >>> configuration = GotOcr2Config()

    >>> # Initializing a model from the Qwen2-VL-7B style configuration
    >>> model = GotOcr2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "got_ocr2"
    sub_configs = {"text_config": AutoConfig, "vision_config": GotOcr2VisionConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=151859,
        image_seq_length=576,
        pad_token_id=-1,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.image_seq_length = image_seq_length
        self.pad_token_id = pad_token_id

        if vision_config is None:
            self.vision_config = GotOcr2VisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = GotOcr2VisionConfig(**vision_config)
        elif isinstance(vision_config, GotOcr2VisionConfig):
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen2"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"](
                vocab_size=151860,
                hidden_size=1024,
                intermediate_size=2816,
                num_hidden_layers=24,
                num_attention_heads=16,
                num_key_value_heads=16,
                hidden_act="silu",
                max_position_embeddings=32768,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=True,
                tie_word_embeddings=True,
                rope_theta=1000000.0,
                rope_scaling=None,
                use_sliding_window=False,
                sliding_window=4096,
                max_window_layers=21,
                attention_dropout=0.0,
            )

        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["GotOcr2VisionConfig", "GotOcr2Config"]


class GotOcr2MLPBlock(SamMLPBlock):
    pass


class GotOcr2VisionAttention(SamVisionAttention):
    pass


class GotOcr2VisionLayer(SamVisionLayer):
    def __init__(self, config, window_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GotOcr2VisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GotOcr2MLPBlock(config)
        self.window_size = window_size


class GotOcr2VisionEncoder(SamVisionEncoder):
    pass


class GotOcr2MultiModalProjector(nn.Module):
    def __init__(self, config: GotOcr2Config):
        super().__init__()
        vision_output_channels = config.vision_config.output_channels
        language_hidden_size = config.text_config.hidden_size
        self.conv_upsampler1 = nn.Conv2d(
            vision_output_channels, vision_output_channels * 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv_upsampler2 = nn.Conv2d(
            vision_output_channels * 2, language_hidden_size, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.multimodal_projector = nn.Linear(language_hidden_size, language_hidden_size)

    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv_upsampler1(vision_embeddings)
        hidden_state = self.conv_upsampler2(hidden_state)
        hidden_state = hidden_state.flatten(2).permute(0, 2, 1)
        hidden_state = self.multimodal_projector(hidden_state)
        return hidden_state


class GotOcr2CausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class GotOcr2PreTrainedModel(LlavaPreTrainedModel):
    pass


GOT_OCR2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        pixel_values (`torch.FloatTensor` of shape `(seq_length, num_channels * image_size * image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`GotOcr2ImageProcessor.__call__`] for details. [`GotOcr2Processor`] uses
            [`GotOcr2ImageProcessor`] for processing images.
"""


class GotOcr2ForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: GotOcr2Config):
        super().__init__(config)
        self.vision_tower = GotOcr2VisionEncoder(config.vision_config)

        self.multi_modal_projector = GotOcr2MultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        self.pad_token_id = config.pad_token_id

        self.post_init()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        image_outputs = self.vision_tower(pixel_values).last_hidden_state
        return self.multi_modal_projector(image_outputs)

    @add_start_docstrings_to_model_forward(GOT_OCR2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GotOcr2CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).


        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, GotOcr2ForConditionalGeneration, TextStreamer

        >>> model = GotOcr2ForConditionalGeneration.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf").to("cuda")
        >>> processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

        >>> url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/multi_box.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(image, return_tensors="pt", color="green").to("cuda")

        >>> # Generate
        >>> streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        >>> generate_ids = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     tokenizer = processor.tokenizer,
        ...     stop_strings='<|im_end|>',
        ...     streamer=streamer,
        ...     max_new_tokens=4096,
        ... )
        "You should keep in mind what features from the module should be used, especially
        when you're planning to sell a template."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values=pixel_values.to(inputs_embeds.dtype))
            n_image_tokens = (input_ids == self.config.image_token_index).sum()
            n_image_features = image_features.shape[0] * image_features.shape[1]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return GotOcr2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


__all__ = [
    "GotOcr2VisionConfig",
    "GotOcr2Config",
    "GotOcr2PreTrainedModel",
    "GotOcr2ForConditionalGeneration",
]
