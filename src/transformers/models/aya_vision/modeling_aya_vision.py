# coding=utf-8
# Copyright 2025 the Cohere Inc. team. All rights reserved.
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
"""PyTorch AyaVision model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_aya_vision import AyaVisionConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "AyaVisionConfig"


@dataclass
class AyaVisionCausalLMOutputWithPast(ModelOutput):
    """
    Base class for AyaVision causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size (batch_size * num_patches, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


# Copied from transformers.models.cohere.modeling_cohere.CohereMultiModalProjector with Cohere->AyaVision
class AyaVisionMultiModalProjector(nn.Module):
    def __init__(self, config: AyaVisionConfig):
        super().__init__()
        self.config = config
        self.downsample_factor = config.downsample_factor
        self.alignment_intermediate_size = getattr(
            config, "alignment_intermediate_size", config.text_config.hidden_size
        )
        self.act_fn = getattr(config, "alignment_activation_fn", "gelu")
        self.layernorm = nn.LayerNorm(
            config.vision_config.hidden_size * (config.downsample_factor**2), eps=config.adapter_layer_norm_eps
        )

        # Same intermediate size for both SwiGLU and regular activation
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * (config.downsample_factor**2),
            self.alignment_intermediate_size,
            bias=True,
        )

        if self.act_fn == "swiglu":
            self.act = ACT2FN["silu"]  # SwiGLU uses SiLU activation
            # For SwiGLU, project down to half size since we split intermediate dim
            self.linear_2 = nn.Linear(self.alignment_intermediate_size // 2, config.text_config.hidden_size, bias=True)
        else:
            self.act = ACT2FN[config.projector_hidden_act]
            self.linear_2 = nn.Linear(self.alignment_intermediate_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        image_features = self.pixel_shuffle(image_features)
        image_features = self.layernorm(image_features)
        hidden_states = self.linear_1(image_features)

        if self.act_fn == "swiglu":
            # Split along last dimension and apply SwiGLU
            x, gate = hidden_states.chunk(2, dim=-1)
            hidden_states = self.act(gate) * x
        else:
            hidden_states = self.act(hidden_states)

        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def pixel_shuffle(self, image_features):  # B, S, D
        b, s, d = image_features.shape
        h = w = int(s**0.5)
        image_features = image_features.reshape(image_features.shape[0], w, h, -1)
        c = image_features.shape[-1]
        image_features = image_features.reshape(b, w, int(h / self.downsample_factor), int(c * self.downsample_factor))
        image_features = image_features.permute(0, 2, 1, 3)
        image_features = image_features.reshape(
            b, int(h / self.downsample_factor), int(w / self.downsample_factor), -1
        )
        image_features = image_features.permute(0, 2, 1, 3)
        return image_features


AYA_VISION_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`AyaVisionConfig`] or [`AyaVisionVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Aya Vision Model outputting raw hidden-states without any specific head on top.",
    AYA_VISION_START_DOCSTRING,
)
# Copied from transformers.models.cohere.modeling_cohere.CoherePreTrainedModel with Cohere->AyaVision
class AyaVisionPreTrainedModel(PreTrainedModel):
    config_class = AyaVisionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AyaVisionVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        # important: this ported version of AyaVision isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


AYA_VISION_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`GotOcr2ImageProcessor.__call__`] for details. [`CohereProcessor`] uses
            [`GotOcr2ImageProcessor`] for processing images.
        image_sizes (`torch.LongTensor` of shape `(batch_size, 2)`, *optional*):
            The sizes of the images in the batch, being (height, width) for each image.
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
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    """The AyaVision model which consists of a vision backbone and a language model.""",
    AYA_VISION_START_DOCSTRING,
)
class AyaVisionModel(AyaVisionPreTrainedModel, GenerationMixin):
    def __init__(self, config: AyaVisionConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = AyaVisionMultiModalProjector(config)

        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        self.image_token = self.config.image_token
        self.post_init()

    @property
    def padding_side(self):
        return self._padding_side

    @padding_side.setter
    def padding_side(self, padding_side: str):
        if padding_side not in ["left", "right"]:
            raise ValueError(f"{padding_side} is not `left` or `right`.")
        self._padding_side = padding_side

    # Copied from transformers.models.cohere.modeling_cohere.AyaVisionForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.cohere.modeling_cohere.AyaVisionForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.cohere.modeling_cohere.AyaVisionForConditionalGeneration.get_output_embeddings
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.cohere.modeling_cohere.AyaVisionForConditionalGeneration.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.cohere.modeling_cohere.AyaVisionForConditionalGeneration.set_decoder
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.cohere.modeling_cohere.AyaVisionForConditionalGeneration.get_decoder
    def get_decoder(self):
        return self.language_model.get_decoder()

    # Copied from transformers.models.cohere.modeling_cohere.AyaVisionForConditionalGeneration.tie_weights
    def tie_weights(self):
        return self.language_model.tie_weights()

    # Copied from transformers.models.cohere.modeling_cohere.AyaVisionForConditionalGeneration.resize_token_embeddings
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_num_patches: torch.Tensor,
        vision_feature_layer: int,
        vision_feature_select_strategy: str,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_patches, channels, height, width)`)
               The tensors corresponding to the input images.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            vision_feature_layer (`int`):
                The index of the layer to select the vision feature.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (List[`torch.Tensor`]): List of image feature tensor, each contains all the visual feature of all patches
            and are of shape `(num_patches, image_length, embed_dim)`).
        """

        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches.tolist(), dim=0)
        # pad image_features to the same length and stack them
        padded_image_features = []
        max_patch_len = max([img.shape[0] for img in image_features])
        for img in image_features:
            padded_image_features.append(
                torch.cat(
                    [
                        img,
                        torch.zeros(max_patch_len - img.shape[0], *img.shape[1:], device=img.device, dtype=img.dtype),
                    ],
                    dim=0,
                )
            )
        padded_image_features = torch.stack(padded_image_features, dim=0)
        return padded_image_features

    @add_start_docstrings_to_model_forward(AYA_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AyaVisionCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_num_patches: Optional[torch.LongTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: int = 0,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, AyaVisionCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

            Returns:
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features = self.get_image_features(
                pixel_values,
                image_num_patches,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

        if image_features is not None:
            inputs_embeds = self._merge_image_text_embeddings(input_ids, image_features, inputs_embeds)

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
            num_logits_to_keep=num_logits_to_keep,
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

        return AyaVisionCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_num_patches=None,
        image_sizes=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = image_sizes
            model_inputs["image_num_patches"] = image_num_patches

        return model_inputs

    def _merge_image_text_embeddings(self, input_ids, image_features, inputs_embeds):
        is_img_patch = input_ids == self.image_token
        image_encoding_to_sequence_map = (is_img_patch.cumsum(axis=1) * is_img_patch).to(torch.int64)  # [bsz, seqlen]
        if len(image_features.shape) == 4:
            image_features = image_features.unsqueeze(0)  # add batch dimension
        B, I, H, W, D = image_features.shape
        image_features = image_features.reshape(B, I * W * H, D)
        pad_tensor = torch.zeros((B, 1, D), dtype=image_features.dtype, device=image_features.device)
        image_features = torch.cat((pad_tensor, image_features), dim=1)
        # pad to the sequence length of inputs_embeds
        image_features = torch.cat(
            [image_features, torch.zeros_like(inputs_embeds[:, : -image_features.size(1)])], dim=1
        )
        batch_indices = torch.arange(B).unsqueeze(1).expand(B, image_encoding_to_sequence_map.shape[1])
        image_features = image_features[batch_indices, image_encoding_to_sequence_map]
        output = inputs_embeds * (~is_img_patch).unsqueeze(-1) + image_features * is_img_patch.unsqueeze(-1)
        return output


__all__ = ["AyaVisionModel", "AyaVisionPreTrainedModel"]
