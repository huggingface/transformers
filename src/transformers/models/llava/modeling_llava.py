# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Llava model."""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ... import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_outputs import ModelOutput
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_llava import LlavaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlavaConfig"

LLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "llava/llava-v1.5-7b",
    "llava/llava-v1.5-13b",
    # See all Llava models at https://huggingface.co/models?filter=llava
]


# Helper function to handle padding based on configuration
def pad_sequence(sequence, max_len, padding_side="right"):
    cur_len = sequence.shape[0]
    if padding_side == "left":
        return torch.cat(
            (
                torch.zeros((max_len - cur_len, sequence.shape[1]), dtype=sequence.dtype, device=sequence.device),
                sequence,
            ),
            dim=0,
        )
    else:
        return torch.cat(
            (
                sequence,
                torch.zeros((max_len - cur_len, sequence.shape[1]), dtype=sequence.dtype, device=sequence.device),
            ),
            dim=0,
        )

@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->Llava
class LlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

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
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


LLAVA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlavaConfig`] or [`LlavaVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAVA_START_DOCSTRING,
)
class LlavaPreTrainedModel(PreTrainedModel):
    config_class = LlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
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


LLAMA_INPUTS_DOCSTRING = r"""
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
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """The LLAVA model which consists of a vision backbone and a language model.""",
    LLAVA_START_DOCSTRING,
)
class LlavaForVisionText2Text(LlavaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.vision_tower._no_split_modules = ["CLIPEncoderLayer"]
        self.multi_modal_projector = LlavaMultiModalProjector(config)

        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=LlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
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
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlavaForVisionText2Text

        >>> model = LlavaForVisionText2Text.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, what's the best tomato based dish?!"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, what's the best tomato based dish?!"
        ```"""

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

        if inputs_embeds is None:
            if pixel_values is None or input_ids.shape[1] == 1:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation without image information
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    target_seqlen = past_key_values[-1][-1].shape[-2] + 1

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            else:
                image_outputs = self.vision_tower(
                    pixel_values, output_hidden_states=True
                )  # this is not memory efficient at all
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                image_features = self.multi_modal_projector(selected_image_feature)

                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                else:
                    attention_mask = attention_mask.bool()

                if position_ids is None:
                    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

                if labels is None:
                    labels = torch.full_like(input_ids, self.config.ignore_index)

                # We need to initiliaze these new arrays to process the inputs embed and labels
                # to correctly deal with all case (e.g. single prompt & multi-image tokens)
                processed_inputs_embeds = []
                processed_labels = []

                # We can have multiple images in a single batch, hence we use different
                # indexes for image and text.
                current_image_index = 0

                # Since we might image tokens in different places of the input string, we need to
                # process that one by one, batch element per batch element
                for batch_idx, (current_input_ids, current_labels) in enumerate(zip(input_ids, labels)):
                    # Get the number of the image tokens in the current prompt.
                    num_image_tokens = (current_input_ids == self.config.image_token_index).sum()

                    if num_image_tokens == 0:
                        # If there is not image tokens, simply use the text tokens embeddings
                        current_inputs_embed = self.get_input_embeddings()(current_input_ids)
                        processed_inputs_embeds.append(current_inputs_embed)
                        processed_labels.append(labels[batch_idx])
                        current_image_index += 1
                        continue

                    input_ids_without_image_token, labels_without_image_token = self._split_input_ids_and_labels(
                        current_input_ids, current_labels, self.config.image_token_index
                    )

                    split_index = torch.where(current_input_ids == self.config.image_token_index)[0][0].item()
                    split_sizes = [split_index, labels_without_image_token.shape[-1] - split_index]

                    inputs_embed_without_image = self.get_input_embeddings()(input_ids_without_image_token)

                    inputs_embed_without_image = torch.split(inputs_embed_without_image, split_sizes, dim=0)
                    labels_without_image_token = torch.split(labels_without_image_token, split_sizes, dim=0)

                    current_processed_inputs_embed = []
                    current_processed_labels = []

                    for i in range(num_image_tokens + 1):
                        current_processed_inputs_embed.append(inputs_embed_without_image[i])
                        current_processed_labels.append(labels_without_image_token[i])

                        if i < num_image_tokens:
                            current_image_features = image_features[current_image_index]
                            current_image_index += 1

                            current_processed_inputs_embed.append(current_image_features)
                            current_processed_labels.append(
                                torch.full(
                                    (current_image_features.shape[0],),
                                    self.config.ignore_index,
                                    device=current_labels.device,
                                    dtype=current_labels.dtype,
                                )
                            )

                    processed_inputs_embeds.append(torch.cat(current_processed_inputs_embed))
                    processed_labels.append(torch.cat(current_processed_labels))

                # Truncate sequences to max length as image embeddings can make the sequence longer
                tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
                if tokenizer_model_max_length is not None:
                    processed_inputs_embeds = [x[:tokenizer_model_max_length] for x in processed_inputs_embeds]
                    processed_labels = [x[:tokenizer_model_max_length] for x in processed_labels]

                # Combine them
                max_len = max(x.shape[0] for x in processed_inputs_embeds)
                batch_size = len(processed_inputs_embeds)

                attention_mask = torch.zeros(
                    (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
                )
                position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

                inputs_embeds, attention_mask, position_ids, labels = self._optionally_pad_input_embeds(
                    processed_inputs_embeds, attention_mask, position_ids, processed_labels, max_len, batch_size
                )

                # Set input_ids to None so that the LM will only use input_embeds
                input_ids = None

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _optionally_pad_input_embeds(
        self, inputs_embeds, attention_mask, position_ids, labels, max_seqlen, batch_size
    ):
        r"""
        Optionally pad the input embeddings by correctly setting the padding tokens
        on the correct places inside the attention mask, position ids and labels.
        """
        padded_inputs_embeds = []
        padded_labels = torch.full(
            (batch_size, max_seqlen),
            self.config.ignore_index,
            dtype=labels[0].dtype,
            device=labels[0].device,
        )

        for i, (current_embeds, cur_new_labels) in enumerate(zip(inputs_embeds, labels)):
            # Get the current sequence length and padding side
            # then optionally padd the input embeds
            current_seq_len = current_embeds.shape[0]
            padding_side = getattr(self.config, "tokenizer_padding_side", "right")
            padded_embedding = pad_sequence(current_embeds, max_seqlen, padding_side)

            padded_inputs_embeds.append(padded_embedding)

            if current_seq_len > 0:
                start_index = -current_seq_len if padding_side == "left" else 0
                end_index = None if padding_side == "left" else current_seq_len

                padded_labels[i, start_index:end_index] = cur_new_labels
                attention_mask[i, start_index:end_index] = True
                position_ids[i, start_index:end_index] = torch.arange(
                    0, current_seq_len, dtype=position_ids.dtype, device=position_ids.device
                )

        inputs_embeds = torch.stack(padded_inputs_embeds, dim=0)
        return inputs_embeds, attention_mask, position_ids, padded_labels

    def _split_input_ids_and_labels(self, input_ids, labels, image_token_id):
        """
        Simple method to split the input ids and labels in 2: the parts before / after `image_token_id`.
        """
        exclude_mask = input_ids != image_token_id

        input_ids_without_image_token = input_ids[exclude_mask]
        labels_without_image_token = labels[exclude_mask]

        return input_ids_without_image_token, labels_without_image_token

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # Pop the pixel_values to pick it up later
        pixel_values = kwargs.pop("pixel_values", None)

        # Call `prepare_inputs_for_generation` from the LM
        model_input = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        # Put back the `pixel_values` in `model_input`
        model_input.update({"pixel_values": pixel_values})
        return model_input
