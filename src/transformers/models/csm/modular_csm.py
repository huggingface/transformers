# coding=utf-8
# Copyright 2025 Sesame and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from transformers.utils.generic import check_model_inputs

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, auto_docstring, can_return_tuple, logging
from ..auto import AutoModel
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    TransformersKwargs,
)
from .configuration_csm import CsmConfig, CsmDepthDecoderConfig
from .generation_csm import CsmGenerationMixin


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for the model autoregressive outputs.
    """
)
class CsmOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    depth_decoder_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction) of the depth decoder model.
    depth_decoder_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the depth decoder (scores for each vocabulary token before SoftMax).
    depth_decoder_past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
    depth_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

        Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    depth_decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`.
    backbone_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction) of the backbone model.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    depth_decoder_loss: Optional[torch.FloatTensor] = None
    depth_decoder_logits: torch.FloatTensor = None
    depth_decoder_past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None
    depth_decoder_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    depth_decoder_attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    backbone_loss: Optional[torch.FloatTensor] = None


# manually specify names for correct naming when converting from modualr
class CsmRMSNorm(LlamaRMSNorm):
    pass


class CsmRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class CsmMLP(LlamaMLP):
    pass


class CsmAttention(LlamaAttention):
    pass


class CsmDecoderLayer(LlamaDecoderLayer):
    pass


@auto_docstring(
    custom_intro="""
    The bare Csm Model outputting raw hidden-states without any specific head on top.
    """
)
@auto_docstring
class CsmPreTrainedModel(PreTrainedModel):
    config: CsmConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CsmDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    # does not because of Mimi codec model
    # _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": CsmDecoderLayer,
        "attentions": CsmAttention,
    }

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, CsmCodebooksHead):
            num_codebooks = module.num_codebooks
            for i in range(num_codebooks - 1):
                module.weight.data[i].normal_(mean=0.0, std=self.config.initializer_range)


@auto_docstring
class CsmDepthDecoderModel(LlamaModel, CsmPreTrainedModel):
    config: CsmDepthDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding((config.num_codebooks * config.vocab_size), config.backbone_hidden_size)
        self.inputs_embeds_projector = nn.Linear(config.backbone_hidden_size, config.hidden_size, bias=False)

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        backbone_last_hidden_state: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        backbone_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, backbone_hidden_size)`, *optional*):
            The last hidden state of the backbone model. Such input is required when the first codebook token (the one generated by the backbone model)
            is provided in the `input_ids` argument.
        """
        if position_ids is not None and not torch.compiler.is_compiling():
            logger.warning_once(
                "Custom `position_ids` were provided but will be ignored. CSM depth decoder automatically determines position_ids "
                "from `cache_position` and as it requires them to be identical across the batch, the provided position_ids will be ignored."
            )
            position_ids = None
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            inputs_seq_length = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_seq_length, device=device)

        if inputs_embeds is None:
            codebook_idxs = torch.clamp(cache_position - 1, min=0)
            offset = codebook_idxs * self.vocab_size
            inputs_embeds = self.embed_tokens(input_ids + offset)

            input_ids_are_first_codebook = cache_position[0] == 0
            if backbone_last_hidden_state is not None:
                inputs_embeds[:, 0] = backbone_last_hidden_state
            else:
                if not torch.compiler.is_compiling() and input_ids_are_first_codebook:
                    logger.warning(
                        "When the first codebook token is provided, `backbone_last_hidden_state` should also be provided for correct inference."
                    )

        inputs_embeds = self.inputs_embeds_projector(inputs_embeds)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class CsmCodebooksHead(nn.Module):
    def __init__(self, hidden_size, num_codebooks, vocab_size):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.weight = nn.Parameter(torch.empty(self.num_codebooks - 1, hidden_size, vocab_size))

    def forward(self, hidden_states, cache_position=None):
        if cache_position is None:
            seq_length = hidden_states.shape[1]
            codebook_weight = self.weight[torch.arange(seq_length)]
        else:
            codebook_idxs = cache_position - 1
            codebook_weight = self.weight[codebook_idxs]

        hidden_states = [
            nn.functional.linear(hidden_states[:, codebook_idx, :], codebook_weight[codebook_idx].T)
            for codebook_idx in range(codebook_weight.shape[0])
        ]
        hidden_states = torch.stack(hidden_states, dim=1)

        return hidden_states


@auto_docstring(
    custom_intro="""
    The CsmDepthDecoder Model transformer, with a [`CsmCodebooksHead`] on top,
    which can be seen a position-specific language modeling head, allowing to use a different linear layer for each codebook
    (e.g. position 0 is the first codebook and uses the first codebook head, etc.)
    """
)
class CsmDepthDecoderForCausalLM(LlamaForCausalLM, GenerationMixin):
    _tied_weights_keys = None
    _tp_plan = None
    _pp_plan = None

    def __init__(self, config):
        super().__init__(config)
        del self.lm_head
        self.codebooks_head = CsmCodebooksHead(config.hidden_size, config.num_codebooks, config.vocab_size)
        self.model = CsmDepthDecoderModel(config)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, cache_position, **kwargs
        )

        is_first_generation_step = model_inputs["cache_position"][0] == 0
        if not is_first_generation_step:
            model_inputs.pop("backbone_last_hidden_state")

        # csm depth decoder does not use position_ids
        model_inputs.pop("position_ids")

        return model_inputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        backbone_last_hidden_state: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        r"""
        backbone_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, backbone_hidden_size)`, *optional*):
            The last hidden state of the backbone model. Such input is required when the first codebook token (the one generated by the backbone model)
            is provided in the `input_ids` argument.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            backbone_last_hidden_state=backbone_last_hidden_state,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        if isinstance(logits_to_keep, int):
            if logits_to_keep == 0:
                # skip idx 0 logits since it's for the concatenated backbone last hidden state
                slice_indices = slice(1, None)
            else:
                slice_indices = slice(-logits_to_keep, None)
        else:
            slice_indices = logits_to_keep

        logits = self.codebooks_head(
            hidden_states[:, slice_indices, :], cache_position[slice_indices] if cache_position is not None else None
        )
        logits = logits.contiguous()

        loss = None
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_function(
                logits=logits, labels=None, vocab_size=self.config.vocab_size, shift_labels=shift_labels, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CsmBackboneModelEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_audio_tokens = nn.Embedding((config.num_codebooks * config.vocab_size), config.hidden_size)
        self.register_buffer(
            "audio_tokens_offsets", torch.arange(config.num_codebooks) * config.vocab_size, persistent=False
        )

    def forward(self, input_ids):
        input_embeds = self.embed_audio_tokens(input_ids + self.audio_tokens_offsets)
        input_embeds = input_embeds.sum(dim=2)
        return input_embeds


@auto_docstring
class CsmBackboneModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = CsmBackboneModelEmbeddings(config)

    @check_model_inputs
    @auto_docstring
    def forward(self, **super_kwargs):
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks) or (batch_size, sequence_length)`):
            1. (batch_size, sequence_length): corresponds to the input sequence prepared with the processor from the text prompt. Such input
            requires `input_values` to be provided so that audio can be encoded in codebook tokens and then merged with the text tokens.

            2. (batch_size, sequence_length, num_codebooks): codebook tokens generated during the autoregressive decoding. Such input is not meant to be used by end users.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        return super().forward(**super_kwargs)


@auto_docstring(
    custom_intro="""
    The Csm model consists of two llama-like auto-regressive transformer models: a backbone model that predicts the first codebook token and a depth decoder that predicts the other codebook tokens.
    """
)
class CsmForConditionalGeneration(CsmPreTrainedModel, CsmGenerationMixin):
    _tied_weights_keys = [
        "backbone_model.embed_tokens.embed_audio_tokens.weight",
        "depth_decoder.model.embed_tokens.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embed_text_tokens = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.backbone_model = CsmBackboneModel._from_config(config)
        self.depth_decoder = CsmDepthDecoderForCausalLM._from_config(config.depth_decoder_config)
        self.codec_model = AutoModel.from_config(config.codec_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone_model.embed_tokens

    def set_input_embeddings(self, value):
        self.backbone_model.embed_tokens = value

    def _tie_weights(self):
        if self.config.tie_codebooks_embeddings:
            self._tie_or_clone_weights(
                self.backbone_model.embed_tokens.embed_audio_tokens,
                self.depth_decoder.model.embed_tokens,
            )

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if kwargs.get("output_loading_info", False):
            model, loading_info = super().from_pretrained(*args, **kwargs)
        else:
            model = super().from_pretrained(*args, **kwargs)

        # copy depth decoder generation conf attr to the depth decoder generation config
        prefix = "depth_decoder_"
        prefix_len = len(prefix)
        depth_decoder_attrs = {
            attr[prefix_len:]: value
            for attr, value in vars(model.generation_config).items()
            if attr.startswith(prefix)
        }

        vars(model.depth_decoder.generation_config).update({"_from_model_config": False, **depth_decoder_attrs})

        # remove the depth decoder generation conf attr from the model generation config
        for attr in depth_decoder_attrs:
            delattr(model.generation_config, prefix + attr)

        if "output_loading_info" in kwargs:
            return model, loading_info
        else:
            return model

    def save_pretrained(self, *args, **kwargs):
        # copy the depth decoder generation config attributes to the model generation config
        prefix = "depth_decoder_"
        depth_decoder_attrs = self.depth_decoder.generation_config.to_diff_dict()
        depth_decoder_attrs.pop("transformers_version", None)
        for attr, value in depth_decoder_attrs.items():
            setattr(self.generation_config, prefix + attr, value)

        super().save_pretrained(*args, **kwargs)

    def _merge_input_ids_with_input_values(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Merges the input_ids and input_values to produce a single inputs_embeds tensor:
        1 - Infers the codec model on the input_values to retreive codebook token.
        2 - Embeds codebook tokens and places them at the correct positions in the inputs_embeds tensor.
        3 - If labels are provided, expands them to match codebook dimensions and position the target codebook tokens in the inputs_embeds tensor.

        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The input ids to embed.
            input_values (`torch.Tensor` of shape `(batch_size, channels, audio_sequence_length)`):
                The audio input values to embed.
            input_values_cutoffs (`torch.Tensor` of shape `(batch_size, max_num_audio)`):
                The cutoffs of the audio input values relative to its batch index, padded with -1 when no audio.
        """
        inputs_embeds = self.embed_text_tokens(input_ids)

        if input_values is not None:
            # infer input_values_mask
            input_values_cutoffs = nn.functional.pad(input_values_cutoffs, (1, 0))
            audio_lengths = input_values_cutoffs[input_values_cutoffs >= 0].diff()
            audio_lengths = audio_lengths[audio_lengths > 0]
            input_values_mask = torch.arange(input_values_cutoffs.max(), device=input_values.device).expand(
                len(audio_lengths), -1
            )
            input_values_mask = input_values_mask < audio_lengths.unsqueeze(1)

            # =======================================
            # TODO: @eustlb, this should be batched !!!
            # but requires making sure batched inference of the codec model works as intended
            with torch.no_grad():
                audio_tokens_list = []
                for batch_input_values, batch_input_values_cutoffs in zip(input_values, input_values_cutoffs):
                    batch_input_values_cutoffs = batch_input_values_cutoffs[batch_input_values_cutoffs >= 0]
                    for i in range(batch_input_values_cutoffs.shape[0] - 1):
                        start_idx = batch_input_values_cutoffs[i]
                        end_idx = batch_input_values_cutoffs[i + 1]
                        audio_batch = batch_input_values[..., start_idx:end_idx]
                        codec_outputs = self.codec_model.encode(audio_batch.unsqueeze(0))
                        codebook_ids = codec_outputs.audio_codes.transpose(1, -1)
                        audio_tokens_list.append(codebook_ids[0])

                max_audio_frames = max(el.shape[0] for el in audio_tokens_list)
                batched_audio_token_ids = torch.stack(
                    [nn.functional.pad(el, (0, 0, 0, max_audio_frames - el.shape[0])) for el in audio_tokens_list]
                )
                audio_codes_mask = self.codec_model.get_audio_codes_mask(input_values_mask)
            # =======================================
            audio_token_id = self.config.audio_token_id
            audio_token_mask = input_ids == audio_token_id

            audio_embeds = self.backbone_model.embed_tokens(batched_audio_token_ids)
            inputs_embeds[audio_token_mask] = audio_embeds[audio_codes_mask]

            # same for the audio eos token
            audio_eos_frame_ids = (
                torch.ones((1, 1, self.config.num_codebooks), device=input_ids.device, dtype=torch.long)
                * self.config.codebook_eos_token_id
            )
            audio_eos_embeds = self.backbone_model.embed_tokens(audio_eos_frame_ids).squeeze(1)

            audio_eos_token_mask = input_ids == self.config.audio_eos_token_id
            inputs_embeds[audio_eos_token_mask] = audio_eos_embeds.repeat(audio_eos_token_mask.sum(), 1)

            # if the labels are provided, we need to expand the labels to (batch_size, seq_length, num_codebooks)
            if labels is not None:
                labels_expanded = labels.unsqueeze(-1).repeat(1, 1, self.config.num_codebooks)
                labels_expanded[audio_token_mask] = batched_audio_token_ids[audio_codes_mask]
                labels_expanded[audio_eos_token_mask] = audio_eos_frame_ids
                # mask depth decoder
                depth_decoder_ignore_frames_idxs = (labels == -101).nonzero(as_tuple=True)
                labels_expanded[depth_decoder_ignore_frames_idxs[0], depth_decoder_ignore_frames_idxs[1], 1:] = -100
                labels = labels_expanded

        return {"inputs_embeds": inputs_embeds, "labels": labels}

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        if input_ids is not None and input_ids.ndim == 2 and model_inputs.get("inputs_embeds") is None:
            merged_inputs = self._merge_input_ids_with_input_values(
                input_ids=input_ids,
                input_values=kwargs.get("input_values"),
                input_values_cutoffs=kwargs.get("input_values_cutoffs"),
                labels=kwargs.get("labels"),
            )
            model_inputs.update(
                {"inputs_embeds": merged_inputs["inputs_embeds"], "labels": merged_inputs["labels"], "input_ids": None}
            )

        return model_inputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_values_cutoffs: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CsmOutputWithPast]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks) or (batch_size, sequence_length)`):
            1. (batch_size, sequence_length): corresponds to the input sequence prepared with the processor from the text prompt. Such input
            requires `input_values` to be provided so that audio can be encoded in codebook tokens and then merged with the text tokens.

            2. (batch_size, sequence_length, num_codebooks): codebook tokens generated during the autoregressive decoding. Such input is not meant to be used by end users.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_values_cutoffs (`torch.Tensor` of shape `(batch_size, max_num_audio)`, *optional*):
            Specify the end positions of audio segments within each batch entry, relative to the concatenated audio input.
            If a batch entry has fewer segments than the maximum, it is padded with -1. For example, in a batch of 2 sequences
            where the first contains 2 audio segments of length l1, and the second contains 1 audio segment of length l2,
            the input_values_cutoffs would be: [[l1, 2 * l1], [l2, -1]].
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[config.audio_token_id, -100, -101]`.
            Requires targeted `input_values` to be provided as audio tokens will be infered from it using the `codec_model`.
            - `config.audio_token_id` indicates an audio frames (considering sequence length elements as frames)
            - `-100` will be ignored in the loss computation
            - `-101` indicates the audio frame will be used only for the backbone model (using the first codebook token as labels)

            Such labels can be prepared using `output_labels=True` when calling [`CsmProcessor`].
        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            Kept for compatibility. Does not support another value than:
            1. `0`, which is equivalent to keeping all logits, used in the training regime
            2. `1`, which is equivalent to keeping only the last logit, used in the generation regime

        Example:

        ```python
        >>> import torch
        >>> from transformers import CsmForConditionalGeneration, AutoProcessor
        >>> from datasets import load_dataset, Audio

        >>> model_id = "sesame/csm-1b"
        >>> torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = AutoProcessor.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        >>> # ensure the audio is 24kHz
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=24000))

        >>> conversation = []
        >>> # prepare a conversation with text and corresponding audio
        >>> for text, audio, speaker_id in zip(ds[:4]["text"], ds[:4]["audio"], ds[:4]["speaker_id"]):
        ...     conversation.append(
        ...         {
        ...             "role": f"{speaker_id}",
        ...             "content": [{"type": "text", "text": text}, {"type": "audio", "path": audio["array"]}],
        ...         }
        ...     )

        >>> inputs = processor.apply_chat_template(
        ...     conversation,
        ...     tokenize=True,
        ...     return_dict=True,
        ...     output_labels=True,
        ... ).to(torch_device)

        >>> model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=torch_device)
        >>> output = model(**inputs)
        >>> output.loss.backward()
        ```"""
        if input_ids is not None and input_ids.ndim == 2:
            merged_inputs = self._merge_input_ids_with_input_values(
                input_ids, input_values, input_values_cutoffs, labels
            )
            inputs_embeds = merged_inputs["inputs_embeds"]
            labels = merged_inputs["labels"]
            input_ids = None

        backbone_outputs = self.backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        backbone_hidden_states = backbone_outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        backbone_logits = self.lm_head(backbone_hidden_states[:, slice_indices, :])

        loss = None
        backbone_loss = None
        depth_decoder_loss = None
        depth_decoder_outputs = None
        if labels is not None:
            # select first codebook as labels for the backbone model
            backbone_labels = labels[:, :, 0]
            backbone_loss = self.loss_function(
                logits=backbone_logits, labels=backbone_labels, vocab_size=self.config.vocab_size, **kwargs
            )

            # for the depth decoder, we need to select the frames to train on
            # those are frames where the label is not uniformly `ignore_index` along the codebook dimension
            train_mask = ~(labels[:, :, 1:] == -100).all(dim=-1)
            depth_decoder_input_ids = labels[train_mask][..., : self.config.num_codebooks - 1]
            # add place holder in position 0 that will be replaced by the backbone_last_hidden_state
            depth_decoder_input_ids = nn.functional.pad(depth_decoder_input_ids, (1, 0), value=0)

            train_idxs = train_mask.nonzero(as_tuple=True)
            backbone_last_hidden_states = backbone_hidden_states[train_idxs[0], train_idxs[1] - 1, :]
            depth_decoder_labels = labels[train_mask]

            depth_decoder_outputs = self.depth_decoder(
                input_ids=depth_decoder_input_ids,
                backbone_last_hidden_state=backbone_last_hidden_states,
                use_cache=use_cache,
                return_dict=True,
                labels=depth_decoder_labels,
                **kwargs,
            )

            depth_decoder_loss = depth_decoder_outputs.loss
            loss = backbone_loss + depth_decoder_loss

        return CsmOutputWithPast(
            loss=loss,
            backbone_loss=backbone_loss,
            depth_decoder_loss=depth_decoder_loss,
            logits=backbone_logits,
            past_key_values=backbone_outputs.past_key_values,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
            depth_decoder_logits=depth_decoder_outputs.logits if depth_decoder_outputs is not None else None,
            depth_decoder_past_key_values=depth_decoder_outputs.past_key_values
            if depth_decoder_outputs is not None
            else None,
            depth_decoder_hidden_states=depth_decoder_outputs.hidden_states
            if depth_decoder_outputs is not None
            else None,
            depth_decoder_attentions=depth_decoder_outputs.attentions if depth_decoder_outputs is not None else None,
        )


__all__ = [
    "CsmPreTrainedModel",
    "CsmBackboneModel",
    "CsmDepthDecoderModel",
    "CsmDepthDecoderForCausalLM",
    "CsmForConditionalGeneration",
]
