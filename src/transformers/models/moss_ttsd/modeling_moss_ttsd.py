# coding=utf-8
# Copyright 2025 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch MOSS-TTSD model."""

from dataclasses import dataclass
from typing import Optional, Union

from ...cache_utils import Cache
from ...generation import GenerationConfig, GenerationMixin, LogitsProcessorList, StoppingCriteriaList
from ...generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateDecoderOnlyOutput
from ...loss.loss_utils import ForCausalLMLoss
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.qwen3.modeling_qwen3 import Qwen3Model
from ...utils import ModelOutput, auto_docstring, is_torch_available
from .configuration_moss_ttsd import MossTTSDConfig


if is_torch_available():
    import torch
    import torch.nn as nn

_CHECKPOINT_FOR_DOC = "fnlp/MOSS-TTSD-v0.5"


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for MOSS-TTSD outputs, with hidden states and attentions.
    """
)
class MossTTSDOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    loss_all (`tuple(torch.FloatTensor)`, *optional*, returned when multiple loss components are calculated):
        Tuple containing all loss components for detailed analysis.
    logits_all (`tuple(torch.FloatTensor)`, *optional*, returned when multiple logit outputs are generated):
        Tuple containing all logit outputs from different model heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    loss_all: Optional[tuple[torch.FloatTensor, ...]] = None
    logits_all: Optional[tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[tuple[tuple[torch.FloatTensor, ...], ...]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for MOSS-TTSD causal language model (or autoregressive) outputs.
    """
)
class MossTTSDCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
        sequence_length)`.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class MossTTSDGenerationMixin(GenerationMixin):
    """
    Generation mixin for MossTTSD model with multi-channel support.
    """

    def _setup_channel_processors(
        self, generation_config: GenerationConfig, channels: int
    ) -> list[LogitsProcessorList]:
        """Setup logits processors for each channel based on generation config."""
        realprocessor = [LogitsProcessorList() for _ in range(channels)]

        if hasattr(generation_config, "layers"):
            for i, layer_config in enumerate(generation_config.layers):
                if i >= channels:
                    break

                if layer_config.get("repetition_penalty") is not None:
                    realprocessor[i].append(
                        RepetitionPenaltyLogitsProcessor(penalty=layer_config.get("repetition_penalty"))
                    )
                if layer_config.get("temperature") is not None:
                    realprocessor[i].append(TemperatureLogitsWarper(temperature=layer_config.get("temperature")))
                if layer_config.get("top_k") is not None:
                    realprocessor[i].append(TopKLogitsWarper(top_k=layer_config.get("top_k")))
                if layer_config.get("top_p") is not None:
                    realprocessor[i].append(TopPLogitsWarper(top_p=layer_config.get("top_p")))

        return realprocessor

    def _generate_next_tokens_with_scores(
        self,
        logits_all: tuple[torch.Tensor, ...],
        input_ids: torch.LongTensor,
        tf_inputs: torch.LongTensor,
        channels: int,
        realprocessor: list[LogitsProcessorList],
        do_samples: list[bool],
        speech_pad_idx: int,
    ) -> tuple[torch.LongTensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Generate next tokens for all channels with scores and logits."""
        # Get next token logits
        next_token_logits = tuple(logits[:, -1, :].clone().float().to(input_ids.device) for logits in logits_all)

        # Apply channel-specific constraints
        for i, channel_logits in enumerate(next_token_logits):
            if i != 0 and input_ids.shape[1] + 1 > tf_inputs.shape[1] - 7 + i:
                channel_logits[:, speech_pad_idx] = -torch.inf
            if i == 0 and input_ids.shape[1] + 1 <= tf_inputs.shape[1]:
                channel_logits[:, self.config.speech_eos_token] = -torch.inf

        # Process logits
        next_token_scores = tuple(
            realprocessor[i](input_ids[..., i], logits) for i, logits in enumerate(next_token_logits)
        )

        # Sample or select tokens
        next_tokens = []
        for i, channel_score in enumerate(next_token_scores):
            if do_samples[i]:
                channel_ntk = torch.multinomial(nn.functional.softmax(channel_score, dim=-1), num_samples=1).squeeze(1)
            else:
                channel_ntk = torch.argmax(channel_score, dim=-1)
            next_tokens.append(channel_ntk)

        return torch.stack(next_tokens, dim=-1), next_token_scores, next_token_logits

    def _process_multi_channel_tokens(
        self,
        next_tokens: torch.LongTensor,
        needs_additional_steps: torch.LongTensor,
        input_ids: torch.LongTensor,
        tf_inputs: torch.LongTensor,
        base_length: int,
        channels: int,
        eos_token_id: Optional[int],
        speech_pad_idx: int,
        unfinished_sequences: torch.LongTensor,
        has_eos_stopping_criteria: bool,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Process tokens for multi-channel TTS generation."""
        # Additional steps logic
        indices = (~self.is_speech_token(next_tokens[:, 0])) & (needs_additional_steps < 0)
        needs_additional_steps[indices] = channels - 1  # For 8 channels, need 7 steps

        if input_ids.shape[1] + 1 <= tf_inputs.shape[1]:
            i = input_ids.shape[1] + 1 - base_length
            next_tokens[:, i:] = tf_inputs[:, input_ids.shape[1], i:]

        # Replace tokens in additional steps
        mask = (needs_additional_steps > 0) & (needs_additional_steps < 7)
        if mask.any().item():
            next_tokens[mask, 0] = eos_token_id
            for i in range(1, channels):
                mask_i = mask & (needs_additional_steps < channels - i)
                next_tokens[mask_i, i] = speech_pad_idx

        if has_eos_stopping_criteria:
            for i in range(channels):
                pddp = eos_token_id if i == 0 else speech_pad_idx
                next_tokens[:, i] = next_tokens[:, i] * unfinished_sequences + pddp * (1 - unfinished_sequences)

        return next_tokens, needs_additional_steps

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional[BaseStreamer],
        **model_kwargs,
    ) -> Union[GenerateDecoderOnlyOutput, torch.LongTensor]:
        """Sample method for multi-channel TTS generation."""
        # Extract configuration parameters
        speech_pad_idx = getattr(self.config, "speech_pad_token", 1024)
        eos_token_id = generation_config.eos_token_id
        channels = getattr(self.config, "channels", 8)

        # Generation config parameters
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # Initialize output tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # Initialize tracking variables
        batch_size, cur_len, input_channels = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        needs_additional_steps = -1 * torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # Adjust input for generation
        tf_inputs = input_ids.clone()
        input_ids = input_ids[:, : -(channels - 1)]
        cur_len = input_ids.shape[1]
        model_kwargs["attention_mask"] = model_kwargs["attention_mask"][:, : -(channels - 1)]
        base_length = input_ids.shape[1]
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        # Setup logits processors and sampling config
        if hasattr(generation_config, "do_samples") and generation_config.do_samples is not None:
            do_samples = generation_config.do_samples
            realprocessor = self._setup_channel_processors(generation_config, channels)
        else:
            do_samples = [do_sample for _ in range(channels)]
            realprocessor = [logits_processor for _ in range(channels)]
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # Prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
            # Forward pass
            outputs = self(**model_inputs, return_dict=True)
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)

            if synced_gpus and this_peer_finished:
                continue

            # Generate next tokens for all channels
            next_tokens, next_token_scores, next_token_logits = self._generate_next_tokens_with_scores(
                outputs.logits_all, input_ids, tf_inputs, channels, realprocessor, do_samples, speech_pad_idx
            )
            # Process tokens for multi-channel TTS
            next_tokens, needs_additional_steps = self._process_multi_channel_tokens(
                next_tokens,
                needs_additional_steps,
                input_ids,
                tf_inputs,
                base_length,
                channels,
                eos_token_id,
                speech_pad_idx,
                unfinished_sequences,
                has_eos_stopping_criteria,
            )

            input_ids = torch.cat([input_ids, next_tokens[:, None, :]], dim=1)
            if streamer is not None:
                streamer.put(next_tokens[:, 0].cpu())

            # Update unfinished_sequences
            needs_additional_steps = torch.where(
                needs_additional_steps > 0, needs_additional_steps - 1, needs_additional_steps
            )
            stopping = stopping_criteria(input_ids[..., 0], scores) | (needs_additional_steps == 0)
            unfinished_sequences = unfinished_sequences & ~stopping
            unfinished_sequences = unfinished_sequences | (needs_additional_steps > 0)
            this_peer_finished = unfinished_sequences.max() == 0

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            cur_len += 1
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids


class MossTTSDPretrainedModel(PreTrainedModel):
    """Base class for MOSS-TTSD pretrained models."""

    config_class = MossTTSDConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True


class MossTTSDModel(MossTTSDPretrainedModel):
    """MOSS-TTSD model for text-to-speech synthesis."""

    base_model_prefix = "language_model"

    def __init__(self, config: MossTTSDConfig):
        super().__init__(config)
        self.text_pad_idx = config.pad_token_id if config.pad_token_id is not None else 0
        self.speech_pad_idx = config.speech_pad_token if config.speech_pad_token is not None else 0

        self.embedding_list = nn.ModuleList([])
        # For text embedding, only use padding_idx if it's valid
        text_padding_idx = self.text_pad_idx if self.text_pad_idx < config.vocab_size else None
        self.embedding_list.append(nn.Embedding(config.vocab_size, config.hidden_size, text_padding_idx))
        # Channels 1 to channels-1: Speech tokens only
        speech_padding_idx = self.speech_pad_idx if self.speech_pad_idx < config.speech_vocab_size else None
        for _ in range(1, config.channels):
            self.embedding_list.append(nn.Embedding(config.speech_vocab_size, config.hidden_size, speech_padding_idx))

        self.language_model = Qwen3Model(config)
        self.post_init()

    def get_input_embeddings(self):
        """Get the input embeddings for the model."""
        return self.embedding_list[0]

    def set_input_embeddings(self, value: nn.Embedding):
        """Set the input embeddings for the model."""
        self.embedding_list[0] = value

    def _prepare_multi_modal_inputs(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Prepare multi-modal embeddings from input_ids of shape (batch_size, sequence_length, channels).

        For channel 0: text + speech tokens, for channels 1 to channels-1: speech tokens padded with speech_pad_token.
        """
        batch_size, seq_length, channels = input_ids.shape
        if channels != self.config.channels:
            raise ValueError(f"Expected {self.config.channels} channels, got {channels}")

        inputs_embeds = torch.zeros(
            batch_size,
            seq_length,
            self.config.hidden_size,
            device=input_ids.device,
            dtype=self.embedding_list[0].weight.dtype,
        )
        for i in range(channels):
            embed_layer = self.embedding_list[i]
            channel_input = input_ids[..., i]
            inputs_embeds += embed_layer(channel_input)

        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        """Forward pass for MOSS-TTSD model."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self._prepare_multi_modal_inputs(input_ids)

        return self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


class MossTTSDForCausalLM(MossTTSDPretrainedModel, MossTTSDGenerationMixin):
    """MOSS-TTSD model for causal language modeling with multi-channel support."""

    _tied_weights_keys = {}
    _tp_plan = {"lm_heads.*": "colwise_rep"}
    _pp_plan = {"lm_heads.*": (["hidden_states"], ["logits"])}

    def __init__(self, config: MossTTSDConfig):
        super().__init__(config)
        self.model = MossTTSDModel(config)
        self.channels = config.channels
        self.weights = [1 for _ in range(self.channels)]
        self._tied_weights_keys = {
            f"lm_heads.{i}.weight": f"model.embedding_list.{i}.weight" for i in range(self.channels)
        }
        self.vocab_size = config.vocab_size
        self.lm_heads = nn.ModuleList([])
        self.lm_heads.append(nn.Linear(config.hidden_size, config.vocab_size, bias=False))
        for _ in range(1, config.channels):
            self.lm_heads.append(nn.Linear(config.hidden_size, config.speech_vocab_size, bias=False))
        self.post_init()

    def get_input_embeddings(self):
        """Get the input embeddings for the model."""
        return self.model.embedding_list[0]

    def can_generate(self):
        """Check if the model can generate."""
        return True

    def is_speech_token(self, tokens: torch.Tensor) -> torch.Tensor:
        """Check if tokens are speech tokens."""
        return (tokens >= self.config.speech_token_range[0]) & (tokens < self.config.speech_token_range[1])

    def set_input_embeddings(self, value: nn.Embedding):
        """Set the input embeddings for the model."""
        self.model.embedding_list[0] = value

    def get_output_embeddings(self):
        """Get the output embeddings for the model."""
        return self.lm_heads[0]

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """Set the output embeddings for the model."""
        self.lm_heads[0] = new_embeddings

    def set_decoder(self, decoder: MossTTSDModel):
        """Set the decoder for the model."""
        self.model = decoder

    def get_decoder(self):
        """Get the decoder for the model."""
        return self.model

    def set_weights(self, weights: list[float]):
        """Set the weights for different channels."""
        self.weights = weights

    def _compute_loss(
        self, hidden_states: torch.Tensor, labels: torch.LongTensor, skip_logits: bool, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[tuple[torch.Tensor, ...]]]:
        """Compute loss for all channels."""
        device = hidden_states.device
        loss_all = torch.empty(self.channels, device=device)
        logits_list = []

        for i in range(self.config.channels):
            vocab_size = self.config.vocab_size if i == 0 else self.config.speech_vocab_size
            logits = self.lm_heads[i](hidden_states)
            loss_all[i] = ForCausalLMLoss(logits, labels[..., i], vocab_size)
            if not skip_logits:
                logits_list.append(logits)

        logits_all = tuple(logits_list) if logits_list else None

        # Compute weighted total loss
        total_weight = sum(self.weights)
        normalized_weights = [w / total_weight for w in self.weights]
        total_loss = sum(w * loss for w, loss in zip(normalized_weights, loss_all))

        return total_loss, loss_all, logits_all

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        skip_logits: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, MossTTSDOutputWithPast]:
        """Forward pass for MOSS-TTSD causal language model."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        skip_logits = skip_logits if skip_logits is not None else (self.training and labels is not None)
        if skip_logits and labels is None:
            skip_logits = False

        # Decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits_all = None
        loss_all = None
        total_loss = None

        if labels is not None:
            total_loss, loss_all, logits_all = self._compute_loss(hidden_states, labels, skip_logits, **kwargs)
        else:
            logits_all = [lm_head(hidden_states) for lm_head in self.lm_heads]
            total_loss = None
            loss_all = None

        if not return_dict:
            output = (logits_all,) + outputs[1:]
            return (
                (
                    total_loss,
                    loss_all,
                )
                + output
                if total_loss is not None
                else output
            )

        return MossTTSDOutputWithPast(
            loss=total_loss,
            logits=logits_all[0] if logits_all is not None else None,
            loss_all=loss_all,
            logits_all=logits_all,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["MossTTSDModel", "MossTTSDForCausalLM"]
