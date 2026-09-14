# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch

from ...generation import GenerationConfig, GenerationMixin, GenerationMode, GenerationState, StoppingCriteria
from ...generation.stopping_criteria import StoppingCriteriaList
from ...utils import ModelOutput


class ParakeetRNNTDecoderCache:
    def __init__(self, config):
        self.config = config
        self.cache: torch.Tensor | None = None
        self.hidden_state: torch.Tensor | None = None
        self.cell_state: torch.Tensor | None = None
        self.is_initialized: bool = False

    def lazy_initialization(self, hidden_states):
        self.cache = torch.zeros(
            hidden_states.shape[0],
            1,
            self.config.decoder_hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        self.hidden_state = torch.zeros(
            self.config.num_decoder_layers,
            hidden_states.shape[0],
            self.config.decoder_hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        self.cell_state = torch.zeros(
            self.config.num_decoder_layers,
            hidden_states.shape[0],
            self.config.decoder_hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        torch._dynamo.mark_static_address(self.cache)
        torch._dynamo.mark_static_address(self.hidden_state)
        torch._dynamo.mark_static_address(self.cell_state)

        self.is_initialized = True

    def update(
        self,
        decoder_output,
        hidden_state,
        cell_state,
        mask=None,
    ):
        if not self.is_initialized:
            self.lazy_initialization(decoder_output)

        if mask is None:
            self.hidden_state.copy_(hidden_state)
            self.cell_state.copy_(cell_state)
            self.cache.copy_(decoder_output)
        else:
            # Mask to update specific batch elements
            mask = mask.to(decoder_output.device)
            batch_size = decoder_output.shape[0]
            mask_h = mask.view(1, batch_size, 1)
            mask_d = mask.view(batch_size, 1, 1)
            self.cache = torch.where(mask_d, decoder_output, self.cache)
            self.hidden_state = torch.where(mask_h, hidden_state, self.hidden_state)
            self.cell_state = torch.where(mask_h, cell_state, self.cell_state)


# BC: see #46331
class ParakeetTDTDecoderCache(ParakeetRNNTDecoderCache): ...


@dataclass
class ParakeetRNNTGenerateOutput(ModelOutput):
    """
    Outputs of Parakeet transducer (RNN-T / TDT) generation.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Generated token sequences (including blank tokens).
        durations (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-step durations in frames. Combined with `sequences`, this is sufficient
            to reconstruct full timestamp information (frame indices are the cumulative sum
            of durations).
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Encoder attention weights per layer.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Encoder hidden states per layer.
    """

    sequences: torch.LongTensor
    durations: torch.LongTensor | None = None
    attentions: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = None


class EncoderExhaustedCriteria(StoppingCriteria):
    """Stops a row once its encoder frame pointer walked past its encoder output length (both in `model_kwargs`)."""

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor | None,
        *,
        model_kwargs: dict[str, Any],
        **kwargs,
    ) -> torch.BoolTensor:
        return model_kwargs["encoder_frame_idxs"] >= model_kwargs["encoder_valid_lengths"]


class ParakeetRNNTGenerationMixin(GenerationMixin):
    """Generation mixin for Parakeet RNN-T models, and the base for all Parakeet transducer generation.

    Handles the transducer machinery shared by RNN-T and TDT: encoder frame tracking, decoder cache preparation,
    encoder-exhaustion stopping, and output-buffer sizing. Each step feeds the decoder the encoder frame its pointer
    (`model_kwargs["encoder_frame_idxs"]`) points at; once the token is selected,
    `_update_model_kwargs_with_next_tokens` decides how far the pointer moves. For RNN-T greedy decoding it advances by
    one frame on every blank emission and stays put on every non-blank emission; a `max_symbols_per_step` guard forces
    an advance after too many consecutive non-blank emissions at the same frame, mirroring NeMo's greedy RNN-T
    decoding. The duration-aware [`ParakeetTDTGenerationMixin`] extends this by advancing the pointer by a predicted
    duration instead. Rows stop through [`EncoderExhaustedCriteria`] once their pointer walked past their encoder
    output length.

    The per-step frame advances (the "durations") and the symbols-per-frame counter live in the generation state
    (`state.extras`), the frame pointers in `model_kwargs`; nothing is stored on the model. `generate` always returns a
    [`ParakeetRNNTGenerateOutput`] with `sequences` and `durations`, whatever `return_dict_in_generate`; a streamer, if
    given, still receives the tokens.
    """

    # `model_kwargs` entries that are not encoder inputs (see `_encoder_kwargs`). All matched by prefix: the main
    # input (also `input_features_generator` in streaming), the masks, the decoder-side caches, decoder/cross-attention
    # inputs, and the `encoder_*` entries the loop maintains (`encoder_outputs`, `encoder_valid_lengths`,
    # `encoder_frame_idxs`).
    _non_encoder_kwarg_prefixes = (
        "input_features",
        "attention_mask",
        "output_attention_mask",
        "decoder_",
        "cross_attn",
        "use_cache",
        "past_key_values",
        "cache_params",
        "encoder_",
    )

    def _get_stopping_criteria(self, *args, **kwargs) -> StoppingCriteriaList:
        criteria = super()._get_stopping_criteria(*args, **kwargs)
        criteria.append(EncoderExhaustedCriteria())
        return criteria

    def _update_model_kwargs_with_next_tokens(
        self,
        next_tokens: torch.LongTensor,
        outputs: ModelOutput | None,
        model_kwargs: dict[str, Any],
        state: GenerationState,
    ) -> dict[str, Any]:
        blank_mask = next_tokens == self.config.blank_token_id

        # Count consecutive non-blank emissions at the current encoder frame; reset on advance.
        symbols_at_frame = state.extras.get("symbols_at_frame")
        if symbols_at_frame is None:
            symbols_at_frame = torch.zeros_like(next_tokens)
        symbols = torch.where(blank_mask, torch.zeros_like(symbols_at_frame), symbols_at_frame + 1)
        force_advance = symbols >= self.max_symbols_per_step
        state.extras["symbols_at_frame"] = torch.where(blank_mask | force_advance, torch.zeros_like(symbols), symbols)

        # Advance the encoder frame pointer on blank (or forced) emissions; stay put otherwise.
        return self._advance_encoder_frames((blank_mask | force_advance).long(), model_kwargs, state)

    def _advance_encoder_frames(
        self, durations: torch.LongTensor, model_kwargs: dict[str, Any], state: GenerationState
    ) -> dict[str, Any]:
        """
        Moves each row's encoder frame pointer forward by `durations` frames and records them: cumulatively summed,
        the per-step durations give the encoder frame of each emitted token (enough to reconstruct timestamps).
        """
        model_kwargs["encoder_frame_idxs"] = model_kwargs["encoder_frame_idxs"] + durations
        state.extras.setdefault("durations", []).append(durations)
        return model_kwargs

    def _prepare_generated_length(
        self,
        generation_config: GenerationConfig,
        has_default_max_length: bool,
        has_default_min_length: bool,
        model_input_name: str,
        input_ids_length: int,
        inputs_tensor: torch.Tensor,
    ) -> GenerationConfig:
        # When the user hasn't explicitly set max_length/max_new_tokens, derive an upper
        # bound from the encoder capacity. The actual stopping is handled by the
        # encoder-exhaustion stopping criteria; this just sizes the output buffer.
        if has_default_max_length and generation_config.max_new_tokens is None:
            encoder_seq_len = self.encoder._get_subsampling_output_length(
                torch.tensor([inputs_tensor.shape[1]], device=inputs_tensor.device)
            ).item()
            generation_config.max_length = self.max_symbols_per_step * encoder_seq_len
            has_default_max_length = False  # prevent super() from overwriting
        return super()._prepare_generated_length(
            generation_config,
            has_default_max_length,
            has_default_min_length,
            model_input_name,
            input_ids_length,
            inputs_tensor,
        )

    def _encoder_kwargs(self, model_kwargs: dict[str, Any]) -> dict[str, Any]:
        """The encoder inputs among `model_kwargs`, for any encoder call during generation (`get_audio_features`)."""
        return {
            key: value for key, value in model_kwargs.items() if not key.startswith(self._non_encoder_kwarg_prefixes)
        }

    def _prepare_model_inputs(self, *args, **kwargs) -> tuple[torch.Tensor, str | None, dict[str, torch.Tensor]]:
        inputs, input_name, model_kwargs = super()._prepare_model_inputs(*args, **kwargs)

        encoder_outputs = self.get_audio_features(
            input_features=inputs,
            attention_mask=model_kwargs.get("attention_mask", None),
            output_attention_mask=True,
            **self._encoder_kwargs(model_kwargs),
        )
        model_kwargs["encoder_outputs"] = encoder_outputs

        if encoder_outputs.attention_mask is not None:
            encoder_valid_lengths = encoder_outputs.attention_mask.sum(-1)
        else:
            batch_size = encoder_outputs.last_hidden_state.shape[0]
            encoder_valid_lengths = torch.full(
                (batch_size,),
                encoder_outputs.last_hidden_state.shape[1],
                dtype=torch.long,
                device=encoder_outputs.last_hidden_state.device,
            )
        model_kwargs["encoder_valid_lengths"] = encoder_valid_lengths

        model_kwargs["encoder_frame_idxs"] = torch.zeros(
            inputs.shape[0],
            device=inputs.device,
            dtype=torch.long,
        )

        return inputs, input_name, model_kwargs

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: dict[str, Any],
        generation_mode: GenerationMode,
        batch_size: int,
        max_cache_length: int,
        max_cache_length_attr: str = "_previous_max_cache_length",
    ) -> None:
        model_kwargs["decoder_cache"] = ParakeetRNNTDecoderCache(self.config)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, *args, **kwargs) -> dict[str, Any]:
        model_inputs = super().prepare_inputs_for_generation(input_ids, *args, **kwargs)
        encoder_frame_idxs = model_inputs.pop("encoder_frame_idxs").to(
            model_inputs["encoder_outputs"].pooler_output.device
        )

        pooler_output = model_inputs["encoder_outputs"].pooler_output
        batch_size, max_encoder_len = pooler_output.shape[0], pooler_output.shape[1]
        encoder_frame_idxs = encoder_frame_idxs.clamp(max=max_encoder_len - 1)
        # Feed the decoder the frame each row points at, as the encoder output class the model expects
        model_inputs["encoder_outputs"] = type(model_inputs["encoder_outputs"])(
            pooler_output=pooler_output[torch.arange(batch_size), encoder_frame_idxs, None],
        )

        return model_inputs

    def _build_generate_output(
        self,
        sequences: torch.LongTensor,
        state: GenerationState,
        generation_config: GenerationConfig,
        model_kwargs: dict[str, Any],
        **kwargs,
    ) -> ParakeetRNNTGenerateOutput:
        step_durations = state.extras.get("durations")
        if step_durations:
            durations = torch.stack(step_durations, dim=1)  # (batch, steps)
        else:
            durations = sequences.new_zeros((sequences.shape[0], 0))
        # The decoder start token that opens `sequences` has no duration
        durations = torch.cat([torch.zeros_like(durations[:, :1]), durations], dim=1)
        # Durations are always returned, whatever `return_dict_in_generate` (see the class docstring)
        return ParakeetRNNTGenerateOutput(sequences=sequences, durations=durations)


class ParakeetTDTGenerationMixin(ParakeetRNNTGenerationMixin):
    """Generation mixin for Parakeet TDT models.

    Extends [`ParakeetRNNTGenerationMixin`] with duration-aware decoding: the joint network predicts tokens and
    durations side by side (`vocab_size` token logits followed by one logit per duration). Tokens are selected from
    the vocabulary logits alone, and instead of advancing the encoder frame pointer by one on each blank emission, the
    pointer advances by the predicted duration (forced to at least one frame on blank emissions, so that decoding
    always progresses). The shared setup (encoder frame tracking, decoder cache, stopping criteria, output buffer
    sizing) is inherited unchanged.
    """

    def _get_next_token_logits(
        self, outputs: ModelOutput, model_kwargs: dict[str, Any], device: torch.device
    ) -> torch.FloatTensor:
        # The joint network predicts tokens and durations side by side; tokens are selected from the vocabulary part
        return outputs.logits[:, -1, : self.config.vocab_size].to(copy=True, dtype=torch.float32, device=device)

    def _update_model_kwargs_with_next_tokens(
        self,
        next_tokens: torch.LongTensor,
        outputs: ModelOutput | None,
        model_kwargs: dict[str, Any],
        state: GenerationState,
    ) -> dict[str, Any]:
        # Advance the encoder frame pointer by the predicted duration (`outputs` is set: greedy only)
        logits = outputs.logits[:, -1, :]
        durations = logits[:, self.config.vocab_size :].argmax(dim=-1)

        # Only force forward progress (duration >= 1) for blank predictions
        blank_mask = next_tokens.to(logits.device) == self.config.blank_token_id
        durations = torch.where(blank_mask & (durations == 0), torch.ones_like(durations), durations)
        return self._advance_encoder_frames(durations, model_kwargs, state)
