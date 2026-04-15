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

from dataclasses import dataclass

import torch

from ...generation import GenerationMixin, GenerationMode, StoppingCriteria
from ...utils import ModelOutput


@dataclass
class ParakeetTDTGenerateOutput(ModelOutput):
    """
    Outputs of Parakeet TDT generation.

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
    """Stops generation when all batch elements have walked past their encoder output length."""

    def __init__(self, model):
        self.model = model

    def __call__(self, input_ids, scores, **kwargs):
        if self.model._encoder_finished is None:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        return self.model._encoder_finished


class ParakeetTDTGenerationMixin(GenerationMixin):
    """Generation mixin for Parakeet TDT models.

    Handles transducer-specific generation logic: encoder frame tracking,
    duration accumulation, and encoder-exhaustion stopping.
    """
    def _get_stopping_criteria(self, *args, **kwargs):
        criteria = super()._get_stopping_criteria(*args, **kwargs)
        criteria.append(EncoderExhaustedCriteria(self))
        return criteria

    def _update_model_kwargs_for_generation(self, outputs, *args, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, *args, **kwargs)

        # Advance encoder frame pointer by the predicted duration
        logits = outputs.logits[:, -1, :]
        tokens = logits[:, : self.config.vocab_size].argmax(dim=-1)
        durations = logits[:, self.config.vocab_size :].argmax(dim=-1)
    
        # Only force forward progress (duration >= 1) for blank predictions;
        blank_mask = tokens == self.config.blank_token_id
        durations = torch.where(blank_mask & (durations == 0), torch.ones_like(durations), durations)
        model_kwargs["encoder_frame_idxs"] = model_kwargs["encoder_frame_idxs"] + durations
        self._step_durations.append(durations)

        # Track which batch elements have exhausted their encoder frames.
        self._encoder_finished = model_kwargs["encoder_frame_idxs"] >= model_kwargs["encoder_valid_lengths"]

        return model_kwargs

    def _prepare_generated_length(
        self, generation_config, has_default_max_length, has_default_min_length,
        model_input_name, input_ids_length, inputs_tensor,
    ):
        # When the user hasn't explicitly set max_length/max_new_tokens, derive an upper
        # bound from the encoder capacity. The actual stopping is handled by the
        # encoder-exhaustion stopping criteria; this just sizes the output buffer.
        if has_default_max_length and generation_config.max_new_tokens is None:
            encoder_seq_len = self.encoder._get_subsampling_output_length(
                torch.tensor([inputs_tensor.shape[1]], device=inputs_tensor.device)
            ).item()
            generation_config.max_length = self.config.max_symbols_per_step * encoder_seq_len
            has_default_max_length = False  # prevent super() from overwriting
        return super()._prepare_generated_length(
            generation_config, has_default_max_length, has_default_min_length,
            model_input_name, input_ids_length, inputs_tensor,
        )

    def _prepare_model_inputs(self, *args, **kwargs):
        inputs, input_name, model_kwargs = super()._prepare_model_inputs(*args, **kwargs)

        encoder_outputs = self.get_audio_features(
            input_features=inputs,
            attention_mask=model_kwargs.get("attention_mask", None),
            output_attention_mask=True,
        )
        model_kwargs["encoder_outputs"] = encoder_outputs

        if encoder_outputs.attention_mask is not None:
            encoder_valid_lengths = encoder_outputs.attention_mask.sum(-1)
        else:
            batch_size = encoder_outputs.shape[0]
            encoder_valid_lengths = torch.full(
                (batch_size,), encoder_outputs.last_hidden_state.shape[1], dtype=torch.long, device=encoder_outputs.device
            )
        model_kwargs["encoder_valid_lengths"] = encoder_valid_lengths

        model_kwargs["encoder_frame_idxs"] = torch.zeros(
            inputs.shape[0],
            device=inputs.device,
            dtype=torch.long,
        )

        return inputs, input_name, model_kwargs

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        from .modeling_parakeet import ParakeetTDTDecoderCache

        model_kwargs["decoder_cache"] = ParakeetTDTDecoderCache()

    def prepare_inputs_for_generation(self, input_ids, *args, **kwargs):
        from .modeling_parakeet import ParakeetEncoderModelOutput

        model_inputs = super().prepare_inputs_for_generation(input_ids, *args, **kwargs)
        encoder_frame_idxs = model_inputs.pop("encoder_frame_idxs").to(model_inputs["encoder_outputs"].pooler_output.device)

        pooler_output = model_inputs["encoder_outputs"].pooler_output
        batch_size, max_encoder_len = pooler_output.shape[0], pooler_output.shape[1]
        encoder_frame_idxs = encoder_frame_idxs.clamp(max=max_encoder_len - 1)
        model_inputs["encoder_outputs"] = ParakeetEncoderModelOutput(
            pooler_output=pooler_output[torch.arange(batch_size), encoder_frame_idxs, None],
        )

        return model_inputs

    def generate(self, inputs=None, generation_config=None, **kwargs):
        # TODO @eustlb: this is temporary — we're going to modularize generate to allow doing this cleanly.
        self._step_durations = []
        self._encoder_finished = None

        outputs = super().generate(inputs=inputs, generation_config=generation_config, **kwargs)
        durations = torch.stack(self._step_durations, dim=1)  # (batch, steps)
        # Prepend a zero duration for the decoder_start_token_id that super().generate() prepends to sequences
        durations = torch.cat([torch.zeros(durations.shape[0], 1, dtype=durations.dtype, device=durations.device), durations], dim=1)
        del self._step_durations, self._encoder_finished

        return ParakeetTDTGenerateOutput(
            sequences=outputs.sequences if isinstance(outputs, ModelOutput) else outputs,
            durations=durations,
        )
