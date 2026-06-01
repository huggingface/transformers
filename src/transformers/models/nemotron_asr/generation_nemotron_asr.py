# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from ...generation import GenerationMixin, StoppingCriteria
from ...utils import ModelOutput


class NemotronAsrDecoderCache:
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


class EncoderExhaustedCriteria(StoppingCriteria):
    """Stops generation when all batch elements have walked past their encoder output length."""

    def __init__(self, model):
        self.model = model

    def __call__(self, input_ids, scores, **kwargs):
        if self.model._encoder_finished is None:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        return self.model._encoder_finished


@dataclass
class NemotronAsrGenerateOutput(ModelOutput):
    """
    Outputs of NemotronAsr RNN-T generation.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Generated token sequences (including blank tokens).
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Encoder attention weights per layer.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Encoder hidden states per layer.
    """

    sequences: torch.LongTensor
    attentions: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = None


class NemotronAsrGenerationMixin(GenerationMixin):
    """Generation mixin for NemotronAsr RNN-T models.

    Emits one token per step with simple advance rules: blank → advance the encoder frame;
    non-blank → emit and stay on the same frame, with a ``max_symbols_per_step`` guard that
    forces an advance after too many non-blank emissions at the same frame.
    """

    def _get_stopping_criteria(self, *args, **kwargs):
        criteria = super()._get_stopping_criteria(*args, **kwargs)
        criteria.append(EncoderExhaustedCriteria(self))
        return criteria

    def _update_model_kwargs_for_generation(self, outputs, *args, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, *args, **kwargs)

        logits = outputs.logits[:, -1, :]
        tokens = logits.argmax(dim=-1)
        blank_mask = tokens == self.config.blank_token_id

        # Count consecutive non-blank emissions at the current encoder frame.
        symbols = model_kwargs["symbols_at_frame"]
        symbols = torch.where(blank_mask, torch.zeros_like(symbols), symbols + 1)
        force_advance = symbols >= self.max_symbols_per_step
        # Reset the counter for elements that will advance this step.
        symbols = torch.where(blank_mask | force_advance, torch.zeros_like(symbols), symbols)
        model_kwargs["symbols_at_frame"] = symbols

        advance = (blank_mask | force_advance).long()
        model_kwargs["encoder_frame_idxs"] = model_kwargs["encoder_frame_idxs"] + advance

        self._encoder_finished = model_kwargs["encoder_frame_idxs"] >= model_kwargs["encoder_valid_lengths"]
        return model_kwargs

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        has_default_min_length,
        model_input_name,
        input_ids_length,
        inputs_tensor,
    ):
        if has_default_max_length and generation_config.max_new_tokens is None:
            encoder_seq_len = self.encoder._get_subsampling_output_length(
                torch.tensor([inputs_tensor.shape[1]], device=inputs_tensor.device)
            ).item()
            generation_config.max_length = self.max_symbols_per_step * encoder_seq_len
            has_default_max_length = False
        return super()._prepare_generated_length(
            generation_config,
            has_default_max_length,
            has_default_min_length,
            model_input_name,
            input_ids_length,
            inputs_tensor,
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
            batch_size = encoder_outputs.last_hidden_state.shape[0]
            encoder_valid_lengths = torch.full(
                (batch_size,),
                encoder_outputs.last_hidden_state.shape[1],
                dtype=torch.long,
                device=encoder_outputs.last_hidden_state.device,
            )
        model_kwargs["encoder_valid_lengths"] = encoder_valid_lengths

        model_kwargs["encoder_frame_idxs"] = torch.zeros(inputs.shape[0], device=inputs.device, dtype=torch.long)
        model_kwargs["symbols_at_frame"] = torch.zeros(inputs.shape[0], device=inputs.device, dtype=torch.long)

        return inputs, input_name, model_kwargs

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        from .modeling_nemotron_asr import NemotronAsrDecoderCache

        model_kwargs["decoder_cache"] = NemotronAsrDecoderCache(self.config)

    def prepare_inputs_for_generation(self, input_ids, *args, **kwargs):
        from .modeling_nemotron_asr import NemotronAsrEncoderModelOutput

        model_inputs = super().prepare_inputs_for_generation(input_ids, *args, **kwargs)
        encoder_frame_idxs = model_inputs.pop("encoder_frame_idxs").to(
            model_inputs["encoder_outputs"].pooler_output.device
        )
        # symbols_at_frame is internal state; pop so it isn't passed to model.forward.
        model_inputs.pop("symbols_at_frame", None)

        pooler_output = model_inputs["encoder_outputs"].pooler_output
        batch_size, max_encoder_len = pooler_output.shape[0], pooler_output.shape[1]
        encoder_frame_idxs = encoder_frame_idxs.clamp(max=max_encoder_len - 1)
        model_inputs["encoder_outputs"] = NemotronAsrEncoderModelOutput(
            pooler_output=pooler_output[torch.arange(batch_size), encoder_frame_idxs, None],
        )

        return model_inputs

    def generate(self, inputs=None, generation_config=None, **kwargs):
        self._encoder_finished = None
        outputs = super().generate(inputs=inputs, generation_config=generation_config, **kwargs)
        del self._encoder_finished
        sequences = outputs.sequences if isinstance(outputs, ModelOutput) else outputs
        return NemotronAsrGenerateOutput(sequences=sequences)
