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
from types import GeneratorType

import torch

from ...generation import GenerationMixin, StoppingCriteria
from ...utils import ModelOutput


class NemotronAsrRNNTDecoderCache:
    """Cache for the RNN-T prediction network (LSTM hidden/cell states + last decoder output)."""

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


class NemotronAsrEncoderExhaustedCriteria(StoppingCriteria):
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

    Handles the transducer machinery: encoder frame tracking, decoder cache preparation, encoder-exhaustion
    stopping, and output-buffer sizing. For RNN-T greedy decoding the encoder frame pointer advances by one
    frame on every blank emission and stays put on every non-blank emission; a ``max_symbols_per_step`` guard
    forces an advance after too many consecutive non-blank emissions at the same frame, mirroring NeMo's greedy
    RNN-T decoding.
    """

    def _get_stopping_criteria(self, *args, **kwargs):
        criteria = super()._get_stopping_criteria(*args, **kwargs)
        criteria.append(NemotronAsrEncoderExhaustedCriteria(self))
        return criteria

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, *args, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, *args, **kwargs)

        logits = outputs.logits[:, -1, :]
        tokens = logits.argmax(dim=-1)
        blank_mask = tokens == self.config.blank_token_id

        # Count consecutive non-blank emissions at the current encoder frame; reset on advance.
        if self._symbols_at_frame is None:
            self._symbols_at_frame = torch.zeros_like(tokens)
        symbols = torch.where(blank_mask, torch.zeros_like(self._symbols_at_frame), self._symbols_at_frame + 1)
        force_advance = symbols >= self.max_symbols_per_step
        self._symbols_at_frame = torch.where(blank_mask | force_advance, torch.zeros_like(symbols), symbols)

        # Advance the encoder frame pointer on blank (or forced) emissions; stay put otherwise.
        advance = (blank_mask | force_advance).long()
        model_kwargs["encoder_frame_idxs"] = model_kwargs["encoder_frame_idxs"] + advance
        self._encoder_finished = model_kwargs["encoder_frame_idxs"] >= model_kwargs["encoder_valid_lengths"]

        if not getattr(self, "_streaming", False):
            return model_kwargs

        # Streaming: pull and encode further chunks whenever the frame pointer has caught up with the buffer.
        generator = model_kwargs.get("input_features_generator")
        while not self._stream_exhausted and bool(
            (model_kwargs["encoder_frame_idxs"] >= model_kwargs["encoder_valid_lengths"]).all()
        ):
            try:
                chunk = next(generator)
            except StopIteration:
                self._stream_exhausted = True
                break
            chunk = chunk.to(device=self.device, dtype=self.dtype)
            self._validate_stream_chunk(chunk, is_first_chunk=False)
            chunk_outputs = self.get_audio_features(
                input_features=chunk,
                past_key_values=model_kwargs["encoder_past_key_values"],
                padding_cache=model_kwargs["padding_cache"],
                num_lookahead_tokens=self._streaming_num_lookahead_tokens,
                use_cache=True,
                output_attention_mask=False,
            )
            pooler = chunk_outputs.pooler_output
            if pooler.shape[1] == 0:
                continue
            encoder_outputs = model_kwargs["encoder_outputs"]
            encoder_outputs.pooler_output = torch.cat([encoder_outputs.pooler_output, pooler], dim=1)
            model_kwargs["encoder_valid_lengths"] = model_kwargs["encoder_valid_lengths"] + pooler.shape[1]

        # Recompute exhaustion now that the buffer may have grown (drives NemotronAsrEncoderExhaustedCriteria).
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
        # When the user hasn't explicitly set max_length/max_new_tokens, size the output buffer. The actual
        # stopping is handled by NemotronAsrEncoderExhaustedCriteria; this just sizes the buffer generously.
        if has_default_max_length and generation_config.max_new_tokens is None:
            if getattr(self, "_streaming", False):
                # Streaming: total audio length is unknown, so the buffer can't be derived from the input.
                generation_config.max_length = int(1e9)
            else:
                # Offline: derive an upper bound from the encoder capacity.
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

    def _required_stream_chunk_frames(self, is_first_chunk: bool) -> int:
        """
        The exact number of mel frames a streaming chunk must carry, given the attention right context.

        For `chunked_limited` cache-aware streaming (NeMo `setup_streaming_params`, with the FastConformer
        subsampling `get_sampling_frames() == [1, subsampling_factor]`):

        - first chunk:      `1 + subsampling_factor * num_lookahead_tokens`
        - subsequent chunk: `subsampling_factor * (num_lookahead_tokens + 1)`

        e.g. for `num_lookahead_tokens == 6` and `subsampling_factor == 8`: 49 then 56 mel frames.
        """
        subsampling_factor = self.config.encoder_config.subsampling_factor
        right = self._streaming_num_lookahead_tokens
        if is_first_chunk:
            return 1 + subsampling_factor * right
        return subsampling_factor * (right + 1)

    def _validate_stream_chunk(self, chunk, is_first_chunk: bool):
        """
        Check a streaming mel chunk has exactly the size required by the attention right context.

        Cache-aware `chunked_limited` streaming consumes fixed-size chunks; a chunk of any other length
        (including a short final chunk) is an error. Pad the final chunk to the required length if needed.
        """
        required = self._required_stream_chunk_frames(is_first_chunk)
        n_frames = chunk.shape[1]
        if n_frames != required:
            which = "first" if is_first_chunk else "subsequent"
            raise ValueError(
                f"Streaming {which} chunk has {n_frames} mel frames but num_lookahead_tokens="
                f"{self._streaming_num_lookahead_tokens} requires exactly {required} "
                f"(first chunk = 1 + subsampling_factor * right, subsequent = subsampling_factor * "
                f"(right + 1)). Pad the final chunk to the required length if needed."
            )

    def _prepare_model_inputs(self, inputs=None, bos_token_id=None, model_kwargs=None):
        from .modeling_nemotron_asr import NemotronAsrEncoderCausalConvPaddingCache, NemotronAsrEncoderModelOutput

        input_features = inputs if inputs is not None else (model_kwargs or {}).get("input_features")

        # Streaming: `input_features` is a generator of mel chunks. Seed the transducer frame buffer by
        # encoding the first chunk. The encoder K/V cache is created implicitly (use_cache=True with no cache
        # passed → a sliding-window DynamicCache from encoder_config.sliding_window), as for voxtral. (A
        # cache_implementation="static" option, with an explicit cache built in `_prepare_cache_for_generation`
        # à la voxtral, can be added later.)
        if isinstance(input_features, GeneratorType):
            model_kwargs = model_kwargs or {}
            generator = input_features
            try:
                first_chunk = next(generator)
            except StopIteration as e:
                raise ValueError("The `input_features` generator did not yield any chunk.") from e
            first_chunk = first_chunk.to(device=self.device, dtype=self.dtype)
            self._validate_stream_chunk(first_chunk, is_first_chunk=True)
            batch_size = first_chunk.shape[0]

            padding_cache = NemotronAsrEncoderCausalConvPaddingCache()
            encoder_outputs = self.get_audio_features(
                input_features=first_chunk,
                padding_cache=padding_cache,
                num_lookahead_tokens=self._streaming_num_lookahead_tokens,
                use_cache=True,
                output_attention_mask=False,
            )

            model_kwargs.pop("input_features", None)
            model_kwargs["input_features_generator"] = generator
            model_kwargs["encoder_past_key_values"] = encoder_outputs.past_key_values
            model_kwargs["padding_cache"] = padding_cache
            model_kwargs["encoder_outputs"] = NemotronAsrEncoderModelOutput(
                pooler_output=encoder_outputs.pooler_output
            )
            model_kwargs["encoder_valid_lengths"] = torch.full(
                (batch_size,), encoder_outputs.pooler_output.shape[1], dtype=torch.long, device=self.device
            )
            model_kwargs["encoder_frame_idxs"] = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            return first_chunk, "input_features", model_kwargs

        # Offline: encode the full mel spectrogram up front.
        inputs, input_name, model_kwargs = super()._prepare_model_inputs(inputs, bos_token_id, model_kwargs)

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

        model_kwargs["encoder_frame_idxs"] = torch.zeros(
            inputs.shape[0],
            device=inputs.device,
            dtype=torch.long,
        )

        return inputs, input_name, model_kwargs

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        model_kwargs["decoder_cache"] = NemotronAsrRNNTDecoderCache(self.config)

    def prepare_inputs_for_generation(self, input_ids, *args, **kwargs):
        from .modeling_nemotron_asr import NemotronAsrEncoderModelOutput

        model_inputs = super().prepare_inputs_for_generation(input_ids, *args, **kwargs)
        encoder_frame_idxs = model_inputs.pop("encoder_frame_idxs").to(
            model_inputs["encoder_outputs"].pooler_output.device
        )

        pooler_output = model_inputs["encoder_outputs"].pooler_output
        batch_size, max_encoder_len = pooler_output.shape[0], pooler_output.shape[1]
        encoder_frame_idxs = encoder_frame_idxs.clamp(max=max_encoder_len - 1)
        model_inputs["encoder_outputs"] = NemotronAsrEncoderModelOutput(
            pooler_output=pooler_output[torch.arange(batch_size), encoder_frame_idxs, None],
        )

        return model_inputs

    def generate(self, inputs=None, generation_config=None, **kwargs):
        input_features = kwargs.get("input_features", inputs)
        self._streaming = isinstance(input_features, GeneratorType)
        if self._streaming:
            self._stream_exhausted = False
            # Resolve the right attention context (lookahead) once; it sets the required mel-chunk sizes
            # and the chunked_limited attention mask used for every chunk of this stream.
            num_lookahead_tokens = kwargs.pop("num_lookahead_tokens", None)
            resolved = self.encoder._resolve_attn_context(num_lookahead_tokens)
            self._streaming_num_lookahead_tokens = resolved[1] if resolved is not None else num_lookahead_tokens
        self._encoder_finished = None
        self._symbols_at_frame = None
        try:
            outputs = super().generate(inputs=inputs, generation_config=generation_config, **kwargs)
        finally:
            for attr in (
                "_streaming",
                "_stream_exhausted",
                "_streaming_num_lookahead_tokens",
                "_encoder_finished",
                "_symbols_at_frame",
            ):
                if hasattr(self, attr):
                    delattr(self, attr)
        sequences = outputs.sequences if isinstance(outputs, ModelOutput) else outputs
        return NemotronAsrGenerateOutput(sequences=sequences)
