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

from types import GeneratorType

import torch

from ...generation import GenerationMixin
from ...models.parakeet.generation_parakeet import (
    ParakeetRNNTDecoderCache,
    ParakeetRNNTGenerateOutput,
    ParakeetRNNTGenerationMixin,
)


class NemotronAsrRNNTDecoderCache(ParakeetRNNTDecoderCache): ...


class NemotronAsrGenerateOutput(ParakeetRNNTGenerateOutput): ...


class NemotronAsrGenerationMixin(ParakeetRNNTGenerationMixin):
    """Generation mixin for NemotronAsr RNN-T models.

    Inherits the shared transducer machinery from [`ParakeetRNNTGenerationMixin`] (encoder frame tracking,
    decoder cache preparation, encoder-exhaustion stopping, per-step durations and output-buffer sizing) and
    extends it with cache-aware ``chunked_limited`` streaming: ``input_features`` may be a generator of mel
    chunks, which are encoded incrementally and appended to the encoder frame buffer as the decoder consumes it.
    """

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, *args, **kwargs):
        # Parakeet handles the shared RNN-T frame bookkeeping (blank counting, frame advance, per-step
        # durations) and the offline encoder-exhaustion signal. NemotronAsr extends it with streaming.
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, *args, **kwargs)

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

        # Recompute exhaustion now that the buffer may have grown (drives the inherited EncoderExhaustedCriteria).
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
        # stopping is handled by the encoder-exhaustion stopping criteria; this just sizes the buffer generously.
        if has_default_max_length and generation_config.max_new_tokens is None:
            if getattr(self, "_streaming", False):
                # Streaming: total audio length is unknown, so the buffer can't be derived from the input.
                generation_config.max_length = int(1e9)
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

        # Offline: encode the full mel spectrogram up front. Delegate to Parakeet's shared implementation.
        return super()._prepare_model_inputs(inputs, bos_token_id, model_kwargs)

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        model_kwargs["decoder_cache"] = NemotronAsrRNNTDecoderCache(self.config)

    def prepare_inputs_for_generation(self, input_ids, *args, **kwargs):
        from .modeling_nemotron_asr import NemotronAsrEncoderModelOutput

        # Bypass ParakeetRNNTGenerationMixin's `prepare_inputs_for_generation` (it would build a
        # `ParakeetEncoderModelOutput`, which `NemotronAsrForRNNT.forward` does not recognize via isinstance
        # and would mangle into `pooler_output=None`). Go straight to the base GenerationMixin and select the
        # current encoder frame into a `NemotronAsrEncoderModelOutput`.
        model_inputs = GenerationMixin.prepare_inputs_for_generation(self, input_ids, *args, **kwargs)
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
            # and the chunked_limited attention mask used for every chunk of this stream. It must be passed
            # explicitly: it governs both the attention right context in every forward and the exact mel-chunk
            # sizes the encoder consumes, so falling back to the config default could silently mismatch the
            # chunks the processor produced (`processor.set_num_lookahead_tokens(...)`) and corrupt the
            # transcript. Validate it before touching the stream so the missing argument — not a downstream
            # chunk-size mismatch — is what gets reported.
            num_lookahead_tokens = kwargs.pop("num_lookahead_tokens", None)
            if num_lookahead_tokens is None:
                raise ValueError(
                    "Streaming `generate` (when `input_features` is a generator of mel chunks) requires "
                    "`num_lookahead_tokens`: it must be passed explicitly. It must match the right attention context "
                    "used to size the chunks (e.g. `processor.set_num_lookahead_tokens(...)`, then pass the same "
                    "`num_lookahead_tokens=...` here). Supported values: "
                    f"{list(self.config.encoder_config.supported_num_lookahead_tokens)}."
                )
            self._streaming_num_lookahead_tokens = self.encoder._resolve_attn_context(num_lookahead_tokens)[1]
        try:
            # Parakeet's generate() runs the decoding loop and assembles sequences + per-step durations.
            outputs = super().generate(inputs=inputs, generation_config=generation_config, **kwargs)
        finally:
            for attr in ("_streaming", "_stream_exhausted", "_streaming_num_lookahead_tokens"):
                if hasattr(self, attr):
                    delattr(self, attr)

        if isinstance(outputs, ParakeetRNNTGenerateOutput):
            return NemotronAsrGenerateOutput(sequences=outputs.sequences, durations=outputs.durations)
        return NemotronAsrGenerateOutput(sequences=outputs)
