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

from types import GeneratorType
from typing import Any

import torch

from ...generation import GenerationConfig, GenerationMode, GenerationState
from ...models.parakeet.generation_parakeet import (
    ParakeetRNNTDecoderCache,
    ParakeetRNNTGenerateOutput,
    ParakeetRNNTGenerationMixin,
)
from ...utils import ModelOutput


class NemotronAsrStreamingRNNTDecoderCache(ParakeetRNNTDecoderCache): ...


class NemotronAsrStreamingGenerateOutput(ParakeetRNNTGenerateOutput): ...


class NemotronAsrStreamingGenerationMixin(ParakeetRNNTGenerationMixin):
    """Generation mixin for NemotronAsrStreaming RNN-T models.

    Inherits the shared transducer machinery from [`ParakeetRNNTGenerationMixin`] (encoder frame tracking,
    decoder cache preparation, encoder-exhaustion stopping, per-step durations and output-buffer sizing) and
    extends it with cache-aware ``chunked_limited`` streaming. Streaming is requested by passing `input_features=` as
    a generator of mel chunks together with `num_lookahead_tokens=`: the chunks are encoded incrementally (threading
    the encoder attention and convolution caches) and appended to the encoder frame buffer once every row consumed
    the frames it had, so the loop only stops when the stream is exhausted. The flag lives on the prepared generation
    config (`generation_config.streaming`, an attribute set on the per-call copy of the config, not a
    `GenerationConfig` field), the generator and `num_lookahead_tokens` in `model_kwargs`, and whether the stream is
    exhausted in the generation state; nothing is stored on the model.
    """

    # The streaming conv cache is passed explicitly to the chunk encoder calls
    _non_encoder_kwarg_prefixes = ParakeetRNNTGenerationMixin._non_encoder_kwarg_prefixes + ("padding_cache",)

    def _prepare_generation_config(
        self, generation_config: GenerationConfig | None, **kwargs: Any
    ) -> tuple[GenerationConfig, dict[str, Any]]:
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        # Set before anything touches the generator: `_prepare_generated_length` sees no `model_kwargs`
        generation_config.streaming = isinstance(model_kwargs.get("input_features"), GeneratorType)
        if generation_config.streaming and model_kwargs.get("num_lookahead_tokens") is None:
            raise ValueError(
                "Streaming `generate` (when `input_features` is a generator of mel chunks) requires "
                "`num_lookahead_tokens`: it must be passed explicitly. It must match the right attention context "
                "used to size the chunks (e.g. `processor.set_num_lookahead_tokens(...)`, then pass the same "
                "`num_lookahead_tokens=...` here)."
            )
        return generation_config, model_kwargs

    def _update_model_kwargs_with_next_tokens(
        self,
        next_tokens: torch.LongTensor,
        outputs: ModelOutput | None,
        model_kwargs: dict[str, Any],
        state: GenerationState,
    ) -> dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_with_next_tokens(next_tokens, outputs, model_kwargs, state)

        generator = model_kwargs.get("input_features_generator")
        if generator is None or state.extras.get("stream_exhausted", False):
            return model_kwargs

        # Runs after the pointer advance (`super()`) and before the stopping criteria of this step: once every row
        # consumed its encoder frames, encode the next mel chunk and append it, so `EncoderExhaustedCriteria` only
        # fires when the stream is exhausted.
        if bool((model_kwargs["encoder_frame_idxs"] >= model_kwargs["encoder_valid_lengths"]).all()):
            try:
                chunk = next(generator)
            except StopIteration:
                state.extras["stream_exhausted"] = True
            else:
                chunk = chunk.to(device=self.device, dtype=self.dtype)
                self._validate_stream_chunk(chunk, model_kwargs["num_lookahead_tokens"], is_first_chunk=False)
                chunk_outputs = self.get_audio_features(
                    input_features=chunk,
                    past_key_values=model_kwargs["encoder_past_key_values"],
                    padding_cache=model_kwargs["padding_cache"],
                    use_cache=True,
                    output_attention_mask=False,
                    **self._encoder_kwargs(model_kwargs),  # carries `num_lookahead_tokens`
                )
                pooler = chunk_outputs.pooler_output
                encoder_outputs = model_kwargs["encoder_outputs"]
                encoder_outputs.pooler_output = torch.cat(
                    [encoder_outputs.pooler_output, pooler.to(encoder_outputs.pooler_output.device)], dim=1
                )
                model_kwargs["encoder_valid_lengths"] = model_kwargs["encoder_valid_lengths"] + pooler.shape[1]
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
        # Streaming: the total audio length is unknown, so the buffer cannot be derived from the input; the
        # encoder-exhaustion criterion stops the loop.
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.streaming:
            generation_config.max_length = int(1e9)
            has_default_max_length = False  # prevent the parents from overwriting
        return super()._prepare_generated_length(
            generation_config,
            has_default_max_length,
            has_default_min_length,
            model_input_name,
            input_ids_length,
            inputs_tensor,
        )

    def _required_stream_chunk_frames(self, num_lookahead_tokens: int, is_first_chunk: bool) -> int:
        """
        The exact number of mel frames a streaming chunk must carry, given the attention right context.

        For `chunked_limited` cache-aware streaming (NeMo `setup_streaming_params`, with the FastConformer
        subsampling `get_sampling_frames() == [1, subsampling_factor]`):

        - first chunk:      `1 + subsampling_factor * num_lookahead_tokens`
        - subsequent chunk: `subsampling_factor * (num_lookahead_tokens + 1)`

        e.g. for `num_lookahead_tokens == 6` and `subsampling_factor == 8`: 49 then 56 mel frames.
        """
        subsampling_factor = self.config.encoder_config.subsampling_factor
        if is_first_chunk:
            return 1 + subsampling_factor * num_lookahead_tokens
        return subsampling_factor * (num_lookahead_tokens + 1)

    def _validate_stream_chunk(self, chunk: torch.Tensor, num_lookahead_tokens: int, is_first_chunk: bool) -> None:
        """
        Check a streaming mel chunk has exactly the size required by the attention right context.

        Cache-aware `chunked_limited` streaming consumes fixed-size chunks; a chunk of any other length
        (including a short final chunk) is an error. Pad the final chunk to the required length if needed.
        """
        required = self._required_stream_chunk_frames(num_lookahead_tokens, is_first_chunk)
        n_frames = chunk.shape[1]
        if n_frames != required:
            which = "first" if is_first_chunk else "subsequent"
            raise ValueError(
                f"Streaming {which} chunk has {n_frames} mel frames but num_lookahead_tokens="
                f"{num_lookahead_tokens} requires exactly {required} "
                f"(first chunk = 1 + subsampling_factor * right, subsequent = subsampling_factor * "
                f"(right + 1)). Pad the final chunk to the required length if needed."
            )

    def _prepare_model_inputs(
        self,
        inputs: torch.Tensor | None = None,
        bos_token_id: torch.Tensor | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, str | None, dict[str, Any]]:
        if isinstance(inputs, GeneratorType):
            raise ValueError(
                "Pass a stream of mel chunks as the keyword argument `input_features=` (with "
                "`num_lookahead_tokens=`), not positionally."
            )
        model_kwargs = model_kwargs or {}
        input_features = model_kwargs.get("input_features")
        if isinstance(input_features, GeneratorType):
            generator = input_features
            try:
                first_chunk = next(generator)
            except StopIteration as e:
                raise ValueError("The `input_features` generator did not yield any chunk.") from e
            first_chunk = first_chunk.to(device=self.device, dtype=self.dtype)
            self._validate_stream_chunk(first_chunk, model_kwargs["num_lookahead_tokens"], is_first_chunk=True)

            model_kwargs.pop("input_features", None)
            model_kwargs["input_features_generator"] = generator
            return first_chunk, "input_features", model_kwargs

        # Offline: encode the full mel spectrogram up front. Delegate to Parakeet's shared implementation.
        return super()._prepare_model_inputs(inputs, bos_token_id, model_kwargs)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs: dict[str, Any],
        model_input_name: str | None,
        generation_config: GenerationConfig,
    ) -> dict[str, Any]:
        # Only reached in streaming: offline, `_prepare_model_inputs` already set `encoder_outputs`, so `generate`
        # skips this step
        if not generation_config.streaming:
            return super()._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # Encode the first chunk, opening the encoder attention and convolution caches the next chunks thread
        first_chunk = inputs_tensor
        batch_size = first_chunk.shape[0]
        encoder_outputs = self.get_audio_features(
            input_features=first_chunk,
            use_cache=True,
            output_attention_mask=False,
            **self._encoder_kwargs(model_kwargs),  # carries `num_lookahead_tokens`
        )

        model_kwargs["encoder_past_key_values"] = encoder_outputs.past_key_values
        model_kwargs["padding_cache"] = encoder_outputs.padding_cache
        # the encoder frame buffer; only `pooler_output` is read and it is grown chunk by chunk
        model_kwargs["encoder_outputs"] = type(encoder_outputs)(pooler_output=encoder_outputs.pooler_output)
        model_kwargs["encoder_valid_lengths"] = torch.full(
            (batch_size,), encoder_outputs.pooler_output.shape[1], dtype=torch.long, device=self.device
        )
        model_kwargs["encoder_frame_idxs"] = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        return model_kwargs

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: dict[str, Any],
        generation_mode: GenerationMode,
        batch_size: int,
        max_cache_length: int,
        max_cache_length_attr: str = "_previous_max_cache_length",
    ) -> None:
        model_kwargs["decoder_cache"] = NemotronAsrStreamingRNNTDecoderCache(self.config)

    def _build_generate_output(
        self,
        sequences: torch.LongTensor,
        state: GenerationState,
        generation_config: GenerationConfig,
        model_kwargs: dict[str, Any],
        **kwargs,
    ) -> NemotronAsrStreamingGenerateOutput:
        output = super()._build_generate_output(sequences, state, generation_config, model_kwargs, **kwargs)
        return NemotronAsrStreamingGenerateOutput(sequences=output.sequences, durations=output.durations)
