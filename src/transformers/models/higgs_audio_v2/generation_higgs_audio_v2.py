# Copyright 2025, The HuggingFace Inc. team. All rights reserved.
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

from ...generation import (
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    GenerationMode,
    GenerationState,
    LogitsProcessorList,
)
from ...generation.logits_process import (
    InfNanRemoveLogitsProcessor,
    LogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from ...utils import ModelOutput, add_start_docstrings, logging


logger = logging.get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class HiggsAudioV2DelayPatternLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for Higgs Audio V2 text-to-speech model to handle codebook delay pattern.

    <Tip warning={true}>

    This logits processor is exclusively compatible with
    [Higgs Audio V2](https://huggingface.co/docs/transformers/main/en/model_doc/higgs_audio_v2)

    </Tip>

    Args:
        delay_pattern (list[int]):
            The delay pattern for the audio bos and eos tokens.
        audio_bos_token_id (int):
            The id of the audio bos token.
        audio_eos_token_id (int):
            The id of the audio eos token.
        audio_stream_bos_id (int):
            The id of the audio stream bos token.
        audio_stream_eos_id (int):
            The id of the audio stream eos token.
        num_codebooks (int):
            The number of codebooks in the audio stream.
        codebook_size (int):
            The size of each codebook in the audio stream.
    """

    def __init__(
        self,
        delay_pattern: list[int],
        audio_bos_token_id: int,
        audio_eos_token_id: int,
        audio_stream_bos_id: int,
        audio_stream_eos_id: int,
        num_codebooks: int,
        codebook_size: int,
    ):
        self.delay_pattern = torch.tensor(delay_pattern)
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.bos_delay_pattern = None
        self.eos_delay_pattern = None
        self.vocab_mask_bos = torch.arange(codebook_size) != audio_stream_bos_id
        self.vocab_mask_eos = torch.arange(codebook_size) != audio_stream_eos_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores.clone().reshape(-1, self.num_codebooks, self.codebook_size)
        batch_size = scores.shape[0]

        # we only look at the n-th last tokens to initialize the bos and eos delay patterns, where n is the delay pattern size
        delay_pattern_size = len(self.delay_pattern)
        input_ids = input_ids[:, -delay_pattern_size:]

        # Initialize bos delay pattern
        if self.bos_delay_pattern is None:
            self.bos_delay_pattern = self.delay_pattern.repeat(batch_size, 1)
            audio_bos_idxs = (input_ids == self.audio_bos_token_id).nonzero()

            if len(audio_bos_idxs) > 0:
                batch_idxs = audio_bos_idxs[:, 0]
                is_first = torch.cat([batch_idxs.new_ones(1, dtype=torch.bool), batch_idxs[1:] != batch_idxs[:-1]])
                min_bos_idxs = audio_bos_idxs[is_first]
                current_after_bos = (delay_pattern_size - min_bos_idxs[:, 1]).unsqueeze(-1)
                unique_batch_idxs = batch_idxs.unique().to(self.bos_delay_pattern.device)
                self.bos_delay_pattern[unique_batch_idxs] = self.bos_delay_pattern[
                    unique_batch_idxs
                ] - current_after_bos.to(self.bos_delay_pattern.device)
            else:
                # there is no audio bos token,
                self.bos_delay_pattern = torch.zeros_like(self.bos_delay_pattern)

        # Initialize eos delay pattern
        if self.eos_delay_pattern is None:
            self.eos_delay_pattern = self.delay_pattern.repeat(batch_size, 1)
            audio_eos_idxs = (input_ids == self.audio_eos_token_id).nonzero()

            if len(audio_eos_idxs) > 0:
                batch_idxs = audio_eos_idxs[:, 0]
                is_first = torch.cat([batch_idxs.new_ones(1, dtype=torch.bool), batch_idxs[1:] != batch_idxs[:-1]])
                min_eos_idxs = audio_eos_idxs[is_first]
                current_after_eos = (delay_pattern_size - min_eos_idxs[:, 1]).unsqueeze(-1)
                unique_batch_idxs = batch_idxs.unique().to(self.eos_delay_pattern.device)
                self.eos_delay_pattern[unique_batch_idxs] = self.eos_delay_pattern[
                    unique_batch_idxs
                ] - current_after_eos.to(self.eos_delay_pattern.device)

        # at each generation step, we decrement the bos delay pattern
        row_mask = self.bos_delay_pattern >= 0
        scores[(row_mask[..., None] & self.vocab_mask_bos).to(scores.device)] = -float("inf")
        self.bos_delay_pattern[row_mask] -= 1

        # when the audio eos token is generated, we decrement the eos delay pattern
        self.eos_delay_pattern[input_ids[:, -1].to(self.eos_delay_pattern.device) == self.audio_eos_token_id] -= 1
        row_mask = self.eos_delay_pattern <= 0
        scores[(row_mask[..., None] & self.vocab_mask_eos).to(scores.device)] = -float("inf")

        return scores.reshape(-1, self.codebook_size)


@dataclass
class HiggsAudioV2GenerationOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of HiggsAudioV2 generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The text sequences: the prompt followed by one placeholder token per generated audio frame
            (`audio_token_id`, `audio_delay_token_id` or `eos_token_id`). The second dimension (sequence_length) is
            either equal to `max_length` or shorter if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the audio head (scores for each codebook token before SoftMax) at each
            generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each
            generated frame), each of shape `(batch_size * config.num_codebooks, config.codebook_size)`: the delay
            pattern logits processor lays the codebooks out along the batch dimension so that the sampling warpers
            apply per codebook.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the audio head (scores for each codebook token before SoftMax) at each
            generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each
            generated frame), each of shape `(batch_size, config.num_codebooks * config.codebook_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
        audio_sequences (`torch.LongTensor` of shape `(batch_size, num_frames, config.num_codebooks)`, *optional*):
            The generated discrete audio codes: the audio prompt frames followed by the generated frames. Finished
            rows are padded with end-of-stream frames (`audio_stream_eos_id` in every codebook).
    """

    audio_sequences: torch.LongTensor | None = None


class HiggsAudioV2GenerationMixin(GenerationMixin):
    """
    HiggsAudioV2 generates one audio frame (`num_codebooks` tokens) per step. The text stream (`sequences`) only
    receives placeholder tokens that drive the delay pattern and the end of generation: the audio token while audio
    streams, the delay token once a codebook emitted `audio_stream_eos_id` (the delay pattern logits processor then
    closes the remaining codebooks one step at a time), and `eos_token_id` once every codebook has ended. The frames
    are accumulated in `model_kwargs["audio_input_ids"]` (with `audio_input_ids_mask`, `False` on all-EOS frames) and
    returned as `audio_sequences`. Finished rows emit end-of-stream frames. Sampling is per codebook, with optional
    repetition-aware resampling (`ras_win_len`, `ras_win_max_num_repeat` on the generation config). Only greedy
    search and sampling are supported.
    """

    _supported_generation_modes = [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]

    # Logits processors that only operate on scores and are safe to apply per-codebook.
    # Other processors (e.g. RepetitionPenaltyLogitsProcessor) use input_ids to index into
    # scores and are incompatible with audio codebook logits.
    _supported_logits_processor_types = (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
        InfNanRemoveLogitsProcessor,
    )

    def _get_logits_processor(self, *args, **kwargs) -> LogitsProcessorList:
        parent_processors = super()._get_logits_processor(*args, **kwargs)

        unsupported = [p for p in parent_processors if not isinstance(p, self._supported_logits_processor_types)]
        if unsupported:
            unsupported_names = [type(p).__name__ for p in unsupported]
            raise ValueError(
                f"HiggsAudioV2 generates audio codebook logits, not text logits. "
                f"The following logits processors are not compatible: {unsupported_names}. "
                f"Only the following processors are supported: "
                f"{[t.__name__ for t in self._supported_logits_processor_types]}."
            )

        delay_pattern_processor = HiggsAudioV2DelayPatternLogitsProcessor(
            delay_pattern=[el + 1 for el in range(self.config.num_codebooks)],
            audio_bos_token_id=self.config.audio_bos_token_id,
            audio_eos_token_id=self.config.audio_delay_token_id,
            audio_stream_bos_id=self.config.audio_stream_bos_id,
            audio_stream_eos_id=self.config.audio_stream_eos_id,
            num_codebooks=self.config.num_codebooks,
            codebook_size=self.config.codebook_size,
        )

        # The delay pattern processor must run first: it reshapes scores from flat
        # (batch_size, num_codebooks * codebook_size) to per-codebook (batch_size * num_codebooks, codebook_size).
        # The sampling warpers (temperature, top_k, top_p) then correctly apply per-codebook.
        # Without this ordering, top_k/top_p would filter across all codebooks combined,
        # zeroing out entire codebooks and producing NaN after softmax.
        logits_processor = LogitsProcessorList()
        logits_processor.append(delay_pattern_processor)
        logits_processor.extend(parent_processors)
        return logits_processor

    def _select_next_tokens(
        self,
        next_token_scores: torch.FloatTensor,
        generation_config: GenerationConfig,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        state: GenerationState,
    ) -> torch.LongTensor:
        # `next_token_scores` is `(batch_size * num_codebooks, codebook_size)` after the delay pattern processor: one
        # token is selected per codebook, then regrouped into `(batch_size, num_codebooks)` frames
        num_codebooks, codebook_size = self.config.num_codebooks, self.config.codebook_size
        next_tokens = super()._select_next_tokens(next_token_scores, generation_config, outputs, model_kwargs, state)
        next_tokens = next_tokens.reshape(-1, num_codebooks)

        # repetition-aware sampling: codebooks whose selected token already appears at least `ras_win_max_num_repeat`
        # times in the last `ras_win_len` frames are resampled from the unprocessed logits (without temperature).
        # Stream BOS / EOS tokens are not counted as repetitions.
        ras_win_len = getattr(generation_config, "ras_win_len", None)
        ras_win_max_num_repeat = getattr(generation_config, "ras_win_max_num_repeat", None)
        audio_input_ids = model_kwargs.get("audio_input_ids")
        if ras_win_len is not None and ras_win_max_num_repeat is not None and audio_input_ids is not None:
            audio_input_ids_window = audio_input_ids[:, -ras_win_len:, :]
            repetition_mask = audio_input_ids_window == next_tokens.unsqueeze(1)
            not_excluded_mask = (audio_input_ids_window != self.config.audio_stream_bos_id) & (
                audio_input_ids_window != self.config.audio_stream_eos_id
            )
            rep_num = (repetition_mask & not_excluded_mask).sum(dim=1)
            replacement_mask = rep_num >= ras_win_max_num_repeat
            next_token_logits = outputs.logits[:, -1].to(dtype=torch.float32, device=next_tokens.device)
            next_token_logits = next_token_logits.reshape(-1, num_codebooks, codebook_size)
            replacement_tokens = (
                next_token_logits[replacement_mask].softmax(dim=-1).multinomial(1, replacement=True).view(-1)
            )
            next_tokens[replacement_mask] = replacement_tokens
        return next_tokens

    def _mask_finished_tokens(
        self,
        next_tokens: torch.LongTensor,
        unfinished_sequences: torch.LongTensor,
        pad_token_id: torch.Tensor | int,
    ) -> torch.LongTensor:
        # finished rows emit end-of-stream frames: the text pad token is not a valid codebook id
        return super()._mask_finished_tokens(next_tokens, unfinished_sequences, self.config.audio_stream_eos_id)

    def _append_next_tokens(self, sequences: torch.LongTensor, next_tokens: torch.LongTensor) -> torch.LongTensor:
        # The text stream gets one placeholder per frame: the audio token while audio streams, the delay token once a
        # codebook ended its stream (or once the previous placeholder was already the delay token, so that the delay
        # pattern processor keeps closing the remaining codebooks), and `eos_token_id` once every codebook has ended.
        is_stream_eos = next_tokens == self.config.audio_stream_eos_id
        next_text_tokens = sequences.new_full((sequences.shape[0],), self.config.audio_token_id)
        previous_is_delay = sequences[:, -1] == self.config.audio_delay_token_id
        next_text_tokens[is_stream_eos.any(dim=-1) | previous_is_delay] = self.config.audio_delay_token_id
        if self.config.eos_token_id is not None:
            next_text_tokens[is_stream_eos.all(dim=-1)] = self.config.eos_token_id
        return super()._append_next_tokens(sequences, next_text_tokens)

    def _update_model_kwargs_with_next_tokens(
        self,
        next_tokens: torch.LongTensor,
        outputs: ModelOutput | None,
        model_kwargs: dict[str, Any],
        state: GenerationState,
    ) -> dict[str, Any]:
        # the frame is stored for the next forward; all-EOS frames are masked out of the audio inputs
        new_frames = next_tokens[:, None, :]
        new_mask = ~(next_tokens == self.config.audio_stream_eos_id).all(dim=-1, keepdim=True)
        audio_input_ids = model_kwargs.get("audio_input_ids")
        audio_input_ids_mask = model_kwargs.get("audio_input_ids_mask")
        model_kwargs["audio_input_ids"] = (
            new_frames if audio_input_ids is None else torch.cat([audio_input_ids, new_frames], dim=1)
        )
        model_kwargs["audio_input_ids_mask"] = (
            new_mask if audio_input_ids_mask is None else torch.cat([audio_input_ids_mask, new_mask], dim=1)
        )
        return model_kwargs

    def _build_generate_output(
        self,
        sequences: torch.LongTensor,
        state: GenerationState,
        generation_config: GenerationConfig,
        model_kwargs: dict[str, Any],
        **kwargs,
    ) -> HiggsAudioV2GenerationOutput | torch.LongTensor:
        # `generate` returns the audio frames, the text placeholders only matter inside the loop
        output = super()._build_generate_output(sequences, state, generation_config, model_kwargs, **kwargs)
        audio_sequences = model_kwargs.get("audio_input_ids")
        if generation_config.return_dict_in_generate:
            return HiggsAudioV2GenerationOutput(**output, audio_sequences=audio_sequences)
        return audio_sequences
