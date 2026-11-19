# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationConfig, GenerationMixin, GenerationMode
from ...generation.logits_process import LogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateDecoderOnlyOutput, GenerateNonBeamOutput, StopCheck
from ...utils import logging


logger = logging.get_logger(__name__)


@dataclass
class Lfm3AudioGenerateOutput(GenerateDecoderOnlyOutput):
    """
    [`~generation.GenerateDecoderOnlyOutput`] plus the generated speech frames.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Prompt + generated tokens: text ids, one `audio_out_token_id` per speech frame, one `audio_token_id` per
            encoded audio token when transcribing a stream.
        audio_sequences (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
            Generated speech codes, end-of-audio frames included, padded with `config.codebook_size`.
        audio_sequences_mask (`torch.BoolTensor` of shape `(batch_size, num_frames)`):
            Frames to decode (no padding, no end-of-audio frame). Pass both to [`Lfm3AudioProcessor.decode`].
        padding_cache ([`Lfm3AudioEncoderCausalConvPaddingCache`], *optional*, returned when transcribing a stream):
            Subsampling state of the speech encoder.
        encoder_cache (`Cache`, *optional*, returned when transcribing a stream):
            Cache of the speech encoder.
    """

    audio_sequences: torch.LongTensor | None = None
    audio_sequences_mask: torch.BoolTensor | None = None
    padding_cache: Any | None = None
    encoder_cache: Cache | None = None


class Lfm3AudioWindowedRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`RepetitionPenaltyLogitsProcessor`] over the last `window` codes of one codebook. `input_ids` rows: past codes of
    one (sequence, codebook) pair, oldest first, padded with negative values.

    Args:
        penalty (`float`):
            Repetition penalty (> 0).
        window (`int`):
            Number of past frames covered.
    """

    def __init__(self, penalty: float, window: int):
        if not (isinstance(penalty, float) and penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")
        if not (isinstance(window, int) and window > 0):
            raise ValueError(f"`window` has to be a strictly positive integer, but is {window}")
        self.penalty = penalty
        self.window = window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        past = input_ids[:, -self.window :].to(scores.device)

        # padding points at a spare column, dropped after the penalty
        num_codes = scores.shape[-1]
        past = torch.where(past < 0, num_codes, past)
        padded = torch.cat((scores, scores.new_zeros((scores.shape[0], 1))), dim=-1)
        score = torch.gather(padded, 1, past)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        return padded.scatter(1, past, score)[:, :num_codes]


class Lfm3AudioOutputPhases:
    """
    Output phase per row + slots left in its interleaved block. Speech frame in `SPEECH`, `INTERLEAVED_SPEECH` and
    `INTERLEAVED_TAIL`, text token otherwise.

    - `TEXT`: `<|audio_start|>` -> `SPEECH`, `<|mixed_start|>` -> `INTERLEAVED_TEXT`.
    - `SPEECH`: end-of-audio frame -> `TEXT`.
    - `INTERLEAVED_TEXT` / `INTERLEAVED_SPEECH`: alternate `interleaved_n_text` tokens and `interleaved_n_audio` frames.
      `<|text_end|>` in a text slot -> `INTERLEAVED_TAIL`, end-of-audio frame -> `TEXT`.
    - `INTERLEAVED_TAIL`: frames until an end-of-audio frame -> `TEXT`.
    """

    TEXT, SPEECH, INTERLEAVED_TEXT, INTERLEAVED_SPEECH, INTERLEAVED_TAIL = range(5)

    def __init__(self, config, input_ids: torch.LongTensor, audio_input_ids: torch.Tensor | None):
        self.audio_start_token_id = config.audio_start_token_id
        self.mixed_start_token_id = config.mixed_start_token_id
        self.text_end_token_id = config.text_end_token_id
        self.audio_out_token_id = config.audio_out_token_id
        self.audio_eos_code = config.audio_eos_code
        self.interleaved_n_text = config.interleaved_n_text
        self.interleaved_n_audio = config.interleaved_n_audio

        self.phase = self._prompt_phases(input_ids, audio_input_ids)
        self.slots_left = torch.full_like(self.phase, self.interleaved_n_text)

    def _prompt_phases(self, input_ids: torch.LongTensor, audio_input_ids: torch.Tensor | None) -> torch.LongTensor:
        """Phase per row, from the prompt's last token."""
        if input_ids.shape[1] == 0:
            return input_ids.new_full((input_ids.shape[0],), self.TEXT)

        last = input_ids[:, -1]
        speaking = last == self.audio_start_token_id
        if audio_input_ids is not None and audio_input_ids.shape[0]:
            # prompt ending on an end-of-audio frame: back to text
            counts = (input_ids == self.audio_out_token_id).sum(-1)
            last_frame = (counts.cumsum(0) - 1).clamp_min(0)
            terminal = (counts > 0) & (audio_input_ids[last_frame] == self.audio_eos_code).any(-1)
            speaking |= (last == self.audio_out_token_id) & ~terminal
        else:
            speaking |= last == self.audio_out_token_id

        phases = torch.where(last == self.mixed_start_token_id, self.INTERLEAVED_TEXT, self.TEXT)
        return torch.where(speaking, self.SPEECH, phases)

    def any_speech(self) -> bool:
        """Whether a row speaks or will (an interleaved block always does)."""
        return bool((self.phase != self.TEXT).any())

    def speaking_rows(self) -> torch.BoolTensor:
        return (
            (self.phase == self.SPEECH)
            | (self.phase == self.INTERLEAVED_SPEECH)
            | (self.phase == self.INTERLEAVED_TAIL)
        )

    def update(
        self, next_tokens: torch.LongTensor, active: torch.BoolTensor, terminal: torch.BoolTensor
    ) -> torch.BoolTensor:
        """Advances rows past `next_tokens`; returns rows whose `<|text_end|>` closed an interleaved text block."""
        phase = self.phase
        # markers open a block from plain text only
        in_text = active & (phase == self.TEXT)
        opens_speech = in_text & (next_tokens == self.audio_start_token_id)
        opens_block = in_text & (next_tokens == self.mixed_start_token_id)

        # `<|text_end|>`: text closed, speech goes on until end-of-audio
        closes_text = active & (phase == self.INTERLEAVED_TEXT) & (next_tokens == self.text_end_token_id)

        counting = active & ((phase == self.INTERLEAVED_TEXT) | (phase == self.INTERLEAVED_SPEECH)) & ~closes_text
        slots_left = torch.where(counting, self.slots_left - 1, self.slots_left)
        to_speech = counting & (slots_left == 0) & (phase == self.INTERLEAVED_TEXT)
        to_text = counting & (slots_left == 0) & (phase == self.INTERLEAVED_SPEECH)
        slots_left = torch.where(to_speech, self.interleaved_n_audio, slots_left)
        self.slots_left = torch.where(to_text | opens_block, self.interleaved_n_text, slots_left)

        phase = torch.where(to_speech, self.INTERLEAVED_SPEECH, phase)
        phase = torch.where(to_text | opens_block, self.INTERLEAVED_TEXT, phase)
        phase = torch.where(closes_text, self.INTERLEAVED_TAIL, phase)
        phase = torch.where(opens_speech, self.SPEECH, phase)
        self.phase = torch.where(terminal, self.TEXT, phase)

        return closes_text


class Lfm3AudioChunkStreamer(BaseStreamer):
    """Relays per-chunk `generate` calls to `streamer` as one call."""

    def __init__(self, streamer: BaseStreamer, prompt_length: int):
        self.streamer = streamer
        self.prompt_length = prompt_length
        self.prompt_pending = True

    def new_chunk(self):
        self.prompt_pending = True

    def put(self, value):
        if self.prompt_pending:
            # prompt = stream so far + chunk placeholders: only the first one is new
            self.prompt_pending = False
            if self.prompt_length:
                self.streamer.put(value[..., : self.prompt_length])
                self.prompt_length = 0
            return

        self.streamer.put(value)

    def end(self):
        # wrapped streamer ends once, in `_generate_stream`
        pass


class Lfm3AudioGenerationMixin(GenerationMixin):
    """
    [`GenerationMixin`] with a per-row output phase, depth-decoder speech frames (`depth_decoder_*` generation config
    fields), and stream transcription (`input_features` as an iterator of chunks).
    """

    _supported_generation_modes = [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]

    def _prepare_generation_config(
        self, generation_config: GenerationConfig | None, **kwargs: Any
    ) -> tuple[GenerationConfig, dict]:
        custom_kwargs = {
            key: kwargs.pop(key)
            for key in list(kwargs)
            if key.startswith("depth_decoder_")
            or key in ("max_new_tokens_per_audio_chunk", "max_new_tokens_last_audio_chunk")
        }

        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        for key, value in custom_kwargs.items():
            setattr(generation_config, key, value)

        return generation_config, model_kwargs

    def _validate_model_kwargs(self, model_kwargs: dict[str, Any]):
        # consumed by `_sample`, not a model input
        model_kwargs.pop("audio_streamer", None)
        super()._validate_model_kwargs(model_kwargs)

    def generate(
        self, inputs: torch.Tensor | None = None, generation_config: GenerationConfig | None = None, **kwargs
    ):
        if isinstance(kwargs.get("input_features"), Iterator):
            return self._generate_stream(inputs, generation_config, **kwargs)
        return super().generate(inputs, generation_config, **kwargs)

    def _generate_stream(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        streamer: BaseStreamer | None = None,
        **kwargs,
    ) -> Lfm3AudioGenerateOutput | torch.LongTensor:
        # circular import
        from .modeling_lfm3_audio import Lfm3AudioEncoderCausalConvPaddingCache

        config = self.config
        if self.model.audio_model is None:
            raise ValueError("Transcribing a stream needs the speech encoder, which `audio_config=None` dropped")
        if "max_new_tokens" in kwargs or "max_length" in kwargs:
            raise ValueError(
                "Transcribing a stream budgets each chunk: pass `max_new_tokens_per_audio_chunk` and "
                "`max_new_tokens_last_audio_chunk` instead of `max_new_tokens` or `max_length`"
            )

        chunks = kwargs.pop("input_features")
        input_ids = inputs if inputs is not None else kwargs.pop("input_ids")
        if input_ids.shape[0] != 1:
            raise ValueError(f"Transcribing a stream takes one prompt, got a batch of {input_ids.shape[0]}")

        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        cache = kwargs.pop("cache", None)
        padding_cache = kwargs.pop("padding_cache", None)
        if padding_cache is None:
            padding_cache = Lfm3AudioEncoderCausalConvPaddingCache()
        encoder_cache = kwargs.pop("encoder_cache", None)
        if encoder_cache is None:
            encoder_cache = DynamicCache(config=config.audio_config)

        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        model_kwargs.pop("audio_streamer", None)
        chunk_streamer = Lfm3AudioChunkStreamer(streamer, input_ids.shape[1]) if streamer is not None else None
        text_end_token_id, subsampling_factor = config.text_end_token_id, config.audio_config.subsampling_factor
        outputs = {"scores": (), "logits": (), "attentions": (), "hidden_states": ()}

        for chunk in chunks:
            is_last = bool(chunk.get("is_last_audio_chunk", False))
            features, features_mask = chunk["input_features"], chunk.get("input_features_mask")
            num_frames = features.shape[1] if features_mask is None else int(features_mask.sum())
            num_tokens = -(num_frames // -subsampling_factor)

            input_ids = torch.cat((input_ids, input_ids.new_full((1, num_tokens), config.audio_token_id)), dim=-1)
            attention_mask = torch.cat((attention_mask, attention_mask.new_ones((1, num_tokens))), dim=-1)

            if chunk_streamer is not None:
                chunk_streamer.new_chunk()

            chunk_outputs = super().generate(
                input_ids,
                self._audio_chunk_generation_config(generation_config, is_last),
                streamer=chunk_streamer,
                attention_mask=attention_mask,
                input_features=features.to(self.device, self.dtype),
                input_features_mask=None if features_mask is None else features_mask.to(self.device),
                cache=cache,
                padding_cache=padding_cache,
                encoder_cache=encoder_cache,
                **model_kwargs,
            )

            input_ids, cache = chunk_outputs.sequences, chunk_outputs.cache
            for key, value in outputs.items():
                outputs[key] = value + (getattr(chunk_outputs, key) or ())

            if is_last:
                break

            if input_ids[0, -1] != text_end_token_id:
                # budget exhausted: close the block anyway
                input_ids = torch.cat((input_ids, input_ids.new_full((1, 1), text_end_token_id)), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, attention_mask.new_ones((1, input_ids.shape[1] - attention_mask.shape[1]))), dim=-1
            )

        if streamer is not None:
            streamer.end()

        if not generation_config.return_dict_in_generate:
            return input_ids

        audio_sequences, audio_sequences_mask = self._gather_audio_sequences([], 1, input_ids.device)
        return Lfm3AudioGenerateOutput(
            sequences=input_ids,
            **{key: value or None for key, value in outputs.items()},
            cache=cache,
            audio_sequences=audio_sequences,
            audio_sequences_mask=audio_sequences_mask,
            padding_cache=padding_cache,
            encoder_cache=encoder_cache,
        )

    def _audio_chunk_generation_config(self, generation_config: GenerationConfig, is_last: bool) -> GenerationConfig:
        """Per-chunk generation config: token budget, end tokens, suppressed tokens."""
        config = self.config
        text_end_token_id = config.text_end_token_id

        eos_token_ids = generation_config.eos_token_id
        eos_token_ids = [eos_token_ids] if isinstance(eos_token_ids, int) else list(eos_token_ids or [])
        # `<|text_end|>` closes every block; other end tokens only on the last chunk
        eos_token_ids = [token_id for token_id in eos_token_ids if token_id != text_end_token_id]

        # no speech output when transcribing
        suppress_tokens = [
            *(generation_config.suppress_tokens or []),
            config.audio_start_token_id,
            config.mixed_start_token_id,
        ]

        if is_last:
            max_new_tokens = getattr(generation_config, "max_new_tokens_last_audio_chunk", None) or 96
            eos_token_ids = [text_end_token_id, *eos_token_ids]
        else:
            max_new_tokens = getattr(generation_config, "max_new_tokens_per_audio_chunk", None) or 48
            # can not end before the last chunk
            suppress_tokens = [*suppress_tokens, config.mixed_end_token_id, *eos_token_ids]
            eos_token_ids = [text_end_token_id]

        chunk_config = copy.deepcopy(generation_config)
        chunk_config.update(
            max_new_tokens=max_new_tokens,
            max_length=None,
            eos_token_id=eos_token_ids,
            suppress_tokens=suppress_tokens,
            return_dict_in_generate=True,
        )
        return chunk_config

    def _get_depth_decoder_generation_config(self, generation_config: GenerationConfig) -> GenerationConfig:
        """Depth decoder sampling settings, from the `depth_decoder_*` fields."""
        prefix = "depth_decoder_"
        settings = {
            key[len(prefix) :]: value
            for key, value in vars(generation_config).items()
            if key.startswith(prefix)
            and key not in ("depth_decoder_repetition_penalty", "depth_decoder_repetition_window")
        }

        if not settings.get("do_sample", False):
            # greedy: sampling settings do not apply
            settings = {"do_sample": False}

        return GenerationConfig(**settings)

    def _get_depth_decoder_logits_processor(
        self, generation_config: GenerationConfig, depth_decoder_config: GenerationConfig
    ) -> LogitsProcessorList:
        """Depth decoder processors: windowed repetition penalty, then the standard ones."""
        processors = self._get_logits_processor(depth_decoder_config, device=self.device)

        penalty = getattr(generation_config, "depth_decoder_repetition_penalty", 1.0)
        window = getattr(generation_config, "depth_decoder_repetition_window", 0)
        if penalty != 1.0 and window:
            processors.insert(0, Lfm3AudioWindowedRepetitionPenaltyLogitsProcessor(float(penalty), int(window)))
        return processors

    def _sample_audio_frames(
        self,
        hidden_states: torch.Tensor,
        history: torch.LongTensor | None,
        processors: LogitsProcessorList,
        do_sample: bool,
    ) -> torch.LongTensor:
        r"""
        One speech frame per row, codebooks revealed over `config.maskgit_schedule` passes. `history`:
        `(batch_size, num_past_frames, num_codebooks)`, `-1` padded, for the repetition penalty.
        """
        config = self.config
        batch_size, num_codebooks, codebook_size = hidden_states.shape[0], config.num_codebooks, config.codebook_size
        # all codebooks start masked (`codebook_size`)
        codes = hidden_states.new_full((batch_size, num_codebooks), codebook_size, dtype=torch.long)

        # MaskGIT: random reveal order, even when greedy
        order = torch.stack([torch.randperm(num_codebooks, device=hidden_states.device) for _ in range(batch_size)])
        for pick in order.split(config.maskgit_schedule, dim=1):
            # depth decoder may sit on another device (`device_map`)
            logits = self.depth_decoder(input_ids=codes, backbone_last_hidden_state=hidden_states).logits
            logits = logits.to(hidden_states.device)
            num_picked = pick.shape[1]
            scores = logits.gather(1, pick[..., None].expand(-1, -1, codebook_size)).reshape(-1, codebook_size).float()

            past = None
            if history is not None:
                # one row of past codes per (sequence, codebook)
                past = history.gather(2, pick[:, None, :].expand(-1, history.shape[1], -1)).transpose(1, 2)
                past = past.reshape(-1, history.shape[1])
            scores = processors(past, scores)

            if do_sample:
                sampled = torch.multinomial(nn.functional.softmax(scores, dim=-1), num_samples=1).squeeze(1)
            else:
                sampled = scores.argmax(-1)

            codes.scatter_(1, pick, sampled.view(batch_size, num_picked).to(codes.device))

        return codes

    def _select_next_tokens(
        self,
        next_token_scores: torch.FloatTensor,
        last_hidden_state: torch.FloatTensor,
        active: torch.BoolTensor,
        speaking: torch.BoolTensor,
        do_sample: bool,
        history: torch.LongTensor | None,
        depth_decoder_processors: LogitsProcessorList,
        depth_decoder_do_sample: bool,
    ) -> tuple[torch.LongTensor, torch.LongTensor | None, torch.BoolTensor]:
        r"""
        Next token per row (text or `audio_out_token_id`), the frames `(batch_size, num_codebooks)` (`-1` on rows not
        speaking, `None` if none speaks), and the rows ending speech.
        """
        config = self.config
        # speaking rows draw no text token: no randomness consumed
        next_tokens = torch.zeros_like(active, dtype=torch.long)
        text_indices = (active & ~speaking).nonzero(as_tuple=True)[0]
        if text_indices.numel():
            if do_sample:
                probs = nn.functional.softmax(next_token_scores[text_indices], dim=-1)
                next_tokens[text_indices] = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens[text_indices] = torch.argmax(next_token_scores[text_indices], dim=-1)

        terminal = torch.zeros_like(speaking)
        audio_indices = speaking.nonzero(as_tuple=True)[0]
        if not audio_indices.numel():
            return next_tokens, None, terminal

        new_frames = self._sample_audio_frames(
            last_hidden_state[audio_indices],
            None if history is None else history[audio_indices],
            depth_decoder_processors,
            depth_decoder_do_sample,
        )

        # any end-of-audio code -> whole end-of-audio frame
        frame_terminal = (new_frames == config.audio_eos_code).any(-1)
        new_frames = torch.where(frame_terminal[:, None], config.audio_eos_code, new_frames)
        terminal[audio_indices] = frame_terminal

        frames = next_tokens.new_full((next_tokens.shape[0], config.num_codebooks), -1)
        frames[audio_indices] = new_frames
        next_tokens = torch.where(speaking, config.audio_out_token_id, next_tokens)
        return next_tokens, frames, terminal

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: BaseStreamer | None = None,
        **model_kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor:
        """
        Overrides [`~generation.GenerationMixin._sample`]: text token or depth-decoder speech frame per row. Changes
        marked "Lfm3Audio specific".
        """
        # *************** Lfm3Audio specific ***************
        config = self.config
        audio_streamer = model_kwargs.pop("audio_streamer", None)
        phases = Lfm3AudioOutputPhases(config, input_ids, model_kwargs.get("audio_input_ids"))

        if phases.any_speech() and self.depth_decoder is None:
            raise ValueError("The prompt asks for speech, which `depth_decoder_config=None` dropped")

        # frames are fed back one per step: needs the cache
        if not model_kwargs.get("use_cache", True) and (
            phases.any_speech() or model_kwargs.get("audio_input_ids") is not None
        ):
            raise ValueError("Speech frames are fed back through the cache: generating speech needs `use_cache=True`")

        cache = model_kwargs.get("cache")
        if cache is not None and cache.get_seq_length():
            # cached speech is already consumed: keep speech inputs only for placeholders in the new part
            new_ids = input_ids[:, cache.get_seq_length() :]
            if not (new_ids == config.audio_token_id).any():
                model_kwargs.pop("input_features", None)
                model_kwargs.pop("input_features_mask", None)
            if not (new_ids == config.audio_out_token_id).any():
                model_kwargs.pop("audio_input_ids", None)

        depth_decoder_config = self._get_depth_decoder_generation_config(generation_config)
        depth_decoder_processors = self._get_depth_decoder_logits_processor(generation_config, depth_decoder_config)

        # past frames per row, for the repetition penalty
        window = getattr(generation_config, "depth_decoder_repetition_window", 0)
        history = input_ids.new_full((input_ids.shape[0], window, config.num_codebooks), -1) if window else None
        frames = []
        # ==================================================

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        if pad_token_id is not None:
            pad_token_id = pad_token_id.to(input_ids.device)

        model_forward = (
            self.get_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )

        prefill_consumed = False
        outputs = self._prefill(
            input_ids,
            generation_config,
            model_kwargs,
            is_first_iteration=not generation_config.is_assistant,
        )
        # *************** Lfm3Audio specific ***************
        # no deferred stop checks: every step syncs anyway, and phases / frames can not be undone
        stop_check = StopCheck(streamer)
        # ==================================================

        with self._optimize_model_for_decode():
            while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
                if prefill_consumed:
                    next_sequence_length = 1 if model_kwargs["use_cache"] else None
                    model_inputs = self.prepare_inputs_for_generation(
                        input_ids, next_sequence_length=next_sequence_length, **model_kwargs
                    )
                    outputs = model_forward(**model_inputs, return_dict=True)
                prefill_consumed = True
                model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
                # *************** Lfm3Audio specific ***************
                model_kwargs["audio_input_ids"] = None
                # ==================================================
                if synced_gpus and this_peer_finished:
                    continue

                next_token_logits = outputs.logits[:, -1].to(copy=True, dtype=torch.float32, device=input_ids.device)

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (outputs.attentions,)
                    if output_hidden_states:
                        decoder_hidden_states += (outputs.hidden_states,)

                # *************** Lfm3Audio specific ***************
                active = unfinished_sequences.bool()
                speaking = active & phases.speaking_rows()
                next_tokens, step_frames, terminal = self._select_next_tokens(
                    next_token_scores,
                    outputs.last_hidden_state[:, -1],
                    active=active,
                    speaking=speaking,
                    do_sample=do_sample,
                    history=history,
                    depth_decoder_processors=depth_decoder_processors,
                    depth_decoder_do_sample=depth_decoder_config.do_sample,
                )

                if step_frames is not None:
                    model_kwargs["audio_input_ids"] = step_frames[speaking]
                    frames.append(step_frames)
                    if audio_streamer is not None:
                        audio_streamer.put(step_frames.cpu())
                    if history is not None:
                        # end-of-audio frames stay out of the window
                        shifted = torch.cat((history[:, 1:], step_frames[:, None]), dim=1)
                        history = torch.where((speaking & ~terminal)[:, None, None], shifted, history)

                # interleaved: `<|text_end|>` starts the speech tail, the turn goes on
                closes_text = phases.update(next_tokens, active, terminal)
                # ==================================================

                # finished sentences should have their next token be a padding token
                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

                # *************** Lfm3Audio specific ***************
                is_done = stopping_criteria(input_ids, scores)
                if closes_text.any():
                    # length / time criteria still apply
                    length_criteria = StoppingCriteriaList(
                        [criteria for criteria in stopping_criteria if not hasattr(criteria, "eos_token_id")]
                    )
                    is_done = torch.where(closes_text, length_criteria(input_ids, scores), is_done)
                unfinished_sequences = unfinished_sequences & ~is_done
                # ==================================================
                this_peer_finished = stop_check(unfinished_sequences, next_tokens, input_ids.shape[1])

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs

        stop_check.finish()
        if streamer is not None:
            streamer.end()
        # *************** Lfm3Audio specific ***************
        if audio_streamer is not None:
            audio_streamer.end()

        if not return_dict_in_generate:
            if frames:
                logger.warning_once(
                    "Speech frames were generated but `return_dict_in_generate=False` returns the token ids only: "
                    "pass `return_dict_in_generate=True` to get `audio_sequences`."
                )
            return input_ids

        audio_sequences, audio_sequences_mask = self._gather_audio_sequences(frames, batch_size, input_ids.device)
        return Lfm3AudioGenerateOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            cache=model_kwargs.get("cache"),
            audio_sequences=audio_sequences,
            audio_sequences_mask=audio_sequences_mask,
        )
        # ==================================================

    def _gather_audio_sequences(self, frames: list, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Frames packed left per row, padded with `codebook_size`, + mask of the frames to decode."""
        config = self.config
        if not frames:
            audio_sequences = torch.full((batch_size, config.num_codebooks, 0), config.codebook_size, device=device)
            return audio_sequences, torch.zeros((batch_size, 0), dtype=torch.bool, device=device)

        stacked = torch.stack(frames, dim=-1)
        valid = stacked[:, 0] != -1
        order = valid.int().argsort(dim=-1, descending=True, stable=True)
        audio_sequences = stacked.gather(2, order[:, None, :].expand(-1, config.num_codebooks, -1))

        num_frames = valid.sum(-1)
        audio_sequences = audio_sequences[:, :, : int(num_frames.max())]
        is_frame = torch.arange(audio_sequences.shape[-1], device=device) < num_frames[:, None]
        audio_sequences = torch.where(is_frame[:, None], audio_sequences, config.codebook_size)
        audio_sequences_mask = is_frame & (audio_sequences != config.audio_eos_code).all(1)
        return audio_sequences, audio_sequences_mask
