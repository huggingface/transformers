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
"""Text and speech generation for LFM3-Audio: the generation mixin's sampling loop with a per-row text/speech mode."""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationConfig, GenerationMixin, GenerationMode
from ...generation.logits_process import LogitsProcessorList, RepetitionPenaltyLogitsProcessor
from ...generation.stopping_criteria import StoppingCriteriaList
from ...generation.utils import GenerateDecoderOnlyOutput, GenerateNonBeamOutput
from ...utils import ModelOutput, auto_docstring


if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer


@dataclass
class Lfm3AudioGenerateOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of [`Lfm3AudioForConditionalGeneration.generate`], a [`~generation.GenerateDecoderOnlyOutput`] with the
    generated speech frames.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The prompt followed by the generated tokens: text tokens as their ids, and one `audio_out_token_id` per
            generated speech frame.
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
            The codes of the generated speech frames in generation order, one frame per `audio_out_token_id` in
            `sequences`, end-of-audio frames included (every codebook holds `config.audio_eos_code`). Rows are padded
            with -1; slice to `audio_lengths` and drop the end-of-audio frames before decoding with the codec.
        audio_lengths (`torch.LongTensor` of shape `(batch_size,)`):
            The number of generated speech frames per row.
    """

    audio_codes: torch.LongTensor | None = None
    audio_lengths: torch.LongTensor | None = None


class Lfm3AudioLiveTranscriptionState:
    """State of one live transcription: the encoder follows the whole stream, the language model one context."""

    def __init__(self):
        self.encoder_cache = None
        self.past_key_values: Cache | None = None
        # the whole context: `generate` continues from a cache by slicing the full sequence
        self.inputs_embeds: torch.FloatTensor | None = None
        # fed with the next chunk's audio: a new prompt, the closing `<|text_end|>`, the last token of a spent budget
        self.pending: torch.FloatTensor | None = None
        self.finished = False


@auto_docstring(custom_intro="Output of [`Lfm3AudioForConditionalGeneration.transcribe_live`].")
@dataclass
class Lfm3AudioLiveTranscriptionOutput(ModelOutput):
    r"""
    sequences (`torch.LongTensor` of shape `(1, num_generated)`):
        The text tokens of this chunk, without the `<|text_end|>` closing the block or the final stop token.
    num_audio_tokens (`int`):
        Audio tokens the chunk completed.
    closed (`bool`):
        Whether the model ended the block itself, with `<|text_end|>` or a stop token on the final chunk, instead of
        exhausting `max_new_tokens`.
    state ([`Lfm3AudioLiveTranscriptionState`]):
        The stream state for the next call.
    """

    sequences: torch.LongTensor | None = None
    num_audio_tokens: int | None = None
    closed: bool | None = None
    state: Lfm3AudioLiveTranscriptionState | None = None


class Lfm3AudioGenerationMixin(GenerationMixin):
    """
    [`GenerationMixin`] with a text/speech mode per row: `<|audio_start|>` switches a row to speech until an
    end-of-audio frame, `<|mixed_start|>` to blocks of `interleaved_n_text` text tokens and `interleaved_n_audio`
    speech frames until `<|text_end|>` closes the text. Every speech frame takes one `audio_out_token_id` slot of
    `sequences`. The depth decoder's own [`GenerationConfig`] is serialized under the `depth_decoder_` prefix and
    overridable per call as `depth_decoder_*` kwargs.
    """

    _supported_generation_modes = [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]

    def _prepare_generation_config(
        self, generation_config: GenerationConfig | None, **kwargs: Any
    ) -> tuple[GenerationConfig, dict]:
        prefix = "depth_decoder_"
        depth_decoder_kwargs = {k[len(prefix) :]: v for k, v in kwargs.items() if k.startswith(prefix)}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith(prefix)}
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        # a passed configuration may carry prefixed fields too; the call's kwargs win
        carried = {k[len(prefix) :]: v for k, v in vars(generation_config).items() if k.startswith(prefix)}
        for key in carried:
            delattr(generation_config, prefix + key)
        self.depth_decoder.generation_config.update(**{**carried, **depth_decoder_kwargs})

        suppress = {*self._placeholder_token_ids, *(generation_config.suppress_tokens or [])}
        generation_config.suppress_tokens = sorted(suppress)
        return generation_config, model_kwargs

    def _get_stopping_criteria(self, generation_config: GenerationConfig, *args, **kwargs) -> StoppingCriteriaList:
        # `<|text_end|>` only ends the text of an interleaved block, so the loop applies the end-of-sequence ids itself
        without_eos = copy.copy(generation_config)
        without_eos._eos_token_tensor = None
        return super()._get_stopping_criteria(without_eos, *args, **kwargs)

    def _depth_logits_processors(self, device) -> tuple[RepetitionPenaltyLogitsProcessor | None, LogitsProcessorList]:
        """The depth decoder's logits processors, the repetition penalty apart: it reads the past frames."""
        processors = self._get_logits_processor(
            self.depth_decoder.generation_config,
            input_ids_seq_length=0,
            encoder_input_ids=None,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
            device=device,
        )
        repetition = next((p for p in processors if isinstance(p, RepetitionPenaltyLogitsProcessor)), None)
        warpers = LogitsProcessorList(p for p in processors if p is not repetition)
        return repetition, warpers

    def sample_audio_frames(
        self,
        hidden_states: torch.Tensor,
        history: torch.LongTensor | None = None,
        processors: tuple[RepetitionPenaltyLogitsProcessor | None, LogitsProcessorList] | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> torch.LongTensor:
        r"""
        Samples one speech frame per row from `(batch_size, hidden_size)` hidden states, revealing the codebooks in a
        random order over the passes of `generation_config.maskgit_schedule`. `history`, `(batch_size,
        num_past_frames, num_codebooks)` padded with `-1`, feeds the repetition penalty.
        """
        config = self.config
        depth_config = self.depth_decoder.generation_config
        batch_size, num_codebooks, codebook_size = hidden_states.shape[0], config.num_codebooks, config.codebook_size
        repetition, warpers = (
            processors if processors is not None else self._depth_logits_processors(hidden_states.device)
        )
        schedule = getattr(generation_config or self.generation_config, "maskgit_schedule", None)
        if schedule is None:
            schedule = [num_codebooks // 3] * 2 + [num_codebooks - 2 * (num_codebooks // 3)]
        codes = hidden_states.new_zeros((batch_size, num_codebooks), dtype=torch.long)
        revealed = torch.zeros_like(codes, dtype=torch.bool)
        order = torch.stack([torch.randperm(num_codebooks, device=hidden_states.device) for _ in range(batch_size)])
        for pick in order.split(schedule, dim=1):
            # under `device_map` the depth decoder answers on its own device
            logits = self.depth_decoder(hidden_states, codes, ~revealed).to(hidden_states.device)
            num_picked = pick.shape[1]
            scores = logits.gather(1, pick[..., None].expand(-1, -1, codebook_size)).reshape(-1, codebook_size).float()
            if repetition is not None and history is not None and history.shape[1]:
                # one penalty per (row, codebook); rows with fewer frames point their padding at a spare column
                past = history.gather(2, pick[:, None, :].expand(-1, history.shape[1], -1)).transpose(1, 2)
                past = past.reshape(-1, history.shape[1]).to(scores.device)
                past = torch.where(past < 0, torch.full_like(past, codebook_size), past)
                scores = repetition(past, torch.cat((scores, scores.new_zeros((scores.shape[0], 1))), dim=-1))
                scores = scores[:, :codebook_size]
            scores = warpers(None, scores)
            if depth_config.do_sample:
                sampled = torch.multinomial(nn.functional.softmax(scores, dim=-1), num_samples=1).squeeze(1)
            else:
                sampled = scores.argmax(-1)
            codes.scatter_(1, pick, sampled.view(batch_size, num_picked).to(codes.device))
            revealed.scatter_(1, pick, True)
        return codes

    def _prompt_modes(self, input_ids: torch.LongTensor, audio_out_codes: torch.Tensor | None) -> torch.LongTensor:
        """The mode of every row read from the prompt's last token: 0 text, 1 audio, 2 interleaved."""
        config = self.config
        if input_ids.shape[1] == 0:
            return input_ids.new_zeros(input_ids.shape[0])
        last = input_ids[:, -1]
        speaking = last == config.audio_start_token_id
        if audio_out_codes is not None and audio_out_codes.shape[0]:
            # a terminal frame in the prompt resumes the text phase, as a sampled one does
            counts = (input_ids == config.audio_out_token_id).sum(-1)
            last_frame = (counts.cumsum(0) - 1).clamp_min(0)
            terminal = (counts > 0) & (audio_out_codes[last_frame] == config.audio_eos_code).any(-1)
            speaking |= (last == config.audio_out_token_id) & ~terminal
        else:
            speaking |= last == config.audio_out_token_id
        modes = torch.where(last == config.mixed_start_token_id, 2, 0)
        return torch.where(speaking, 1, modes)

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: "BaseStreamer | None" = None,
        **model_kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor:
        """
        Overrides [`~generation.utils.GenerationMixin._sample`]: each step samples a text token for the rows in text
        mode and a depth-decoder frame for the rows in speech mode, fed back as an `audio_out_token_id` slot.
        """
        config = self.config
        if generation_config.prefill_chunk_size is not None:
            raise ValueError(
                "LFM3-Audio generation does not support `prefill_chunk_size`: the prompt's speech is not chunked"
            )
        # without a cache the prompt is recomputed every step and the sampled frames would need splicing into its codes
        cacheless = not model_kwargs.get("use_cache", True)
        pad_token_id = generation_config._pad_token_tensor
        eos_token_ids = generation_config._eos_token_tensor
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size, device = input_ids.shape[0], input_ids.device
        mode = self._prompt_modes(input_ids, model_kwargs.get("audio_out_codes"))
        if cacheless and (bool((mode != 0).any()) or model_kwargs.get("past_key_values") is not None):
            raise ValueError("Speech output and a caller's cache need `use_cache=True` (the default)")
        speaking = torch.zeros(batch_size, dtype=torch.bool, device=device)
        text_done = torch.zeros_like(speaking)
        n_text = getattr(generation_config, "interleaved_n_text", 6)
        n_audio = getattr(generation_config, "interleaved_n_audio", 12)
        slots_left = torch.full((batch_size,), n_text, device=device)
        window = getattr(self.depth_decoder.generation_config, "repetition_window", 0) or 0
        history = input_ids.new_full((batch_size, window, config.num_codebooks), -1) if window else None
        frames = []
        depth_processors = self._depth_logits_processors(device)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        this_peer_finished = False

        model_forward = (
            self.get_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )
        prefill_consumed = False
        cache = model_kwargs.get("past_key_values")
        if cache is not None and cache.get_seq_length():
            # a prompt continued from its cache had its speech consumed when the cache was filled
            for key in ("input_features", "feature_attention_mask", "audio_out_codes"):
                model_kwargs.pop(key, None)
        outputs = self._prefill(
            input_ids, generation_config, model_kwargs, is_first_iteration=not generation_config.is_assistant
        )
        if not cacheless:
            for key in ("input_features", "feature_attention_mask", "audio_out_codes"):
                model_kwargs.pop(key, None)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=device):
            if prefill_consumed:
                model_inputs = self.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=None if cacheless else 1, **model_kwargs
                )
                outputs = model_forward(**model_inputs, return_dict=True)
            prefill_consumed = True
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            if not cacheless:
                model_kwargs["audio_out_codes"] = None
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1].to(copy=True, dtype=torch.float32, device=device)
            hidden_states = outputs.last_hidden_state[:, -1]
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)
            active = unfinished_sequences.bool()
            interleaving = mode == 2
            audio_mask = active & ((mode == 1) | (interleaving & speaking))
            # speech and finished rows draw no text token, so they consume no randomness
            text_rows = (active & ~audio_mask).nonzero(as_tuple=True)[0]
            next_tokens = input_ids.new_zeros(batch_size)
            if text_rows.numel():
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores[text_rows], dim=-1)
                    next_tokens[text_rows] = torch.multinomial(probs, num_samples=1)[:, 0]
                else:
                    next_tokens[text_rows] = next_token_scores[text_rows].argmax(-1)
            terminal = torch.zeros_like(audio_mask)
            new_frames = None
            audio_rows = audio_mask.nonzero(as_tuple=True)[0]
            if audio_rows.numel():
                new_frames = self.sample_audio_frames(
                    hidden_states[audio_rows],
                    None if history is None else history[audio_rows],
                    depth_processors,
                    generation_config,
                )
                frame_terminal = (new_frames == config.audio_eos_code).any(-1)
                new_frames = torch.where(
                    frame_terminal[:, None], torch.full_like(new_frames, config.audio_eos_code), new_frames
                )
                terminal[audio_rows] = frame_terminal
                if history is not None:
                    shifted = torch.cat((history[audio_rows][:, 1:], new_frames[:, None]), dim=1)
                    history[audio_rows] = torch.where(frame_terminal[:, None, None], history[audio_rows], shifted)
                next_tokens = torch.where(audio_mask, config.audio_out_token_id, next_tokens)
                step_frames = input_ids.new_full((batch_size, config.num_codebooks), -1)
                step_frames[audio_rows] = new_frames
                frames.append(step_frames)
            elif frames:
                frames.append(input_ids.new_full((batch_size, config.num_codebooks), -1))

            text_slot = active & ~audio_mask
            is_text_end = text_slot & (next_tokens == config.text_end_token_id)
            is_eos = torch.zeros_like(text_slot)
            if eos_token_ids is not None:
                # `<|text_end|>` listed as an end-of-sequence id still closes the text of an interleaved block first
                is_eos = text_slot & ~(is_text_end & interleaving) & torch.isin(next_tokens, eos_token_ids)
            finished_now = is_eos
            closes_text = is_text_end & interleaving
            text_done |= closes_text
            speaking |= closes_text
            # the mode markers only open a block; inside one they are ordinary text
            opens_audio = text_slot & ~interleaving & (next_tokens == config.audio_start_token_id)
            opens_block = text_slot & ~interleaving & (next_tokens == config.mixed_start_token_id)
            mode = torch.where(opens_audio, 1, mode)
            mode = torch.where(opens_block, 2, mode)
            slots_left = torch.where(opens_block, n_text, slots_left)
            speaking &= ~opens_block
            text_done &= ~opens_block
            mode = torch.where(terminal, 0, mode)
            speaking &= ~terminal
            text_done &= ~terminal
            # slots are counted by the phase at the start of the step
            counting = active & interleaving & ~text_done
            slots_left = torch.where(counting, slots_left - 1, slots_left)
            flip = counting & (slots_left == 0)
            speaking = torch.where(flip, ~speaking, speaking)
            slots_left = torch.where(flip, torch.where(speaking, n_audio, n_text), slots_left)

            if pad_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if new_frames is not None:
                model_kwargs["audio_out_codes"] = new_frames
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            unfinished_sequences = unfinished_sequences & ~finished_now.long()
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if frames:
                stacked = torch.stack(frames, dim=-1)
                valid = stacked[:, 0] != -1
                order = valid.int().argsort(dim=-1, descending=True, stable=True)
                audio_codes = stacked.gather(2, order[:, None, :].expand(-1, config.num_codebooks, -1))
                audio_lengths = valid.sum(-1)
                audio_codes = audio_codes[:, :, : int(audio_lengths.max())]
            else:
                audio_lengths = input_ids.new_zeros(batch_size)
                audio_codes = input_ids.new_full((batch_size, config.num_codebooks, 0), -1)
            return Lfm3AudioGenerateOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                audio_codes=audio_codes,
                audio_lengths=audio_lengths,
            )
        return input_ids

    @torch.no_grad()
    def transcribe_live(
        self,
        input_features: torch.FloatTensor,
        state: Lfm3AudioLiveTranscriptionState | None = None,
        input_ids: torch.LongTensor | None = None,
        is_final: bool = False,
        max_new_tokens: int = 48,
        suppress_token_ids: list[int] | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> Lfm3AudioLiveTranscriptionOutput:
        """
        Transcribe one chunk of a live stream with a checkpoint trained for live ASR. The audio goes inside the
        assistant turn in chunks of `generation_config.interleaved_n_audio` encoder tokens, each followed by its text:

            <prompt> [A x 12] text<|text_end|> [A x 12] text<|text_end|> ... [A x r] <|mixed_end|> text<|text_end|><|im_end|>

        Args:
            input_features (`torch.FloatTensor` of shape `(1, num_mel_bins, num_frames)`):
                The chunk's new mel frames, normally `8 * generation_config.interleaved_n_audio`. A chunk that
                completes no audio token decodes no text.
            state ([`Lfm3AudioLiveTranscriptionState`], *optional*):
                The state returned by the previous call. Omit for a new stream.
            input_ids (`torch.LongTensor` of shape `(1, prompt_length)`, *optional*):
                The prompt of a new language-model context (the system turn and the assistant header), required
                for a new stream. With a `state`, it starts the context over while the encoder keeps following the
                stream, the way training samples of at most 30 seconds are chained.
            is_final (`bool`, *optional*, defaults to `False`):
                Flush the encoder and decode the closing block, which ends at `<|text_end|>` or a stop token.
            max_new_tokens (`int`, *optional*, defaults to 48):
                Text budget of the block; a block that exhausts it is still closed with `<|text_end|>`.
            suppress_token_ids (`list[int]`, *optional*):
                Tokens never valid before the final chunk, on top of `<|audio_start|>`, `<|mixed_start|>` and the
                end-of-turn token: pass `<|mixed_end|>` and `<|endoftext|>`.
            stop_token_ids (`list[int]`, *optional*):
                Tokens ending the final block on top of the end-of-turn token: pass `<|endoftext|>` and `<|im_start|>`.

        Returns:
            [`Lfm3AudioLiveTranscriptionOutput`]: the chunk's text tokens (the final block starts with
            `<|mixed_end|>`), the number of audio tokens it completed, and the updated state.
        """
        config = self.config
        if state is None:
            if input_ids is None:
                raise ValueError("A new stream needs the prompt `input_ids`")
            state = Lfm3AudioLiveTranscriptionState()
        eos_token_ids = config.text_config.eos_token_id
        eos_token_ids = (
            [] if eos_token_ids is None else [eos_token_ids] if isinstance(eos_token_ids, int) else list(eos_token_ids)
        )
        text_end = config.text_end_token_id
        stop = {text_end, *(eos_token_ids + list(stop_token_ids or []) if is_final else [])}
        # the end-of-turn ids wait for the final chunk; `<|text_end|>` closes every block, so it is never suppressed
        suppress = {*self._placeholder_token_ids, config.audio_start_token_id, config.mixed_start_token_id}
        if not is_final:
            suppress |= {*eos_token_ids, *(suppress_token_ids or [])}
        suppress -= {text_end}

        # the encoder runs before a restart replaces the context, so a rejected chunk leaves the context intact
        embeddings, state.encoder_cache = self.model.get_audio_features_stream(
            input_features, state.encoder_cache, is_final
        )
        embed_tokens = self.get_input_embeddings()
        embeddings = embeddings.to(dtype=embed_tokens.weight.dtype)
        if input_ids is not None:
            state.past_key_values = DynamicCache(config=config.text_config)
            state.pending = embed_tokens(input_ids)
            state.inputs_embeds = state.pending[:, :0]
        num_audio_tokens = embeddings.shape[1]
        generated, closed = [], False
        if num_audio_tokens or is_final:
            if state.pending is not None:
                embeddings = torch.cat((state.pending, embeddings.to(state.pending.device)), 1)
            state.pending = None
            state.inputs_embeds = torch.cat((state.inputs_embeds, embeddings.to(state.inputs_embeds.device)), 1)
            live_config = GenerationConfig(
                max_new_tokens=max(max_new_tokens, 1),
                do_sample=False,
                suppress_tokens=sorted(suppress),
                eos_token_id=sorted(stop),
                pad_token_id=config.text_config.pad_token_id,
                return_dict_in_generate=False,
            )
            sequences = self.generate(
                inputs_embeds=state.inputs_embeds,
                past_key_values=state.past_key_values,
                generation_config=live_config,
            )[0].tolist()
            # `generate` feeds every token but the last one it sampled
            fed = torch.tensor([sequences[:-1]], dtype=torch.long, device=state.inputs_embeds.device)
            state.inputs_embeds = torch.cat((state.inputs_embeds, embed_tokens(fed)), 1)
            if max_new_tokens < 1:
                sequences = []  # no text budget: the chunk only advances the context
            closed = bool(sequences) and sequences[-1] in stop
            generated = sequences[:-1] if closed else sequences
            unfed = [] if closed or is_final else sequences[-1:]
            state.pending = embed_tokens(fed.new_tensor([unfed])) if unfed else None
            if is_final:
                state.finished = True
            else:
                # the block closes with `<|text_end|>`, fed together with the next chunk's audio
                closing = embed_tokens(fed.new_tensor([[text_end]]))
                state.pending = closing if state.pending is None else torch.cat((state.pending, closing), 1)
        return Lfm3AudioLiveTranscriptionOutput(
            sequences=torch.tensor([generated], dtype=torch.long, device=input_features.device),
            num_audio_tokens=num_audio_tokens,
            closed=closed,
            state=state,
        )
