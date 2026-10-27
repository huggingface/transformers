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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from ...generation import GenerateDecoderOnlyOutput, GenerationConfig, GenerationMixin, GenerationMode
from ...generation.logits_process import LogitsProcessorList
from ...generation.stopping_criteria import MaxLengthCriteria, StoppingCriteria, StoppingCriteriaList
from ...generation.utils import ALL_CACHE_NAMES, GenerateNonBeamOutput
from ...utils import logging


if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer


logger = logging.get_logger(__name__)

# Defaults of the Pocket TTS-specific generation parameters, the released checkpoints' values, used when neither the
# `generate` call nor the generation config sets them.
_GENERATION_DEFAULTS = {"num_flow_steps": 1, "eos_threshold": -4.0, "min_frames_before_eos": 6}


@dataclass
class PocketTTSGenerateOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of `PocketTTSForConditionalGeneration.generate`.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The prompt token ids followed by one `audio_latent_token_id` per generated frame. The audio itself is in
            `audio_latents`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            The end-of-speech logit of every generated frame, one tensor of shape `(batch_size,)` per frame.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Same as `scores`: Pocket TTS has no vocabulary, so there is nothing to process between the two.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated frame) of tuples (one element for each layer) of `torch.FloatTensor`
            of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated frame) of tuples (one element for each layer) of `torch.FloatTensor`
            of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True`):
            The key/value cache of the backbone at the end of generation.
        audio_latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`):
            The generated latents, in the audio tokenizer's latent space, right-padded with zeros past each sample's own
            frames. [`PocketTTSProcessor.decode`] turns them into waveforms.
        audio_latents_mask (`torch.LongTensor` of shape `(batch_size, num_frames)`):
            Mask of each sample's frames (`1`), the end-of-speech frame and the frames after it included, as opposed to
            padding (`0`).
    """

    audio_latents: torch.FloatTensor | None = None
    audio_latents_mask: torch.LongTensor | None = None


class PocketTTSEosThresholdCriteria(StoppingCriteria):
    """
    Pocket TTS's end-of-speech rule. The backbone emits one end-of-speech logit per frame; the first frame, from frame
    `min_frames_before_eos` on, whose logit is above `eos_threshold` starts a countdown of `frames_after_eos` more frames,
    which lets the codec close the utterance, and generation stops once they are out. The frame that crossed the
    threshold and the tail frames are all kept.

    Expects `scores` to be the end-of-speech logits of the frame that was just generated, of shape `(batch_size,)`.

    Args:
        eos_threshold (`float`):
            Logit above which a frame counts as end of speech.
        frames_after_eos (`torch.LongTensor` of shape `(batch_size,)` or `int`):
            Number of frames generated after the end-of-speech frame, per sample.
        min_frames_before_eos (`int`, *optional*, defaults to 0):
            Number of first frames whose end-of-speech logit is ignored.
    """

    def __init__(self, eos_threshold: float, frames_after_eos: torch.LongTensor | int, min_frames_before_eos: int = 0):
        self.eos_threshold = eos_threshold
        self.frames_after_eos = frames_after_eos
        self.min_frames_before_eos = min_frames_before_eos
        # Index of the frame the next call sees.
        self.frame_index = 0
        # Frames left to generate per sample; -1 until end of speech is seen. Allocated on the first call.
        self.remaining = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        if self.remaining is None:
            self.remaining = torch.full((batch_size,), -1, dtype=torch.long, device=input_ids.device)
            if not isinstance(self.frames_after_eos, torch.Tensor):
                self.frames_after_eos = torch.full((batch_size,), self.frames_after_eos, dtype=torch.long)
            # per-text values (processor output) follow the rows when generation expands the batch
            # (`num_return_sequences`), as every other per-row input does
            if self.frames_after_eos.numel() != batch_size:
                self.frames_after_eos = self.frames_after_eos.repeat_interleave(
                    batch_size // self.frames_after_eos.numel()
                )
            self.frames_after_eos = self.frames_after_eos.to(input_ids.device)
        can_end = self.frame_index >= self.min_frames_before_eos
        self.frame_index += 1
        just_ended = (self.remaining < 0) & (scores.to(input_ids.device) > self.eos_threshold) & can_end
        self.remaining = torch.where(just_ended, self.frames_after_eos, self.remaining)
        is_done = self.remaining == 0
        self.remaining = torch.where(self.remaining > 0, self.remaining - 1, self.remaining)
        return is_done


class PocketTTSGenerationMixin(GenerationMixin):
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> PocketTTSGenerateOutput | list[torch.FloatTensor]:
        r"""
        Generates speech. This method overrides [`~generation.utils.GenerationMixin.generate`] to default
        `max_new_tokens` to the original runtime's frame budget, `ceil((num_text_tokens / 3 + 2) * frame_rate)` for the
        longest text of the batch. Everything else goes through the regular generation hooks.

        Parameters:
            inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The `input_ids` prepared by [`PocketTTSProcessor`]. Can also be passed as `input_ids=`.
            kwargs (`dict[str, Any]`, *optional*):
                `attention_mask`, the voice prompt (`audio_prompt_latents` and `audio_prompt_latents_mask`, or the
                cache of a voice prefix as `past_key_values`, the shipped voices, with prompts of the text only),
                `frames_after_eos` from the processor, the generation parameters (`do_sample`,
                `temperature`, `max_new_tokens`, ...), and the Pocket TTS-specific `num_flow_steps`, `eos_threshold` and
                `min_frames_before_eos`, which default to the generation config.

        Returns:
            [`~models.pocket_tts.generation_pocket_tts.PocketTTSGenerateOutput`] when `return_dict_in_generate=True` (the
            checkpoints' generation config): the padded `audio_latents` and their `audio_latents_mask`, which
            [`PocketTTSProcessor.decode`] decodes as one batch. Otherwise the list of generated latents, one
            `(num_frames, latent_dim)` tensor per sample, each decoded on its own.
        """
        if inputs is not None:
            kwargs["input_ids"] = inputs
        input_ids = kwargs.get("input_ids")
        if input_ids is not None and not (input_ids[:, -1] == self.config.audio_bos_token_id).all():
            raise ValueError(
                "Every prompt must end with the first audio position (`audio_bos_token_id`), from which the first frame "
                "is generated, and be left-padded: prepare the inputs with `PocketTTSProcessor`."
            )
        if (
            input_ids is not None
            and kwargs.get("audio_prompt_latents") is None
            and (input_ids == self.config.audio_prompt_token_id).any()
        ):
            raise ValueError(
                "The prompts hold a voice placeholder but no voice: pass the voice to the processor, as an `audio` item of "
                "the conversation or as `audio_prompt_latents`."
            )
        past_key_values = kwargs.get("past_key_values")
        attention_mask = kwargs.get("attention_mask")
        if (
            past_key_values is not None
            and input_ids is not None
            and attention_mask is not None
            and attention_mask.shape[1] == input_ids.shape[1]
            and not (input_ids == self.config.audio_prompt_token_id).any()
        ):
            # The voice is the cache of its prefix (a shipped voice) and the prompts hold the text only: the mask covers
            # the cached positions too, so `generate` takes `input_ids` as the new tokens and derives their positions.
            cached = attention_mask.new_ones(attention_mask.shape[0], past_key_values.get_seq_length())
            kwargs["attention_mask"] = torch.cat([cached, attention_mask], dim=-1)
        generation_config = kwargs.get("generation_config")
        length_is_set = any(
            getattr(source, key, None) is not None if source is not kwargs else kwargs.get(key) is not None
            for source in (kwargs, generation_config)
            for key in ("max_new_tokens", "max_length")
        )
        if not length_is_set and input_ids is not None:
            # The original's budget, from the longest text: a shorter row whose end of speech never fires can run past its
            # own budget, up to the batch's.
            placeholder_ids = torch.tensor(self._placeholder_token_ids(), device=input_ids.device)
            is_text = ~torch.isin(input_ids, placeholder_ids)
            if attention_mask is not None:
                is_text &= attention_mask[:, -input_ids.shape[1] :].bool()
            num_text_tokens = int(is_text.sum(-1).max())
            kwargs["max_new_tokens"] = math.ceil((num_text_tokens / 3 + 2) * self.config.frame_rate)
        return super().generate(**kwargs)

    def _placeholder_token_ids(self) -> list[int]:
        """The ids of the prompt's audio positions (voice and generated audio), which are not text."""
        return [
            self.config.audio_prompt_bos_token_id,
            self.config.audio_prompt_token_id,
            self.config.audio_bos_token_id,
            self.config.audio_latent_token_id,
        ]

    def _prepare_generation_config(
        self, generation_config: GenerationConfig | None, **kwargs: Any
    ) -> tuple[GenerationConfig, dict]:
        """
        Resolves the Pocket TTS-specific runtime knobs: `num_flow_steps`, `eos_threshold` and `min_frames_before_eos`
        (kwargs defaulting to the generation config, then to the released checkpoints' values) and `frames_after_eos` (a
        processor output; 1 with a warning when absent), and restricts generation to the greedy and sampling modes, the
        only two the flow sampler has.
        """
        num_flow_steps = kwargs.pop("num_flow_steps", None)
        eos_threshold = kwargs.pop("eos_threshold", None)
        min_frames_before_eos = kwargs.pop("min_frames_before_eos", None)
        frames_after_eos = kwargs.pop("frames_after_eos", None)

        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)

        if num_flow_steps is not None:
            generation_config.num_flow_steps = num_flow_steps
        if eos_threshold is not None:
            generation_config.eos_threshold = eos_threshold
        if min_frames_before_eos is not None:
            generation_config.min_frames_before_eos = min_frames_before_eos
        for key, default in _GENERATION_DEFAULTS.items():
            if getattr(generation_config, key, None) is None:
                setattr(generation_config, key, default)
        if self.config.flow_head_config.flow_type == "drifting" and generation_config.num_flow_steps != 1:
            logger.warning_once("The drifting flow head samples in one call: `num_flow_steps` is ignored.")
        if frames_after_eos is None:
            logger.warning_once(
                "`frames_after_eos` was not passed; defaulting to 1 frame after end of speech. Pass the processor's "
                "`frames_after_eos` to match the original runtime (which depends on the text length)."
            )
            frames_after_eos = 1
        generation_config.frames_after_eos = frames_after_eos

        generation_mode = generation_config.get_generation_mode()
        if generation_mode not in (GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE):
            raise ValueError(
                f"Generation mode {generation_mode} is not supported by PocketTTS: its flow sampler is either deterministic "
                "(`do_sample=False`) or driven by Gaussian noise (`do_sample=True`)."
            )
        return generation_config, model_kwargs

    def _get_logits_processor(
        self, *args, logits_processor: LogitsProcessorList | None = None, **kwargs
    ) -> LogitsProcessorList:
        # Pocket TTS has no vocabulary: `temperature` is the variance of the flow sampler's noise, read in `_sample`, and
        # no other processor applies.
        if logits_processor:
            logger.warning("Pocket TTS samples continuous latents, not tokens: the logits processors are ignored.")
        return LogitsProcessorList()

    def _get_stopping_criteria(self, generation_config: GenerationConfig, *args, **kwargs) -> StoppingCriteriaList:
        criteria = super()._get_stopping_criteria(generation_config, *args, **kwargs)
        kept_criteria = StoppingCriteriaList()
        for criterion in criteria:
            if isinstance(criterion, MaxLengthCriteria):
                kept_criteria.append(criterion)
            else:
                logger.warning(
                    f"PocketTTS does not support the {criterion.__class__.__name__} stopping criterion, ignoring it."
                )
        kept_criteria.append(
            PocketTTSEosThresholdCriteria(
                generation_config.eos_threshold,
                generation_config.frames_after_eos,
                generation_config.min_frames_before_eos,
            )
        )
        return kept_criteria

    def prepare_inputs_for_generation(self, *args, is_first_iteration: bool | None = False, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, is_first_iteration=is_first_iteration, **kwargs)
        # The voice prompt is scattered into its placeholders at prefill; with a cache, later steps carry none.
        if not is_first_iteration and model_inputs.get("use_cache", True):
            for key in ("audio_prompt_latents", "audio_prompt_latents_mask"):
                model_inputs.pop(key, None)
        return model_inputs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> GenerateNonBeamOutput | list[torch.FloatTensor]:
        """
        This method overrides [~generation.utils.GenerationMixin._sample]. To ease maintenance, modifications are
        marked with the comment "PocketTTS specific".

        Pocket TTS generates one continuous latent per step instead of a token: the backbone runs on the previous latent
        (the learnt BOS latent first), its last hidden state conditions the flow head, which integrates the velocity field
        from Gaussian noise (zero when `do_sample=False`) to the next latent. The end-of-speech logit of the same hidden
        state drives `PocketTTSEosThresholdCriteria`. The latents are mapped to the audio tokenizer's space after the loop;
        the processor decodes them.
        """
        # init values
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
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

        model_forward = (
            self.get_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )

        # *************** PocketTTS specific ***************
        if streamer is not None:
            logger.warning("Pocket TTS generates audio, not tokens; the streamer is ignored.")
        num_flow_steps = generation_config.num_flow_steps
        noise_scale = math.sqrt(generation_config.temperature) if do_sample else 0.0
        latent_dim = self.config.latent_dim
        use_cache = model_kwargs.get("use_cache", True)

        # The prompt ends with the BOS audio position, so the prefill yields the first frame like a token model yields
        # its first token; every later frame is one step on the previous latent.
        latent_placeholder = torch.full((batch_size, 1), self.config.audio_latent_token_id, device=input_ids.device)
        all_latents = self.model.bos_latent.new_zeros(batch_size, 0, latent_dim)
        last_latent = None
        num_frames = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        # ============================================

        prefill_consumed = False
        outputs = self._prefill(input_ids, generation_config, model_kwargs, is_first_iteration=True)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if prefill_consumed:
                # *************** PocketTTS specific ***************
                # With a cache the step input is the last latent only; without one the whole latent history is
                # re-embedded.
                # `last_latent` is a fresh contiguous tensor: a constant stride across steps, so a compiled forward does
                # not recompile.
                model_kwargs["audio_latents"] = last_latent if use_cache else all_latents
                # ============================================
                model_inputs = self.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=1 if use_cache else None, **model_kwargs
                )
                model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
                outputs = model_forward(**model_inputs, return_dict=True)
            prefill_consumed = True

            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            if synced_gpus and this_peer_finished:
                continue

            # *************** PocketTTS specific ***************
            condition = outputs.last_hidden_state[:, -1, :]
            eos_logits = outputs.logits[:, -1, 0].to(copy=True, dtype=torch.float32, device=input_ids.device)
            # ============================================

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                # Pocket TTS has no vocabulary: `scores` and `logits` are both the raw end-of-speech logits
                if output_scores:
                    scores += (eos_logits,)
                if output_logits:
                    raw_logits += (eos_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # *************** PocketTTS specific ***************
            noise = torch.randn(batch_size, latent_dim, device=condition.device, dtype=condition.dtype) * noise_scale
            next_latent = self.sample_latent(condition, noise, num_flow_steps)
            last_latent = next_latent[:, None, :]
            all_latents = torch.cat([all_latents, last_latent.to(all_latents.device)], dim=1)
            # one placeholder per generated frame; it is the next step's input position
            input_ids = torch.cat([input_ids, latent_placeholder], dim=-1)
            # The frame just generated belongs to every sample that was still running, the stopping frame included.
            num_frames += unfinished_sequences
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, eos_logits)
            # ============================================
            this_peer_finished = unfinished_sequences.max() == 0

            del outputs

        # *************** PocketTTS specific ***************
        # a finished sample keeps running with the batch: its later frames are zeroed
        audio_latents = self.denormalize_latents(all_latents)
        frame_positions = torch.arange(audio_latents.shape[1], device=num_frames.device)
        audio_latents_mask = (frame_positions < num_frames[:, None]).long()
        audio_latents = audio_latents * audio_latents_mask[..., None].to(audio_latents.device, audio_latents.dtype)
        # ============================================

        if return_dict_in_generate:
            cache = None
            if any(cache_key in model_kwargs for cache_key in ALL_CACHE_NAMES):
                cache_key = next(cache_key for cache_key in ALL_CACHE_NAMES if cache_key in model_kwargs)
                cache = model_kwargs[cache_key]
            # *************** PocketTTS specific ***************
            return PocketTTSGenerateOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=cache,
                audio_latents=audio_latents,
                audio_latents_mask=audio_latents_mask,
            )
        else:
            # The generated ids are placeholders only, so each sample's latents are what a caller without the dict gets.
            return [audio_latents[sample_idx, : num_frames[sample_idx]] for sample_idx in range(batch_size)]
            # ============================================
