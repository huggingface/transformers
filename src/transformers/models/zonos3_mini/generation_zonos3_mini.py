# Copyright 2026 Zyphra and The HuggingFace Inc. team. All rights reserved.
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

from typing import Any, Optional

import torch
import torch.nn as nn

from ...generation import (
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    GenerationMode,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from ...generation.logits_process import (
    ClassifierFreeGuidanceLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessor,
    MinPLogitsWarper,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from ...generation.stopping_criteria import EosTokenCriteria, MaxLengthCriteria
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateNonBeamOutput
from ...utils import add_start_docstrings, logging


logger = logging.get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks + 1)`):
            The rows so far: the prompt and the generated rows.
        scores (`torch.FloatTensor` of shape `(batch_size * num_codebooks, codebook_size + 2)`):
            The logits of each audio column of the next row (the delay pattern processor, which runs first, also takes
            them as `(batch_size, num_codebooks * (codebook_size + 2))`).

    Return:
        `torch.FloatTensor` of shape `(batch_size * num_codebooks, codebook_size + 2)`: The processed scores.
"""


class Zonos3MiniDelayPatternLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing the ZONOS3-Mini audio vocabulary and delay pattern: the padding id and the ids past the
    codebook other than end-of-audio are never sampled, and while a prompt without an audio prefix starts, the delay
    triangle (codebook `j` of the first `j` rows) stays padding.

    Args:
        prompt_length (`int`):
            The number of prompt rows.
        num_codebooks (`int`):
            The number of audio columns.
        codebook_size (`int`):
            The number of codes of each codebook.
        audio_eos_token_id (`int`):
            The end-of-audio id.
        audio_pad_token_id (`int`):
            The padding id.
    """

    def __init__(
        self,
        prompt_length: int,
        num_codebooks: int,
        codebook_size: int,
        audio_eos_token_id: int,
        audio_pad_token_id: int,
    ):
        self.prompt_length = prompt_length
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.cold_start = None

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        scores = scores.reshape(batch_size, self.num_codebooks, -1)
        vocab_size = scores.shape[-1]
        token_ids = torch.arange(vocab_size, device=scores.device)
        invalid = (token_ids == self.audio_pad_token_id) | (
            (token_ids >= self.codebook_size) & (token_ids != self.audio_eos_token_id)
        )
        scores = scores.masked_fill(invalid, -float("inf"))

        if self.cold_start is None:
            # the prompt's last row has no audio: generation starts the delay triangle
            last_prompt_row = input_ids[:, self.prompt_length - 1, : self.num_codebooks]
            self.cold_start = (last_prompt_row == self.audio_pad_token_id).all(-1)
        step = input_ids.shape[1] - self.prompt_length
        if step < self.num_codebooks - 1 and self.cold_start.any():
            delayed = torch.arange(self.num_codebooks, device=scores.device) > step
            mask = self.cold_start.to(scores.device)[:, None] & delayed[None]
            only_pad = torch.full_like(scores[0, 0], -float("inf"))
            only_pad[self.audio_pad_token_id] = 0.0
            scores = torch.where(mask[..., None], only_pad, scores)
        return scores.view(-1, vocab_size)


class Zonos3MiniRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] penalizing, on the first `num_codebooks_penalized` audio columns, the codes generated in the last
    `window` rows: a seen code's score `s` becomes `s / penalty` if positive, `s * penalty` otherwise.

    Args:
        penalty (`float`):
            The repetition penalty (no-op when `<= 1`).
        window (`int`):
            The number of generated rows looked at.
        num_codebooks_penalized (`int`):
            The number of leading audio columns penalized.
        prompt_length (`int`):
            The number of prompt rows (never penalized).
        num_codebooks (`int`):
            The number of audio columns.
        codebook_size (`int`):
            The number of codes of each codebook.
    """

    def __init__(
        self,
        penalty: float,
        window: int,
        num_codebooks_penalized: int,
        prompt_length: int,
        num_codebooks: int,
        codebook_size: int,
    ):
        self.penalty = penalty
        self.window = window
        self.num_codebooks_penalized = min(num_codebooks_penalized, num_codebooks)
        self.prompt_length = prompt_length
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_generated = input_ids.shape[1] - self.prompt_length
        active = min(self.window, num_generated)
        if self.penalty <= 1.0 or active <= 0 or self.num_codebooks_penalized <= 0:
            return scores
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[-1]
        scores = scores.view(batch_size, self.num_codebooks, vocab_size)
        recent = input_ids[:, -active:, : self.num_codebooks_penalized].transpose(1, 2).to(scores.device)
        valid = (recent >= 0) & (recent < self.codebook_size)
        counts = torch.zeros(
            (batch_size, self.num_codebooks_penalized, vocab_size), dtype=torch.int32, device=scores.device
        )
        counts.scatter_add_(-1, recent.clamp(0, vocab_size - 1), valid.to(torch.int32))
        penalized = scores[:, : self.num_codebooks_penalized]
        adjusted = torch.where(penalized > 0, penalized / self.penalty, penalized * self.penalty)
        scores = torch.cat(
            [torch.where(counts > 0, adjusted, penalized), scores[:, self.num_codebooks_penalized :]], 1
        )
        return scores.view(-1, vocab_size)


class Zonos3MiniGenerationMixin(GenerationMixin):
    # Logits processors that only read the scores, and so are safe to apply per audio column
    _supported_logits_processor_types = (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
        MinPLogitsWarper,
        InfNanRemoveLogitsProcessor,
    )

    def _prepare_generation_config(
        self, generation_config: GenerationConfig | None, **kwargs: Any
    ) -> tuple[GenerationConfig, dict]:
        repetition_kwargs = {
            key: kwargs.pop(key) for key in ("repetition_window", "repetition_codebooks") if key in kwargs
        }
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        for key, value in repetition_kwargs.items():
            setattr(generation_config, key, value)

        original_get_generation_mode = generation_config.get_generation_mode

        def patched_get_generation_mode(assistant_model=None):
            generation_mode = original_get_generation_mode(assistant_model)
            if generation_mode not in [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]:
                raise ValueError(
                    f"Generation mode {generation_mode} is not supported for ZONOS3-Mini. Please set generation "
                    "parameters to use greedy or sampling generation."
                )
            return generation_mode

        generation_config.get_generation_mode = patched_get_generation_mode
        return generation_config, model_kwargs

    def _get_logits_processor(self, generation_config: GenerationConfig, input_ids_seq_length=None, *args, **kwargs):
        # the repetition penalty and the speaker guidance are applied by the ZONOS3-Mini processors below
        repetition_penalty, guidance_scale = generation_config.repetition_penalty, generation_config.guidance_scale
        generation_config.repetition_penalty = generation_config.guidance_scale = None
        try:
            parent_processors = super()._get_logits_processor(generation_config, input_ids_seq_length, *args, **kwargs)
        finally:
            generation_config.repetition_penalty, generation_config.guidance_scale = repetition_penalty, guidance_scale

        unsupported = [p for p in parent_processors if not isinstance(p, self._supported_logits_processor_types)]
        if unsupported:
            raise ValueError(
                "ZONOS3-Mini generates audio codebook logits, not text logits. The following logits processors are not "
                f"compatible: {[type(p).__name__ for p in unsupported]}. Only the following processors are supported: "
                f"{[t.__name__ for t in self._supported_logits_processor_types]}."
            )

        config = self.config
        processors = LogitsProcessorList()
        if guidance_scale is not None and guidance_scale != 1:
            processors.append(ClassifierFreeGuidanceLogitsProcessor(guidance_scale))
        processors.append(
            Zonos3MiniDelayPatternLogitsProcessor(
                prompt_length=input_ids_seq_length,
                num_codebooks=config.num_codebooks,
                codebook_size=config.codebook_size,
                audio_eos_token_id=config.audio_eos_token_id,
                audio_pad_token_id=config.audio_pad_token_id,
            )
        )
        if repetition_penalty is not None:
            processors.append(
                Zonos3MiniRepetitionPenaltyLogitsProcessor(
                    penalty=repetition_penalty,
                    window=getattr(generation_config, "repetition_window", None) or 0,
                    num_codebooks_penalized=getattr(generation_config, "repetition_codebooks", None)
                    or config.num_codebooks,
                    prompt_length=input_ids_seq_length,
                    num_codebooks=config.num_codebooks,
                    codebook_size=config.codebook_size,
                )
            )
        processors.extend(parent_processors)
        return processors

    def _get_stopping_criteria(self, *args, **kwargs) -> StoppingCriteriaList:
        # items end on their end-of-audio row countdown (see `_sample`): the text EOS criteria never applies to rows
        criteria = super()._get_stopping_criteria(*args, **kwargs)
        kept_criteria = StoppingCriteriaList()
        for criterion in criteria:
            if isinstance(criterion, MaxLengthCriteria):
                kept_criteria.append(criterion)
            elif not isinstance(criterion, EosTokenCriteria):
                logger.warning_once(
                    f"ZONOS3-Mini does not support the {criterion.__class__.__name__} stopping criteria, it is ignored."
                )
        return kept_criteria

    def prepare_inputs_for_generation(self, input_ids, next_sequence_length=None, is_first_iteration=False, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, next_sequence_length=next_sequence_length, is_first_iteration=is_first_iteration, **kwargs
        )
        if not is_first_iteration:
            # the speaker and condition inputs only describe prompt rows
            for key in (
                "speaker_embeddings",
                "speaker_input_features",
                "speaker_input_features_mask",
                "speaker_token_mask",
                "condition_ids",
            ):
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
    ) -> GenerateNonBeamOutput | torch.LongTensor:
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # ===========================
        # BELOW DIFFERENCES WITH GenerationMixin._sample()
        # with speaker guidance, the batch holds the conditional prompts then their unconditional twins
        config = self.config
        num_codebooks = config.num_codebooks
        full_batch_size = input_ids.shape[0]
        batch_size = full_batch_size // 2 if self._uses_guidance else full_batch_size
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        # rows left once the end-of-audio was sampled: the delay pattern needs `num_codebooks + 1` rows to flush
        eos_countdown = torch.full((batch_size,), -1, dtype=torch.long, device=input_ids.device)
        finished_row = torch.full((num_codebooks + 1,), config.audio_pad_token_id, device=input_ids.device)
        finished_row[-1] = config.text_pad_token_id
        this_peer_finished = False
        # ============================

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

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if prefill_consumed:
                next_sequence_length = 1 if model_kwargs["use_cache"] else None
                model_inputs = self.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=next_sequence_length, **model_kwargs
                )
                outputs = model_forward(**model_inputs, return_dict=True)
            prefill_consumed = True
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if synced_gpus and this_peer_finished:
                continue

            # ===========================
            # BELOW DIFFERENCES WITH GenerationMixin._sample()
            next_token_logits = outputs.logits[:, -1].to(copy=True, dtype=torch.float32, device=input_ids.device)
            # the delay pattern processor turns them into one row per audio column (after speaker guidance, which
            # combines the conditional and unconditional halves of the batch)
            next_token_scores = logits_processor(
                input_ids[:batch_size], next_token_logits.reshape(full_batch_size, -1)
            )
            vocab_size = next_token_scores.shape[-1]

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores.reshape(batch_size, num_codebooks, vocab_size),)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_codes = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_codes = torch.argmax(next_token_scores, dim=-1)
            next_codes = next_codes.reshape(batch_size, num_codebooks)

            # finished items continue with all-padding rows
            text_column = torch.full((batch_size, 1), config.text_pad_token_id, device=input_ids.device)
            next_rows = torch.cat([next_codes, text_column], dim=-1)
            next_rows = torch.where(unfinished_sequences[:, None].bool(), next_rows, finished_row)

            # an item ends `num_codebooks + 1` rows after its first end-of-audio
            has_eos = (next_codes == config.audio_eos_token_id).any(-1) & unfinished_sequences.bool()
            eos_countdown = torch.where((eos_countdown < 0) & has_eos, num_codebooks + 1, eos_countdown)
            eos_countdown = torch.where(eos_countdown > 0, eos_countdown - 1, eos_countdown)
            unfinished_sequences = unfinished_sequences & (eos_countdown != 0).long()

            if self._uses_guidance:
                next_rows = next_rows.repeat(2, 1)
            input_ids = torch.cat([input_ids, next_rows[:, None]], dim=1)
            if streamer is not None:
                streamer.put(next_rows[:batch_size].cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids[:batch_size], scores)
            this_peer_finished = unfinished_sequences.max() == 0
            # ============================

            del outputs

        if streamer is not None:
            streamer.end()

        sequences = input_ids[:batch_size]
        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=sequences,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                cache=model_kwargs.get("cache"),
            )
        return sequences

    def generate(
        self,
        input_ids: torch.LongTensor | None = None,
        uncond_input_ids: torch.LongTensor | None = None,
        uncond_attention_mask: torch.LongTensor | None = None,
        uncond_condition_ids: torch.LongTensor | None = None,
        generation_config: GenerationConfig | None = None,
        **kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor:
        r"""
        Generates the delayed codec rows of the speech, until every item has ended (`num_codebooks + 1` rows after its
        first end-of-audio) or `max_new_tokens` rows are generated. Decode them with [`Zonos3MiniProcessor.decode`].

        Speaker guidance (classifier-free guidance on the voice) runs when `guidance_scale > 1`: it needs the
        unconditional prompts the processor returns with `speaker_guidance=True`.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks + 1)`):
                The prompt rows (see [`Zonos3MiniProcessor`]).
            uncond_input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks + 1)`, *optional*):
                The unconditional prompt rows, for speaker guidance.
            uncond_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The attention mask of `uncond_input_ids`.
            uncond_condition_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The condition ids of `uncond_input_ids`.
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration; `repetition_window` and `repetition_codebooks` can also be passed.
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and the model inputs the processor returns.

        Return:
            `torch.LongTensor` of shape `(batch_size, sequence_length + num_new_rows, num_codebooks + 1)`, or a
            [`~generation.GenerateDecoderOnlyOutput`] holding it when `return_dict_in_generate=True`.
        """
        guidance_scale = kwargs.get("guidance_scale")
        if guidance_scale is None:
            guidance_scale = (generation_config or self.generation_config).guidance_scale
        self._uses_guidance = guidance_scale is not None and guidance_scale != 1
        if self._uses_guidance:
            if uncond_input_ids is None:
                raise ValueError(
                    "speaker guidance needs the unconditional prompt: call the processor with speaker_guidance=True"
                )
            batch_size = input_ids.shape[0]
            input_ids = torch.cat([input_ids, uncond_input_ids.to(input_ids.device)])
            if (attention_mask := kwargs.get("attention_mask")) is not None:
                if uncond_attention_mask is None:
                    uncond_attention_mask = torch.ones_like(attention_mask)
                kwargs["attention_mask"] = torch.cat([attention_mask, uncond_attention_mask.to(attention_mask.device)])
            if (condition_ids := kwargs.get("condition_ids")) is not None:
                if uncond_condition_ids is None:
                    uncond_condition_ids = torch.full_like(condition_ids, -1)
                kwargs["condition_ids"] = torch.cat([condition_ids, uncond_condition_ids.to(condition_ids.device)])
            if (speaker_token_mask := kwargs.get("speaker_token_mask")) is not None:
                # the twins have no speaker row: the speaker embeddings stay conditional-only
                kwargs["speaker_token_mask"] = torch.cat([speaker_token_mask, torch.zeros_like(speaker_token_mask)])
            if input_ids.shape[0] != 2 * batch_size:
                raise ValueError("`uncond_input_ids` must hold one unconditional prompt per item.")

        return super().generate(input_ids=input_ids, generation_config=generation_config, **kwargs)
