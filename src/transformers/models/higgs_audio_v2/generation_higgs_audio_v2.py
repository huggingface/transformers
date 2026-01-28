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
from ...generation.logits_process import LogitsProcessor
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateNonBeamOutput
from ...utils import add_start_docstrings, logging


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
        scores = scores.reshape(-1, self.num_codebooks, self.codebook_size)
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
        sequences (`torch.LongTensor` of shape `(batch_size, audio_sequence_length, num_codebooks)`):
            The generated text sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.num_codebooks, self.model.codebook_size)`
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head or the audio head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.num_codebooks, self.model.codebook_size)`
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
        audio_sequences (`tuple(torch.LongTensor)` *optional*):
            The generated discrete audio codes.
    """

    audio_sequences: list[torch.LongTensor] | None = None


class HiggsAudioV2GenerationMixin(GenerationMixin):
    def _get_logits_processor(self, *args, **kwargs):
        if kwargs.get("logits_processor") is None:
            logits_processor = LogitsProcessorList()
        else:
            logits_processor = kwargs.get("logits_processor")

        logits_processor.append(
            HiggsAudioV2DelayPatternLogitsProcessor(
                delay_pattern=[el + 1 for el in range(self.config.num_codebooks)],
                audio_bos_token_id=self.config.audio_bos_token_id,
                audio_eos_token_id=self.config.audio_delay_token_id,
                audio_stream_bos_id=self.config.audio_stream_bos_id,
                audio_stream_eos_id=self.config.audio_stream_eos_id,
                num_codebooks=self.config.num_codebooks,
                codebook_size=self.config.codebook_size,
            )
        )
        return logits_processor

    def _prepare_generation_config(
        self, generation_config: GenerationConfig | None, **kwargs: Any
    ) -> tuple[GenerationConfig, dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        original_get_generation_mode = generation_config.get_generation_mode

        def patched_get_generation_mode(assistant_model=None):
            generation_mode = original_get_generation_mode(assistant_model)
            if generation_mode not in [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]:
                raise ValueError(
                    f"Generation mode {generation_mode} is not supported for HiggsAudioV2 model. Please set generation parameters to use greedy or sampling generation."
                )

            return generation_mode

        generation_config.get_generation_mode = patched_get_generation_mode

        return generation_config, model_kwargs

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
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        model_forward = (
            self.get_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )

        # Assisted generation completes the prefill stage in candidate generator so that
        # we don't have several `prefill` calls in one generation loop. Skip `_prefill` for assistants
        if not generation_config.is_assistant:
            outputs = self._prefill(input_ids, generation_config, model_kwargs)
            prefill_consumed = False
        else:
            model_kwargs = self._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)
            prefill_consumed = True

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if prefill_consumed:
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                with self._optimize_model_for_decode():
                    outputs = model_forward(**model_inputs, return_dict=True)
            prefill_consumed = True
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # ===========================
            # BELOW DIFFERENCES WITH GenerationMixin._sample()
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (
                        next_token_scores.reshape(batch_size, self.config.num_codebooks, self.config.codebook_size),
                    )
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            next_token_logits = next_token_logits.reshape(-1, self.config.num_codebooks, self.config.codebook_size)
            next_tokens = next_tokens.reshape(batch_size, self.config.num_codebooks)

            ras_win_len = generation_config.ras_win_len if hasattr(generation_config, "ras_win_len") else None
            ras_win_max_num_repeat = (
                generation_config.ras_win_max_num_repeat
                if hasattr(generation_config, "ras_win_max_num_repeat")
                else None
            )
            audio_input_ids = model_kwargs.get("audio_input_ids")
            if ras_win_len is not None and ras_win_max_num_repeat is not None and audio_input_ids is not None:
                # check if there are repetitions over a window of tokens.
                audio_inputs_ids_window = audio_input_ids[:, -ras_win_len:, :]
                repetition_mask = audio_inputs_ids_window == next_tokens.unsqueeze(1)

                # avoid counting the repetition of the audio stream EOS and BOS tokens
                not_excluded_mask = (audio_inputs_ids_window != self.config.audio_stream_bos_id) & (
                    audio_inputs_ids_window != self.config.audio_stream_eos_id
                )
                repetition_mask = repetition_mask & not_excluded_mask
                rep_num = repetition_mask.sum(dim=1)

                # if we saw repeated tokens in the most recent window of tokens, resample without temperature.
                replacement_mask = rep_num >= ras_win_max_num_repeat
                replacement_tokens = (
                    next_token_logits[replacement_mask].softmax(dim=-1).multinomial(1, replacement=True).view(-1)
                )
                next_tokens[replacement_mask] = replacement_tokens

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences[:, None] + self.config.audio_stream_eos_id * (
                    1 - unfinished_sequences[:, None]
                )

            has_audio_stream_eos = (next_tokens == self.config.audio_stream_eos_id).any(dim=-1)
            has_all_audio_stream_eos = (next_tokens == self.config.audio_stream_eos_id).all(dim=-1)
            next_tokens = next_tokens[:, None, :]

            if audio_input_ids is not None:
                model_kwargs["audio_input_ids"] = torch.cat([audio_input_ids, next_tokens], dim=1)
            else:
                model_kwargs["audio_input_ids"] = next_tokens

            next_audio_input_ids_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=next_tokens.device)
            next_audio_input_ids_mask[has_all_audio_stream_eos] = 0
            audio_input_ids_mask = model_kwargs.get("audio_input_ids_mask")
            if audio_input_ids_mask is not None:
                model_kwargs["audio_input_ids_mask"] = torch.cat(
                    [audio_input_ids_mask, next_audio_input_ids_mask], dim=1
                )
            else:
                model_kwargs["audio_input_ids_mask"] = next_audio_input_ids_mask

            # generation of a stream eos audio token will start delay pattern masking in the logits processor
            # for that, we need to set next text token to audio_eos_start_delay_token_id
            next_tokens_flat = input_ids.new_ones(batch_size) * self.config.audio_token_id
            next_tokens_flat[has_audio_stream_eos | (input_ids[:, -1] == self.config.audio_delay_token_id)] = (
                self.config.audio_delay_token_id
            )
            if self.config.eos_token_id is not None:
                next_tokens_flat[has_all_audio_stream_eos] = self.config.eos_token_id
            next_tokens = next_tokens_flat
            # ============================

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return HiggsAudioV2GenerationOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                audio_sequences=model_kwargs.get("audio_input_ids"),
            )
        else:
            return model_kwargs.get("audio_input_ids")
