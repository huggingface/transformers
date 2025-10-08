# coding=utf-8
# Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved.
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
"""HiggsAudioGenerationMixin."""

import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from ...generation import (
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from ...generation.logits_process import (
    HiggsAudioEOSDelayPatternLogitsProcessor,
    HiggsAudioSuppressTokensAtBeginLogitsProcessor,
)
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateNonBeamOutput
from ...utils import logging


logger = logging.get_logger(__name__)


@dataclass
class HiggsAudioGenerationOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of HiggsAudio generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.audio_num_codebooks, self.model.audio_codebook_size)`
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head or the audio head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.audio_num_codebooks, self.model.audio_codebook_size)`
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
            The generated discrete audio codes. These codes can be used to fill-in related locations of <|AUDIO_OUT|> at input sequences.
    """

    audio_sequences: Optional[list[torch.LongTensor]] = None


class HiggsAudioGenerationMixin(GenerationMixin):
    def _get_logits_processor(self, *args, **kwargs):
        if kwargs.get("logits_processor") is None:
            logits_processor = LogitsProcessorList()
        else:
            logits_processor = kwargs.get("logits_processor")

        logits_processor.append(
            HiggsAudioSuppressTokensAtBeginLogitsProcessor(
                begin_token_id=self.config.audio_out_bos_token_id,
                num_codebooks=self.config.num_codebooks,
                codebook_size=self.config.codebook_size,
                audio_stream_bos_id=self.config.audio_stream_bos_id,
                device=kwargs.get("device"),
            )
        )
        logits_processor.append(
            HiggsAudioEOSDelayPatternLogitsProcessor(
                delay_pattern=list(range(self.config.num_codebooks)),
                eos_token_id=self.config.audio_stream_eos_id,
                max_generation_len=kwargs.get("generation_config").max_length,
                device=kwargs.get("device"),
                audio_eos_token_id=self.config.audio_eos_token_id,
            )
        )

        return super()._get_logits_processor(*args, **kwargs)

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
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
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":
                # only raise warning if the user passed an explicit compile-config
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
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

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # ===========================
            next_token_logits = next_token_logits.reshape(-1, self.config.num_codebooks, self.config.codebook_size)
            ras_win_len = 7
            ras_win_max_num_repeat = 2
            if ras_win_len is not None:
                # check if there are repetitions over a window of tokens.
                # TODO: add detailed explanation
                # Find number of consecutive audio_out_token_idx for each batch element starting from the end
                flipped_ids = torch.flip(input_ids, dims=[1])
                is_audio = flipped_ids == self.config.audio_out_token_idx
                consecutive_counts = torch.argmax((~is_audio).int(), dim=1)
                consecutive_counts = torch.where(torch.all(is_audio, dim=1), input_ids.shape[1], consecutive_counts)
                # NOTE: consecutive_counts should be the same across batch idx
                cut_idx = min(consecutive_counts.max(), ras_win_len)
                if cut_idx > 0:
                    generated_audio_inputs_ids = model_kwargs["audio_input_ids"][:, -cut_idx:, :]
                    # Create mask to exclude counting when equality is 1024
                    equality_mask = generated_audio_inputs_ids[:, -ras_win_len:, :] == next_tokens[None, :]
                    not_1024_mask = generated_audio_inputs_ids[:, -ras_win_len:, :] != 1024
                    rep_num = (equality_mask & not_1024_mask).sum(dim=1)

                    row_indices = rep_num >= ras_win_max_num_repeat
                    resampled_next_tokens = (
                        next_token_logits[row_indices].softmax(dim=-1).multinomial(1, replacement=True).squeeze(1)
                    )
                    next_tokens[row_indices.flatten()] = resampled_next_tokens

            # ===========================

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # ============================
            if input_ids.shape[-1] == 820:
                next_tokens = torch.tensor([128013], device=input_ids.device)
            elif input_ids.shape[-1] == 821:
                next_tokens = torch.tensor([128016], device=input_ids.device)
                model_kwargs["audio_input_ids"] = torch.cat(
                    [model_kwargs["audio_input_ids"], torch.tensor([[[1024] * 8]], device=input_ids.device)], dim=1
                )
            else:
                next_tokens = next_tokens.reshape(batch_size, self.config.num_codebooks)

                # Check which batch elements have audio stream EOS tokens
                has_audio_stream_eos = (next_tokens == self.config.audio_stream_eos_id).any(dim=-1)
                # Check which batch elements have all audio stream EOS tokens
                has_all_audio_stream_eos = (next_tokens == self.config.audio_stream_eos_id).all(dim=-1)

                next_tokens = next_tokens[:, None, :]
                model_kwargs["audio_input_ids"] = torch.cat([model_kwargs["audio_input_ids"], next_tokens], dim=1)

                # For batches with audio stream EOS, set next token to audio_eos_token_id
                next_tokens_flat = (
                    torch.ones(batch_size, device=input_ids.device, dtype=torch.long) * self.config.audio_out_token_idx
                )
                next_tokens_flat[has_audio_stream_eos] = self.config.audio_eos_token_id
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
            return HiggsAudioGenerationOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                audio_sequences=model_kwargs.get("audio_input_ids"),
            )
        else:
            return input_ids
