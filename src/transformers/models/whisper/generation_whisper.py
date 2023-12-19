# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import math
import warnings
import zlib
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ...generation.logits_process import (
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    WhisperNoSpeechDetection,
    WhisperTimeStampLogitsProcessor,
)
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE


logger = logging.get_logger(__name__)


class WhisperGenerationMixin:

    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        condition_on_prev_tokens: Optional[bool] = None,
        no_speech_threshold: Optional[float] = None,
        temperature: Union[float, Tuple[float, ...]] = 0.0,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        num_segment_frames: Optional[int] = None,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: int = 0.02,
        return_dict_in_generate: Optional[bool] = None,
        **kwargs,
    ):
        """
        Transcribes or translates passed mel input features to a sequence of token ids.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            return_timestamps (`bool`, *optional*):
                Whether to return the timestamps with the text. This enables the `WhisperTimestampsLogitsProcessor`.
            task (`str`, *optional*):
                Task to use for generation, either "translate" or "transcribe". The `model.config.forced_decoder_ids`
                will be updated accordingly.
            language (`str`, *optional*):
                Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`. You can
                find all the possible language tokens in the `model.generation_config.lang_to_id` dictionary.
            is_multilingual (`bool`, *optional*):
                Whether or not the model is multilingual.
            prompt_ids (`torch.Tensor`, *optional*):
                Rank-1 tensor of token IDs created by passing text to [`~WhisperProcessor.get_prompt_ids`] that is
                provided as a prompt to each chunk. This can be used to provide or "prompt-engineer" a context for
                transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those words
                correctly. It cannot be used in conjunction with `decoder_start_token_id` as it overwrites this value.
            return_token_timestamps (`bool`, *optional*):
                Whether to return token-level timestamps with the text. This can be used with or without the
                `return_timestamps` option. To get word-level timestamps, use the tokenizer to group the tokens into
                words.
            return_segments (`bool`, *optional*, defaults to `False`):
                Whether to additionally return a list of all segments. Note that this option can only be enabled
                when doing long-form transcription.
            attention_mask (`torch.Tensor`, *optional*):
                `attention_mask` needs to be passed when doing long-form transcription using a batch size > 1.
            time_precision (`int`, *optional*, defaults to 0.02):
                The duration of output token in seconds. *E.g.* 0.02 means that a generated token on average accounts
                for 20 ms.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of just returning the generated tokens.
                Note that when doing long-form transcription, `return_dict_in_generate` can only be enabled when
                `return_segments` is set True. In this case the generation outputs of each segment is added to each
                segment.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor` or `Dict[str, Any]`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor` or a dict of segments when `return_segments=True`.

                If the passed input is > 30 seconds / > 3000 mel input features and `return_segments=True` then a dictionary of generated sequence ids, called `sequences` and a list of each generated segment is returned.

                else if the passed input is <= 30 seconds / >= 3000 mel input features, the possible [`~utils.ModelOutput`] types are:

                    - [`~generation.GreedySearchEncoderDecoderOutput`],
                    - [`~generation.SampleEncoderDecoderOutput`],
                    - [`~generation.BeamSearchEncoderDecoderOutput`],
                    - [`~generation.BeamSampleEncoderDecoderOutput`]

                else only the generated output sequence ids are returned.

        Example:

        - *Longform transcription*: To transcribe or translate audios longer than 30 seconds, process the audio files without truncation and pass all mel features at once to generate.

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset, Audio

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        >>> model.cuda()

        >>> # load audios > 30 seconds
        >>> ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
        >>> # resample to 16kHz
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        >>> # take first 8 audios and retrieve array
        >>> audio = ds[:8]["audio"]
        >>> audio = [x["array"] for x in audio]

        >>> # make sure to NOT truncate the input audio, to return the `attention_mask` and to pad to the longest audio
        >>> inputs = processor(audio, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=16_000)
        >>> inputs = inputs.to("cuda", torch.float32)

        >>> # transcribe audio to ids
        >>> generated_ids = model.generate(**inputs)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        >>> transcription[0]
        ' Folks, if you watch the show, you know, I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories developing the central headline pawns, definitely maneuvering an oso topical night to F6, fainting a classic Sicilian, nade door variation on the news, all the while seeing eight moves deep and patiently marshalling the latest press releases into a fisher's shows in Lip Nitsky attack that culminates in the elegant lethal slow-played, all-passant checkmate that is my nightly monologue. But sometimes, sometimes, folks, I. CHEERING AND APPLAUSE Sometimes I startle away, cubside down in the monkey bars of a condemned playground on a super fun site. Get all hept up on goofballs. Rummage that were discarded tag bag of defective toys. Yank out a fist bowl of disembodied doll limbs, toss them on a stained kid's place mat from a defunct dennies. set up a table inside a rusty cargo container down by the Wharf and challenged toothless drifters to the godless bughouse blitz of tournament that is my segment. Meanwhile!'
        ```

        - *Shortform transcription*: If passed mel input features are < 30 seconds, the whole audio will be transcribed with a single call to generate.

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```

        """
        # 0. deprecate old inputs
        if "inputs" in kwargs:
            input_features = kwargs.pop("inputs")
            warnings.warn(
                "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
                FutureWarning,
            )
        # 1. copy generation config
        if generation_config is None:
            generation_config = copy.deepcopy(self.generation_config)
        else:
            generation_config = copy.deepcopy(generation_config)

        # 2. set global generate variables
        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        num_segment_frames = input_stride * self.config.max_source_positions
        total_input_frames = self._retrieve_total_input_frames(input_features=input_features, input_stride=input_stride, kwargs=kwargs)
        is_shortform = total_input_frames <= num_segment_frames

        # 3. Make sure generation config is correctly set
        # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        self._set_return_outputs(return_dict_in_generate=return_dict_in_generate, return_token_timestamps=return_token_timestamps, is_shortform=is_shortform, logprob_threshold=logprob_threshold, generation_config=generation_config)
        self._set_return_timestamps(return_timestamps=return_timestamps, is_shortform=is_shortform, generation_config=generation_config)
        self._set_language_and_task(language=language, task=task, is_multilingual=is_multilingual, generation_config=generation_config)
        # pass self.config for backward compatibility
        self._set_forced_decoder_ids(task=task, language=language, prompt_ids=prompt_ids, generation_config=generation_config, config=self.config, kwargs=kwargs)
        self._set_num_frames(return_token_timestamps=return_token_timestamps, generation_config=generation_config, kwargs=kwargs)

        # 4. Retrieve logits processors
        # => TODO(Patrick): The way `num_start_tokens` is retrieved here is too brittle. Need a better approach
        num_start_tokens = len(generation_config.forced_decoder_ids) if generation_config.forced_decoder_ids is not None else 1
        logits_processor = self._retrieve_logit_processors(generation_config=generation_config, logits_processor=logits_processor, no_speech_threshold=no_speech_threshold, num_start_tokens=num_start_tokens, is_shortform=is_shortform)

        # 5. If we're in shortform mode, simple generate the whole input at once and return the output
        if is_shortform:
            outputs = super().generate(
                input_features,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )

            if generation_config.return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                outputs["token_timestamps"] = self._extract_token_timestamps(
                    outputs, generation_config.alignment_heads, num_frames=generation_config.num_frames
                )

            return outputs

        # 6. Else we're in longform mode which is more complex.
        # We need to chunk the audio input depending on when the model generates timestamp tokens

        # 6.1 Set and retrieve global longform generation variables
        self._set_condition_on_prev_tokens(condition_on_prev_tokens=condition_on_prev_tokens, generation_config=generation_config)

        timestamp_begin = generation_config.no_timestamps_token_id + 1
        temperatures = [temperature] if not isinstance(temperature, (list, tuple)) else temperature
        temperature = temperatures[0]
        batch_size = input_features.shape[0]

        max_frames, seek = self._retrieve_max_frames_and_seek(batch_size=batch_size, attention_mask=attention_mask, total_input_frames=total_input_frames)
        init_tokens = self._retrieve_init_tokens_from_forced_decoder_ids(generation_config=generation_config)

        # 6.2 Preppare running variables, list for generation
        cur_bsz = batch_size
        current_segments = [[] for _ in range(batch_size)]
        batch_idx_map = list(range(batch_size))
        do_condition_on_prev_tokens = [condition_on_prev_tokens for _ in range(batch_size)]

        # 6.2 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            # 6.3 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
            # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
            # to know which original audio is being decoded
            # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
            input_features, cur_bsz, batch_idx_map = self._maybe_reduce_batch(input_features=input_features, seek=seek, max_frames=max_frames, cur_bsz=cur_bsz, batch_idx_map=batch_idx_map)
            time_offset = seek * time_precision / input_stride
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

            # 6.4 cut out next 30s segment from input features
            segment_input = self._get_input_segment(input_features=input_features, seek=seek, seek_num_frames=seek_num_frames, num_segment_frames=num_segment_frames, cur_bsz=cur_bsz, batch_idx_map=batch_idx_map)

            # 6.5 prepare decoder input ids
            # TODO(Patrick) - clean up prev_start_of_text
            suppress_tokens = self._get_attr_from_logit_processors(logits_processor, SuppressTokensLogitsProcessor, "suppress_tokens")
            prev_start_of_text = suppress_tokens[-2] if suppress_tokens is not None else None
            decoder_input_ids, kwargs = self._prepare_decoder_input_ids(cur_bsz=cur_bsz, init_tokens=init_tokens, current_segments=current_segments, batch_idx_map=batch_idx_map, do_condition_on_prev_tokens=do_condition_on_prev_tokens, generation_config=generation_config, config=self.config, device=segment_input.device, kwargs=kwargs, prev_start_of_text=prev_start_of_text)

            # 6.6 set max new tokens or max length
            kwargs = self._set_max_new_tokens_and_length(config=self.config, decoder_input_ids=decoder_input_ids, generation_config=generation_config, kwargs=kwargs)

            # 6.7 Set current `begin_index` for all logit processors
            for proc in logits_processor:
                if hasattr(proc, "set_begin_index"):
                    proc.set_begin_index(decoder_input_ids.shape[-1])

            print("hf  in tokens", decoder_input_ids[0].tolist())
            # 6.8 Run generate with fallback
            seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens = self.generate_with_fallback(segment_input=segment_input, decoder_input_ids=decoder_input_ids, cur_bsz=cur_bsz, batch_idx_map=batch_idx_map, seek=seek, num_segment_frames=num_segment_frames, max_frames=max_frames, temperatures=temperatures, generation_config=generation_config, logits_processor=logits_processor, stopping_criteria=stopping_criteria, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, synced_gpus=synced_gpus, return_token_timestamps=return_token_timestamps, compression_ratio_threshold=compression_ratio_threshold, logprob_threshold=logprob_threshold, no_speech_threshold=no_speech_threshold, do_condition_on_prev_tokens=do_condition_on_prev_tokens, condition_on_prev_tokens=condition_on_prev_tokens, kwargs=kwargs)

            # 6.9 In every generated sequence, split by timestamp tokens and extract segments
            for i, seek_sequence in enumerate(seek_sequences):
                prev_i = batch_idx_map[i]

                if should_skip[i]:
                    seek[prev_i] += seek_num_frames[prev_i]
                    print("Skipped!")
                    continue

                # TODO(Patrick: delete cut type)
                segments, segment_offset, cut_type = self._retrieve_segment(
                    seek_sequence=seek_sequence,
                    seek_outputs=seek_outputs,
                    time_offset=time_offset,
                    timestamp_begin=timestamp_begin,
                    seek_num_frames=seek_num_frames,
                    time_precision=time_precision,
                    input_stride=input_stride,
                    prev_idx=prev_i,
                    idx=i,
                )

                current_segments[prev_i] += segments
                seek[prev_i] += segment_offset

        # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
        # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
        sequences = self._pad_to_max_length(current_segments, generation_config.pad_token_id, padding="right")

        # 8. If we return all segments, the predicted output sequences are put under `"sequences"`.
        if return_segments:
            return {"sequences": sequences, "segments": current_segments}

        return sequences

    def generate_with_fallback(self, segment_input, decoder_input_ids, cur_bsz, batch_idx_map, seek, num_segment_frames, max_frames, temperatures, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, return_token_timestamps, compression_ratio_threshold, logprob_threshold, no_speech_threshold, do_condition_on_prev_tokens, condition_on_prev_tokens, kwargs):
        # 6.6 Batch generate current chunk
        seek_sequence_list = [None for _ in range(cur_bsz)]
        seek_outputs_list = [None for _ in range(cur_bsz)]
        needs_fallback = [False for _ in range(cur_bsz)]
        should_skip = [False for _ in range(cur_bsz)]
        fallback_index_map = list(range(cur_bsz))

        for fallback_idx, temperature in enumerate(temperatures):
            generation_config.do_sample = temperature > 0.0
            generation_config.temperature = temperature
            generation_config.num_beams = kwargs.pop("num_beams", 1) if not generation_config.do_sample else 1

            seek_outputs = super().generate(
                segment_input,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                decoder_input_ids=decoder_input_ids,
                **kwargs,
            )

            if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                num_frames = getattr(generation_config, "num_frames", None)
                seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                    seek_outputs, generation_config.alignment_heads, num_frames=num_frames
                )

            if generation_config.return_dict_in_generate:
                def split_by_batch_index(values, key, batch_idx):
                    if key == "scores":
                        return [v[batch_idx].cpu() for v in values]
                    if key == "past_key_values":
                        # we don't save `past_key_values` as this is too costly
                        return None
                    return values[batch_idx].cpu()

                sequence_tokens = seek_outputs["sequences"]
                seek_outputs = [{k: split_by_batch_index(v, k, i) for k, v in seek_outputs.items()} for i in range(sequence_tokens.shape[0])]
            else:
                sequence_tokens = seek_outputs

            # remove all previously passed decoder input ids
            seek_sequences = sequence_tokens[:, decoder_input_ids.shape[-1]:]

            new_fallback_index_map = []
            new_segment_input = []
            new_decoder_input_ids = []
            new_decoder_attention_mask = []

            # 6.7 Loop over each decoded audio individually as each decoding can be of a different length
            for i, seek_sequence in enumerate(seek_sequences):
                # make sure we cut a predicted EOS token if we are not finished with the generation yet
                prev_i = batch_idx_map[fallback_index_map[i]]
                is_not_final = (seek[prev_i] + num_segment_frames) < max_frames[prev_i]

                # remove eos token id
                if is_not_final and seek_sequence[-1] == generation_config.eos_token_id:
                    seek_sequence = seek_sequence[:-1]

                print("hf  out tokens", seek_sequence.tolist())

                # remove all padding tokens
                if seek_sequence[-1] == generation_config.pad_token_id:
                    num_paddings = (seek_sequence == generation_config.pad_token_id).sum()
                    seek_sequence = seek_sequence[:-num_paddings]

                if compression_ratio_threshold is not None:
                    compression_ratio = self._retrieve_compression_ratio(seek_sequence, self.config.vocab_size)

                    if compression_ratio > compression_ratio_threshold:
                        print("fallback compression")
                        print("current temp", temperature)
                        needs_fallback[i] = True

                if logprob_threshold is not None:
                    if "sequences_scores" in seek_outputs[0]:
                        logprobs = [s["sequences_scores"] for s in seek_outputs][i]
                    else:
                        scores = seek_outputs[i]["scores"]
                        logprobs = self._retrieve_avg_logprobs(scores, seek_sequence, self.config.eos_token_id, temperature)

                    # TODO(PVP) only works for batch size = 1 currently
                    if logprobs < logprob_threshold:
                        print("fallback logprobs", logprobs)
                        print("current temp", temperature)
                        needs_fallback[i] = True

                if no_speech_threshold is not None:
                    # TODO(PVP) only works for batch size = 1 currently
                    # Need to do before all other logit processors
                    no_speech_prob = self._get_attr_from_logit_processors(logits_processor, WhisperNoSpeechDetection, "no_speech_prob")
                    if no_speech_prob[i] > no_speech_threshold and logprobs < logprob_threshold:
                        print("Skip because of VAD")
                        needs_fallback[i] = False
                        should_skip[i] = True

                seek_sequence_list[fallback_index_map[i]] = seek_sequence
                seek_outputs_list[fallback_index_map[i]] = seek_outputs[i]
                do_condition_on_prev_tokens[fallback_index_map[i]] = condition_on_prev_tokens and temperature < 0.5

                if needs_fallback[i]:
                    new_fallback_index_map.append(fallback_index_map[i])
                    new_segment_input.append(segment_input[i])
                    new_decoder_input_ids.append(decoder_input_ids[i])
                    if "decoder_attention_mask" in kwargs:
                        new_decoder_attention_mask.append(kwargs['decoder_attention_mask'][i])

            fallback_index_map = new_fallback_index_map

            # if no sequence needs to be run with temperature fallback, we're finished
            if len(fallback_index_map) == 0 or fallback_idx == len(temperatures) - 1:
                seek_sequences = seek_sequence_list
                seek_outputs = seek_outputs_list
                break

            # if we're still in the loop, make sure that decoder_input_ids and segment inputs are tensors
            decoder_input_ids = torch.stack(new_decoder_input_ids)
            segment_input = torch.stack(new_segment_input)
            if "decoder_attention_mask" in kwargs:
                kwargs["decoder_attention_mask"] = torch.stack(new_decoder_attention_mask)

        return seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens


    @staticmethod
    def _get_attr_from_logit_processors(logits_processor, logit_processor_class, attribute_name):
        logit_processor = next((cls for cls in logits_processor if isinstance(cls, logit_processor_class)), None)
        if logit_processor:
            return getattr(logit_processor, attribute_name, None)
        return None

    @staticmethod
    def _retrieve_total_input_frames(input_features, input_stride, kwargs):
        if input_features is not None:
            return input_features.shape[-1]

        if "encoder_outputs" in kwargs:
            encoder_outputs_shape = (
                kwargs["encoder_outputs"][0].shape
                if isinstance(kwargs["encoder_outputs"], BaseModelOutput)
                else kwargs["encoder_outputs"].shape
            )
            return encoder_outputs_shape[1] * input_stride

        raise ValueError("Make sure to provide either `input_features` or `encoder_outputs` to `generate`.")

    @staticmethod
    def _set_return_outputs(return_dict_in_generate, return_token_timestamps, is_shortform, logprob_threshold, generation_config):
        if return_dict_in_generate is None:
            return_dict_in_generate = generation_config.return_dict_in_generate

        generation_config.return_token_timestamps = return_token_timestamps
        if return_token_timestamps:
            return_dict_in_generate = True
            generation_config.output_attentions = True

        if not is_shortform and logprob_threshold is not None:
            return_dict_in_generate = True
            generation_config.output_scores = True

        generation_config.return_dict_in_generate = return_dict_in_generate

    @staticmethod
    def _set_return_timestamps(return_timestamps, is_shortform, generation_config):
        if return_timestamps is True:
            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You are trying to return timestamps, but the generation config is not properly set. "
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                )
            generation_config.return_timestamps = True
        elif not is_shortform:
            if return_timestamps is False:
                raise ValueError(
                    "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
                    "requires the model to predict timestamp tokens. Please either pass `return_timestamps=True` or make sure to pass no more than 3000 mel input features."
                )

            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
                    "requires the generation config to have `no_timestamps_token_id` correctly. "
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                    "or make sure to pass no more than 3000 mel input features."
                )

            logger.info("Setting `return_timestamps=True` for long-form generation.")
            generation_config.return_timestamps = True
        else:
            generation_config.return_timestamps = False

    @staticmethod
    def _set_language_and_task(language, task, is_multilingual, generation_config):
        if is_multilingual is not None:
            if not hasattr(generation_config, "is_multilingual"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `is_multilingual` argument "
                    "to `generate`. Please update the generation config as per the instructions "
                    "https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.is_multilingual = is_multilingual

        if hasattr(generation_config, "is_multilingual") and not generation_config.is_multilingual:
            if task is not None or language is not None:
                raise ValueError(
                    "Cannot specify `task` or `language` for an English-only model. If the model is intended to be "
                    "multilingual, pass `is_multilingual=True` to generate, or update the generation config."
                )

        if language is not None:
            if not hasattr(generation_config, "lang_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `language` argument "
                    "to `generate`. Either set the language using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            language = language.lower()
            generation_config.language = language

        if task is not None:
            if not hasattr(generation_config, "task_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `task` argument "
                    "to `generate`. Either set the task using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            generation_config.task = task

    @staticmethod
    def _set_forced_decoder_ids(task, language, prompt_ids, generation_config, config, kwargs):
        forced_decoder_ids = None
        # Legacy code for backward compatibility
        if hasattr(config, "forced_decoder_ids") and config.forced_decoder_ids is not None:
            forced_decoder_ids = config.forced_decoder_ids
        elif (
            hasattr(generation_config, "forced_decoder_ids")
            and generation_config.forced_decoder_ids is not None
        ):
            forced_decoder_ids = generation_config.forced_decoder_ids
        else:
            forced_decoder_ids = kwargs.pop("forced_decoder_ids", None)

        if task is not None or language is not None or (forced_decoder_ids is None and prompt_ids is not None):
            forced_decoder_ids = []
            if hasattr(generation_config, "language"):
                if generation_config.language in generation_config.lang_to_id.keys():
                    language_token = generation_config.language
                elif generation_config.language in TO_LANGUAGE_CODE.keys():
                    language_token = f"<|{TO_LANGUAGE_CODE[generation_config.language]}|>"
                elif generation_config.language in TO_LANGUAGE_CODE.values():
                    language_token = f"<|{generation_config.language}|>"
                else:
                    is_language_code = len(generation_config.language) == 2
                    raise ValueError(
                        f"Unsupported language: {generation_config.language}. Language should be one of:"
                        f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                    )
                if language_token not in generation_config.lang_to_id:
                    raise ValueError(
                        f"{language_token} is not supported by this specific model as it is not in the `generation_config.lang_to_id`."
                        "(You should just add it to the generation config)"
                    )
                forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
            else:
                forced_decoder_ids.append((1, None))  # automatically detect the language

            if hasattr(generation_config, "task"):
                if generation_config.task in TASK_IDS:
                    forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
                else:
                    raise ValueError(
                        f"The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`"
                    )
            elif hasattr(generation_config, "task_to_id"):
                forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))  # defaults to transcribe
            if hasattr(generation_config, "no_timestamps_token_id") and not generation_config.return_timestamps:
                idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

        if forced_decoder_ids is not None:
            generation_config.forced_decoder_ids = forced_decoder_ids

        if prompt_ids is not None:
            if kwargs.get("decoder_start_token_id") is not None:
                raise ValueError(
                    "When specifying `prompt_ids`, you cannot also specify `decoder_start_token_id` as it gets overwritten."
                )
            prompt_ids = prompt_ids.tolist()
            decoder_start_token_id, *text_prompt_ids = prompt_ids
            # Slicing the text prompt ids in a manner consistent with the OpenAI implementation
            # to accomodate context space for the prefix (see https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/decoding.py#L599)
            text_prompt_ids = text_prompt_ids[-config.max_target_positions // 2 - 1 :]
            # Set the decoder_start_token_id to <|startofprev|>
            kwargs.update({"decoder_start_token_id": decoder_start_token_id})

            # If the user passes `max_new_tokens`, increase its number to account for the prompt
            if kwargs.get("max_new_tokens", None) is not None:
                kwargs["max_new_tokens"] += len(text_prompt_ids)
                if kwargs["max_new_tokens"] >= config.max_target_positions:
                    raise ValueError(
                        f"The length of the sliced `prompt_ids` is {len(text_prompt_ids)}, and the `max_new_tokens` "
                        f"{kwargs['max_new_tokens'] - len(text_prompt_ids)}. Thus, the combined length of the sliced "
                        f"`prompt_ids` and `max_new_tokens` is: {kwargs['max_new_tokens']}. This exceeds the "
                        f"`max_target_positions` of the Whisper model: {config.max_target_positions}. "
                        "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                        f"so that their combined length is less that {config.max_target_positions}."
                    )

            # Reformat the forced_decoder_ids to incorporate the prompt
            non_prompt_forced_decoder_ids = (
                kwargs.pop("forced_decoder_ids", None) or generation_config.forced_decoder_ids
            )
            forced_decoder_ids = [
                *text_prompt_ids,
                generation_config.decoder_start_token_id,
                *[token for _, token in non_prompt_forced_decoder_ids],
            ]
            forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_decoder_ids)]
            generation_config.forced_decoder_ids = forced_decoder_ids

    @staticmethod
    def _set_num_frames(return_token_timestamps, generation_config, kwargs):
        if return_token_timestamps:
            if getattr(generation_config, "task", None) == "translate":
                logger.warning("Token-level timestamps may not be reliable for task 'translate'.")
            if not hasattr(generation_config, "alignment_heads"):
                raise ValueError(
                    "Model generation config has no `alignment_heads`, token-level timestamps not available. "
                    "See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config."
                )

            generation_config.num_frames = kwargs.pop("num_frames", None)

    @staticmethod
    def _set_condition_on_prev_tokens(condition_on_prev_tokens, generation_config):
        condition_on_prev_tokens = (
            condition_on_prev_tokens
            if condition_on_prev_tokens is not None
            else getattr(generation_config, "condition_on_prev_tokens", False)
        )
        generation_config.condition_on_prev_tokens = condition_on_prev_tokens

    @staticmethod
    def _retrieve_max_frames_and_seek(batch_size, attention_mask, total_input_frames):
        if batch_size > 1 and attention_mask is None:
            raise ValueError(
                "When doing long-form audio transcription, make sure to pass an `attention_mask`. You can retrieve the `attention_mask` by doing `processor(audio, ..., return_attention_mask=True)` "
            )
        elif batch_size > 1:
            max_frames = attention_mask.sum(-1).cpu().to(torch.long)
            seek = torch.zeros((batch_size,), dtype=torch.long)
        else:
            max_frames = torch.ones((1,), dtype=torch.long) * total_input_frames
            seek = torch.zeros((1,), dtype=torch.long)

        return max_frames, seek

    @staticmethod
    def _retrieve_init_tokens_from_forced_decoder_ids(generation_config):
        init_tokens = [generation_config.decoder_start_token_id]
        forced_decoder_ids = generation_config.forced_decoder_ids
        if forced_decoder_ids is not None and forced_decoder_ids[0][0] == 1:
            i = 1
            while len(forced_decoder_ids) > 0 and forced_decoder_ids[0][0] == i:
                init_tokens += [forced_decoder_ids[0][1]]
                forced_decoder_ids = forced_decoder_ids[1:]
                i += 1

            forced_decoder_ids = forced_decoder_ids if len(forced_decoder_ids) > 0 else None
            generation_config.forced_decoder_ids = forced_decoder_ids

        return init_tokens

    @staticmethod
    def _retrieve_logit_processors(generation_config, logits_processor, no_speech_threshold, num_start_tokens, is_shortform):
        begin_index = 1
        if generation_config.return_timestamps is True:
            forced_decoder_ids = generation_config.forced_decoder_ids
            last_forced_decoder_ids = forced_decoder_ids[-1][-1] if forced_decoder_ids is not None else None
            if last_forced_decoder_ids == generation_config.no_timestamps_token_id:
                # remove no_timestamp to be forcefully generated if we want to return timestamps
                # this is also important to make sure `WhisperTimeStampLogitsProcessor` functions correctly
                forced_decoder_ids = forced_decoder_ids[:-1] if len(forced_decoder_ids) > 1 else None
                # Make sure that if list is empty we set it to None
                generation_config.forced_decoder_ids = forced_decoder_ids

            begin_index = begin_index + len(forced_decoder_ids) if forced_decoder_ids is not None else begin_index

            timestamp_processor = WhisperTimeStampLogitsProcessor(generation_config, begin_index=begin_index)
            logits_processor = (
                [timestamp_processor] if logits_processor is None else [timestamp_processor] + logits_processor
            )

        if generation_config.suppress_tokens is not None:
            suppress_tokens_processor = SuppressTokensLogitsProcessor(generation_config.suppress_tokens)
            logits_processor = (
                [suppress_tokens_processor] if logits_processor is None else [suppress_tokens_processor] + logits_processor
            )
            generation_config.suppress_tokens = None

        if generation_config.begin_suppress_tokens is not None:
            begin_suppress_processor = SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index=begin_index)
            logits_processor = (
                [begin_suppress_processor] if logits_processor is None else [begin_suppress_processor] + logits_processor
            )
            generation_config.begin_suppress_tokens = None

        if no_speech_threshold is not None and not is_shortform:
            no_speech_detector = WhisperNoSpeechDetection(no_speech_token=generation_config.no_timestamps_token_id - 1, begin_index=begin_index, begin_index_offset=num_start_tokens)
            logits_processor = (
                [no_speech_detector] if logits_processor is None else [no_speech_detector] + logits_processor
            )

        return logits_processor

    @staticmethod
    def _maybe_reduce_batch(input_features, seek, max_frames, cur_bsz, batch_idx_map):
        prev_bsz = cur_bsz
        new_batch_idx_map = []
        for i in range(prev_bsz):
            prev_i = batch_idx_map[i]
            if seek[prev_i] >= max_frames[prev_i]:
                cut_index = i + (cur_bsz - prev_bsz)
                cur_bsz -= 1
                input_features = torch.cat([input_features[:cut_index], input_features[cut_index + 1 :]], dim=0)
            else:
                # cut out index that goes away
                new_batch_idx_map.append(prev_i)

        return input_features, cur_bsz, new_batch_idx_map

    @staticmethod
    def _get_input_segment(input_features, seek, seek_num_frames, num_segment_frames, cur_bsz, batch_idx_map):
        segment_input = []
        for i in range(cur_bsz):
            prev_i = batch_idx_map[i]
            segment_input_slice = input_features[
                i : i + 1, :, seek[prev_i] : seek[prev_i] + seek_num_frames[prev_i]
            ]

            if segment_input_slice.shape[-1] < num_segment_frames:
                # pad to 3000 if necessary
                segment_input_slice = F.pad(
                    segment_input_slice, pad=(0, num_segment_frames - segment_input_slice.shape[-1])
                )

            segment_input.append(segment_input_slice)

        segment_input = torch.cat(segment_input, dim=0)

        return segment_input

    # TODO(Patrick) - remove prev_start_of_text
    @staticmethod
    def _prepare_decoder_input_ids(cur_bsz, init_tokens, current_segments, batch_idx_map, do_condition_on_prev_tokens, generation_config, config, device, kwargs, prev_start_of_text):
        cut_off_length = config.max_target_positions // 2 - 1

        one_tensor = torch.ones((cur_bsz, 1), device=device, dtype=torch.long)
        decoder_input_ids = torch.cat([t * one_tensor for t in init_tokens], dim=-1)

        # if condition_on_prev_tokens and len(current_segments[0]) > 0 and temperature < 0.5:
        if any(do_condition_on_prev_tokens) and len(current_segments[0]) > 0:
            # according to https://github.com/openai/whisper/blob/e58f28804528831904c3b6f2c0e473f346223433/whisper/decoding.py#L609
            active_segments = [current_segments[i] if do_condition_on_prev_tokens[i] else None for i in batch_idx_map]
            prev_start_of_text = getattr(generation_config, "prev_bos_token_id", None) or prev_start_of_text

            bos_token_tensor = prev_start_of_text * one_tensor[0]
            prev_tokens = WhisperGenerationMixin._pad_to_max_length(
                active_segments, generation_config.pad_token_id, padding="left", bos_token_tensor=bos_token_tensor, cut_off_length=cut_off_length
            )
            decoder_input_ids = torch.cat([prev_tokens, decoder_input_ids], dim=-1)

            kwargs["decoder_attention_mask"] = (decoder_input_ids != generation_config.pad_token_id)

        return decoder_input_ids, kwargs

    @staticmethod
    def _set_max_new_tokens_and_length(config, decoder_input_ids, generation_config, kwargs):
        num_initial_tokens = min(config.max_target_positions // 2 - 1, decoder_input_ids.shape[-1] - 1)

        passed_max_length = kwargs.pop("max_length", None)
        passed_max_new_tokens = kwargs.pop("max_new_tokens", None)
        max_length_config = getattr(generation_config, "max_length", None)
        max_new_tokens_config = getattr(generation_config, "max_new_tokens", None)

        max_new_tokens = None
        max_length = None

        # Make sure we don't get larger than `max_length`
        if passed_max_length is not None and passed_max_new_tokens is None:
            max_length = min(
                passed_max_length + num_initial_tokens, config.max_target_positions
            )
            logger.info(
                f"Increase max_length from {passed_max_length} to {max_length} since input is conditioned on previous segment."
            )
        elif max_length_config is not None and passed_max_new_tokens is None and max_new_tokens_config is None:
            max_length = min(
                generation_config.max_length + num_initial_tokens, config.max_target_positions
            )
            logger.info(
                f"Increase max_length from {max_length_config} to {max_length} since input is conditioned on previous segment."
            )
        elif (
            passed_max_new_tokens is not None
            and passed_max_new_tokens + decoder_input_ids.shape[-1] > config.max_target_positions
        ):
            max_new_tokens = config.max_target_positions - decoder_input_ids.shape[-1]
        elif (
            passed_max_new_tokens is None
            and max_new_tokens_config is not None
            and max_new_tokens_config + decoder_input_ids.shape[-1] > config.max_target_positions
        ):
            max_new_tokens = config.max_target_positions - decoder_input_ids.shape[-1]

        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens

        if max_length is not None:
            kwargs["max_length"] = max_length

        return kwargs

    @staticmethod
    def _pad_to_max_length(current_segments, pad_token_id, padding="right", bos_token_tensor=None, cut_off_length=None):
        max_total_length = 0
        sequences = []
        if padding not in ["right", "left"]:
            raise ValueError(f"`padding` must be either 'right' or 'left', not {padding}")

        for current_segment_list in current_segments:
            if current_segment_list is not None:
                sequence = torch.cat([d["tokens"] for d in current_segment_list], dim=-1)

                if cut_off_length is not None:
                    sequence = sequence[-cut_off_length:]

                if bos_token_tensor is not None:
                    sequence = torch.cat([bos_token_tensor, sequence])

                sequences.append(sequence)
                max_total_length = max(max_total_length, len(sequences[-1]))
            else:
                sequences.append(bos_token_tensor)

        for i in range(len(current_segments)):
            pad_length = max_total_length - len(sequences[i])
            pad = (0, pad_length) if padding == "right" else (pad_length, 0)
            sequences[i] = F.pad(sequences[i], pad=pad, value=pad_token_id)

        sequences = torch.stack(sequences, dim=0)
        return sequences

    @staticmethod
    def _retrieve_compression_ratio(tokens, vocab_size):
        """ Compute byte length of zlib compressed token bytes vs. byte length of raw token bytes """
        length = int(math.log2(vocab_size) / 8) + 1
        token_bytes = b''.join([t.to_bytes(length, 'little') for t in tokens.tolist()])
        compression_ratio = len(token_bytes) / len(zlib.compress(token_bytes))

        return compression_ratio

    @staticmethod
    def _retrieve_avg_logprobs(scores, tokens, eos_token_id, temperature):
        rescale_temperature = temperature if temperature > 0.0 else 1
        scores = torch.stack(scores).to(tokens.device)

        # TODO(Patrick) - only leave scores = scores[:tokens.shape[0]] part
        if scores.shape[0] > tokens.shape[0]:
            scores = scores[:tokens.shape[0]]
        else:
            tokens = tokens[-scores.shape[0]:]

        logprobs = F.log_softmax((scores * rescale_temperature).float(), dim=-1).to(scores.dtype)

        # retrieve logprob of selected tokens and sum
        sum_logprobs = sum((logprobs[i][tokens[i]] * (tokens[i] != eos_token_id)) for i in range(logprobs.shape[0]))
        length = (tokens != eos_token_id).sum(-1)

        avg_logprobs = sum_logprobs / (length + 1)
        return avg_logprobs

    @staticmethod
    def _retrieve_segment(
        seek_sequence,
        seek_outputs,
        time_offset,
        timestamp_begin,
        seek_num_frames,
        time_precision,
        input_stride,
        prev_idx,
        idx,
    ):
        # find the predicted "end of segment" predictions of Whisper
        # "end of segment" predictions occur whenever Whisper predicts a timestamp token
        timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
        timestamp_segment_indices = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
        timestamp_segment_indices.add_(1)

        # If whisper predicted a "end of segment" via a timestep token, let's go ever each
        # "end of segment" prediction and slice the decoding into segments accordingly
        if len(timestamp_segment_indices) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = timestamp_segment_indices.tolist()
            segments = []
            if single_timestamp_ending:
                slices.append(len(seek_sequence))

            last_slice = 0
            # Add each segment to list of all segments
            for current_slice in slices:
                sliced_tokens = seek_sequence[last_slice : current_slice]
                start_timestamp_pos = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_pos = sliced_tokens[-1].item() - timestamp_begin
                segments.append(
                    {
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + end_timestamp_pos * time_precision,
                        "tokens": sliced_tokens,
                        "result": seek_outputs[idx],
                    }
                )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                segment_offset = seek_num_frames[prev_idx]
                cut_type = "single ending"
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                # here we throw away all predictions after the last predicted "end of segment"
                # since we are cutting right in the middle of an audio
                last_timestamp_pos = seek_sequence[last_slice - 1].item() - timestamp_begin
                segment_offset = last_timestamp_pos * input_stride
                cut_type = "cut"
        else:
            # If whisper does not predict any "end of segment" token, then
            # the whole decoding is considered a segment and we add it to the list of segments
            timestamps = seek_sequence[timestamp_tokens.nonzero().flatten()]
            last_timestamp_pos = seek_num_frames[prev_idx]
            if timestamps.numel() > 0 and timestamps[-1].item() != timestamp_begin:
                # no consecutive timestamps but it has a timestamp; use the last one.
                last_timestamp_pos = timestamps[-1].item() - timestamp_begin

            segments = [
                {
                    "start": time_offset[prev_idx],
                    "end": time_offset[prev_idx] + last_timestamp_pos * time_precision,
                    "tokens": seek_sequence,
                    "result": seek_outputs[idx],
                }
            ]
            segment_offset = seek_num_frames[prev_idx]
            cut_type = "all"

        return segments, segment_offset, cut_type

