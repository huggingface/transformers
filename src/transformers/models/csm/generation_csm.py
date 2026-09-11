# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from ...generation import (
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    GenerationMode,
    GenerationState,
)
from ...generation.logits_process import LogitsProcessorList
from ...generation.stopping_criteria import MaxLengthCriteria, StoppingCriteria, StoppingCriteriaList
from ...utils import ModelOutput, logging


if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer


logger = logging.get_logger(__name__)


@dataclass
class CsmGenerateOutput(GenerateDecoderOnlyOutput):
    """
    Outputs of CsmForConditionalGeneration.generate.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, num_frames, num_codebooks)`):
            The generated audio frames. `num_frames` is either equal to `max_length` or shorter if all batches
            finished early on an end-of-audio frame. A codebook prompt is returned with the generated frames; a text
            prompt is not.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
        audio (`list(torch.FloatTensor)` of length `batch_size`, *optional*, returned when `output_audio=True`):
            The generated audio, one waveform per batch item, each cut at its first end-of-audio frame.
    """

    audio: list[torch.Tensor] | None = None


class CsmEosFrameCriteria(StoppingCriteria):
    """
    Stops a sequence once the last generated frame is an end-of-audio frame: every codebook but the last holds
    `codebook_eos_token_id`. `input_ids` has shape `(batch_size, num_frames, num_codebooks)`.
    """

    def __init__(self, codebook_eos_token_id: int):
        # No `eos_token_id` attribute on purpose: the generation loop only pads finished rows when a criterion exposes
        # one, and CSM does not pad them (the released checkpoints set no `eos_token_id`, so the previous loop never
        # padded either; the `eos_token_id`-gated padding with `codebook_pad_token_id` it had is dropped).
        self.codebook_eos_token_id = codebook_eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor | None, **kwargs) -> torch.BoolTensor:
        return (input_ids[:, -1, :-1] == self.codebook_eos_token_id).all(dim=-1)


class CsmGenerationMixin(GenerationMixin):
    """
    Csm generates one audio frame (`num_codebooks` tokens) per step: the backbone selects the first codebook token
    and the depth decoder generates the other codebooks, conditioned on the backbone's last hidden state. A row stops
    once its last frame holds `codebook_eos_token_id` in every codebook but the last; finished rows are not padded. A
    text prompt is dropped from the output, which then holds the generated frames only (`max_new_tokens` and
    `max_length` count frames); `output_audio=True` decodes the frames into one waveform per batch item.
    """

    _supported_generation_modes = [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]

    def _get_stopping_criteria(self, *args, **kwargs) -> StoppingCriteriaList:
        criteria = super()._get_stopping_criteria(*args, **kwargs)

        kept_criteria = StoppingCriteriaList()
        for criterion in criteria:
            if not isinstance(criterion, MaxLengthCriteria):
                logger.warning(
                    f"Csm does not support {criterion.__class__.__name__} stopping criteria, it will be ignored."
                )
            else:
                kept_criteria.append(criterion)
        kept_criteria.append(CsmEosFrameCriteria(self.config.codebook_eos_token_id))
        return kept_criteria

    def _prepare_generation_config(
        self, generation_config: GenerationConfig | None, **kwargs: Any
    ) -> tuple[GenerationConfig, dict]:
        """
        This method overrides [~generation.utils.GenerationMixin._prepare_generation_config].
        It ensures that the depth decoder generation config is initialized and that passed args as depth_decoder_* are properly handled.
        `output_audio` is kept on the prepared generation config for `_build_generate_output`.
        """
        output_audio = kwargs.pop("output_audio", False)

        # extract depth decoder kwargs and remove them from the main kwargs
        depth_decoder_kwargs = {
            k[len("depth_decoder_") :]: v for k, v in kwargs.items() if k.startswith("depth_decoder_")
        }

        # remove the depth decoder keys from the original kwargs
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("depth_decoder_")}

        # initialize the generation config
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        self.depth_decoder.generation_config.update(**depth_decoder_kwargs)

        # ensure the depth decoder generation config is valid
        depth_decoder_min_new_tokens = getattr(self.depth_decoder.generation_config, "min_new_tokens") or (
            self.config.num_codebooks - 1
        )
        depth_decoder_max_new_tokens = getattr(self.depth_decoder.generation_config, "max_new_tokens") or (
            self.config.num_codebooks - 1
        )

        if {depth_decoder_min_new_tokens, depth_decoder_max_new_tokens} != {self.config.num_codebooks - 1}:
            raise ValueError(
                f"depth_decoder_generation_config's min_new_tokens ({depth_decoder_min_new_tokens}) and max_new_tokens ({depth_decoder_max_new_tokens}) must be equal to self.config.num_codebooks - 1 ({self.config.num_codebooks - 1})"
            )
        elif self.depth_decoder.generation_config.return_dict_in_generate:
            logger.warning(
                "depth_decoder_generation_config.return_dict_in_generate is set to True, but this will be ignored as the depth decoder model does not return a dictionary in generate"
            )
            self.depth_decoder.generation_config.return_dict_in_generate = False

        self.depth_decoder.generation_config.min_new_tokens = depth_decoder_min_new_tokens
        self.depth_decoder.generation_config.max_new_tokens = depth_decoder_max_new_tokens

        generation_config.output_audio = output_audio
        # the depth decoder is conditioned on the backbone's last hidden state
        model_kwargs["output_hidden_states"] = True

        return generation_config, model_kwargs

    def _init_sequences(self, input_ids: torch.LongTensor, model_kwargs: dict[str, Any]) -> torch.LongTensor:
        # A text prompt is not part of the generated audio: start from an empty frame tensor so that only frames are
        # returned and `max_new_tokens` / `max_length` count frames (the cache is still sized for the prompt positions,
        # see `_prepare_generation`). A codebook prompt `(batch_size, seq_len, num_codebooks)` is returned together
        # with the generated frames.
        if input_ids.ndim == 2:
            return input_ids.new_zeros((input_ids.shape[0], 0, self.config.num_codebooks))
        return input_ids

    def _select_next_tokens(
        self,
        next_token_scores: torch.FloatTensor,
        generation_config: GenerationConfig,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        state: GenerationState,
    ) -> torch.LongTensor:
        # the backbone selects the first codebook token of the frame
        first_codebook_ids = super()._select_next_tokens(
            next_token_scores, generation_config, outputs, model_kwargs, state
        )

        # the depth decoder generates the other codebooks; position 0 is a placeholder it replaces by the backbone's
        # last hidden state
        depth_decoder_input_ids = nn.functional.pad(first_codebook_ids[:, None], (1, 0), value=0)
        backbone_last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        depth_decoder_outputs = self.depth_decoder.generate(
            input_ids=depth_decoder_input_ids, backbone_last_hidden_state=backbone_last_hidden_state.clone()
        )
        codebook_ids = (
            depth_decoder_outputs
            if isinstance(depth_decoder_outputs, torch.Tensor)
            else depth_decoder_outputs.sequences
        )
        # remove the placeholder in position 0 -> `(batch_size, num_codebooks)` frame
        return codebook_ids[:, 1:]

    def _build_generate_output(
        self,
        sequences: torch.LongTensor,
        state: GenerationState,
        generation_config: GenerationConfig,
        model_kwargs: dict[str, Any],
        **kwargs,
    ) -> CsmGenerateOutput | torch.LongTensor | list[torch.Tensor]:
        output = super()._build_generate_output(sequences, state, generation_config, model_kwargs, **kwargs)
        audio = self._decode_audio(sequences) if generation_config.output_audio else None
        if generation_config.return_dict_in_generate:
            return CsmGenerateOutput(audio=audio, **output)
        return audio if generation_config.output_audio else output

    def _decode_audio(self, audio_codes: torch.LongTensor) -> list[torch.Tensor]:
        """
        Decodes `(batch_size, num_frames, num_codebooks)` audio codes into one waveform per batch item, each cut at
        its first end-of-audio frame.
        """
        audio = []
        # TODO: @eustlb, this should be batched !!!
        # but requires making sure batched inference of the codec model works as intended
        for audio_codes_batch in audio_codes:
            # the stop criterion checks all codebooks but the last; the audio is cut at the first frame where *every*
            # codebook is EOS
            eos_idxs = (audio_codes_batch == self.config.codebook_eos_token_id).all(dim=-1).nonzero()
            cutoff_idx = eos_idxs.min() if eos_idxs.numel() != 0 else audio_codes_batch.shape[0]
            if cutoff_idx == 0:
                audio.append(audio_codes_batch.new_zeros(0, dtype=self.codec_model.dtype))
                continue
            audio_codes_batch = audio_codes_batch[:cutoff_idx]
            codec_decode_output = self.codec_model.decode(audio_codes_batch.transpose(0, 1).unsqueeze(0))
            audio.append(codec_decode_output.audio_values[0, 0])
        return audio

    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        input_values: torch.Tensor | None = None,
        input_values_cutoffs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        synced_gpus: bool | None = None,
        streamer: Optional["BaseStreamer"] = None,
        output_audio: bool | None = False,
        **kwargs,
    ) -> CsmGenerateOutput | torch.LongTensor | list[torch.Tensor]:
        r"""
        This method overrides [`~generation.utils.GenerationMixin.generate`] to match the specifics of the Csm model.
        Indeed, Csm model requires a custom generation sampling step:
        1. Infer the backbone model to sample the first codebook token
        2. Call generate on the depth decoder with the first codebook token as `input_ids` to sample the next codebook tokens
        3. Use these generated codebook tokens as `input_ids` to sample the next first codebook token using the backbone model
        4. Repeat until stopping criteria is met

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, do_sample=True)`.
        </Tip>

        Parameters:
            input_ids (`torch.Tensor` of shape (batch_size, seq_length), *optional*):
                The sequence used as a prompt for the backbone model. A text prompt is not returned: the output holds
                the generated audio frames only, and `max_new_tokens` / `max_length` count frames.
            input_values (`torch.Tensor` of shape (batch_size, channels, max_concatenated_audio_length), *optional*):
                The batched audio input values, where each batch entry contains the concatenation of all audio segments for that entry.
                These values will be encoded into codebook tokens using the codec model and merged with the text input ids provided in `input_ids`.
            input_values_cutoffs (`torch.Tensor` of shape (batch_size, max_num_audio), *optional*):
                Specify the end positions of audio segments within each batch entry, relative to the concatenated audio input.
                If a batch entry has fewer segments than the maximum, it is padded with -1. For example, in a batch of 2 sequences
                where the first contains 2 audio segments of length l1, and the second contains 1 audio segment of length l2,
                the input_values_cutoffs would be: [[l1, 2 * l1], [l2, -1]].
            generation_config ([`~generation.GenerationConfig`], *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
                to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
                deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            output_audio (`bool`, *optional*):
                Whether to return the generated audio.
            kwargs (`dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. Depth decoder specific kwargs should be prefixed with *depth_decoder_*.

        Return:
            [`CsmGenerateOutput`] or `torch.LongTensor` or `list[torch.FloatTensor]`: A [`CsmGenerateOutput`]
            (if `return_dict_in_generate=True` or when `config.return_dict_in_generate=True`) or a `torch.LongTensor` when `output_audio=False`
            or a `list[torch.FloatTensor]` otherwise.

        Example:

        ```python
        >>> from transformers import CsmProcessor, CsmForConditionalGeneration
        >>> from datasets import load_dataset, Audio

        >>> model_id = "sesame/csm-1b"
        >>> torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        >>> processor = AutoProcessor.from_pretrained(model_id)

        >>> ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        >>> # ensure the audio is 24kHz
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=24000))

        >>> conversation = []
        >>> # prepare a conversation with text and corresponding audio
        >>> for text, audio, speaker_id in zip(ds[:4]["text"], ds[:4]["audio"], ds[:4]["speaker_id"]):
        ...     conversation.append(
        ...         {
        ...             "role": f"{speaker_id}",
        ...             "content": [{"type": "text", "text": text}, {"type": "audio", "path": audio["array"]}],
        ...         }
        ...     )

        >>> # text prompt
        >>> conversation.append({"role": f"{ds[4]['speaker_id']}", "content": [{"type": "text", "text": ds[4]["text"]}]})

        >>> inputs = processor.apply_chat_template(
        ...     conversation,
        ...     tokenize=True,
        ...     return_dict=True,
        ... ).to(torch_device)

        >>> model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=torch_device)
        >>> audio = model.generate(**inputs, output_audio=True)
        >>> processor.save_audio(audio, "output.wav")
        ```
        """
        return super().generate(
            input_ids=input_ids,
            input_values=input_values,
            input_values_cutoffs=input_values_cutoffs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            synced_gpus=synced_gpus,
            streamer=streamer,
            output_audio=output_audio,
            **kwargs,
        )
