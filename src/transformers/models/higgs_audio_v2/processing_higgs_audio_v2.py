# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import re
from itertools import islice
from pathlib import Path

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_soundfile_available, is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn.functional as F


if is_soundfile_available():
    import soundfile as sf


logger = logging.get_logger(__name__)


class HiggsAudioV2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "padding": False,
            "sampling_rate": 24000,
        },
    }


class HiggsAudioV2Processor(ProcessorMixin):
    r"""
    Constructs a Higgs Audio processor which wraps a [`DacFeatureExtractor`], a [`AutoTokenizer`],
    and a [`HiggsAudioV2TokenizerModel`] into a single processor. It inherits, the audio feature extraction, tokenizer,
    and audio encode/decode functionalities.
    See [`~HiggsAudioV2Processor.__call__`] and [`~HiggsAudioV2Processor.decode`] for more information.

    Args:
        feature_extractor (`DacFeatureExtractor`):
            An instance of [`DacFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of [`AutoTokenizer`]. The tokenizer is a required input.
        audio_tokenizer (`HiggsAudioV2TokenizerModel`):
            An instance of [`HiggsAudioV2TokenizerModel`]. The audio tokenizer is a required input.
        chat_template (`str`, *optional*):
            A template string for chat formatting when combining text and audio interactions.
        audio_token (`str`, *optional*, defaults to `"<|AUDIO_OUT|>"`):
            The token used to represent audio output in the text sequence.
        audio_bos_token (`str`, *optional*, defaults to `"<|audio_out_bos|>"`):
            The beginning-of-sequence token for audio output.
        audio_eos_token (`str`, *optional*, defaults to `"<|audio_eos|>"`):
            The end-of-sequence token for audio output.
        audio_delay_token (`str`, *optional*, defaults to `"<|reserved_special_token_6|>"`):
            The token used for audio delay pattern in multi-codebook generation.
        audio_stream_bos_id (`int`, *optional*, defaults to 1024):
            The ID for the beginning-of-stream token in audio sequences.
        audio_stream_eos_id (`int`, *optional*, defaults to 1025):
            The ID for the end-of-stream token in audio sequences.
    """

    feature_extractor_class = "DacFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "HiggsAudioV2TokenizerModel"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_tokenizer,
        chat_template=None,
        audio_token="<|AUDIO_OUT|>",
        audio_bos_token="<|audio_out_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_delay_token="<|reserved_special_token_6|>",
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
    ):
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        self.audio_bos_token = tokenizer.audio_bos_token if hasattr(tokenizer, "audio_bos_token") else audio_bos_token
        self.audio_eos_token = tokenizer.audio_eos_token if hasattr(tokenizer, "audio_eos_token") else audio_eos_token
        self.audio_delay_token = (
            tokenizer.audio_delay_token if hasattr(tokenizer, "audio_delay_token") else audio_delay_token
        )
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token_id = tokenizer.convert_tokens_to_ids(self.audio_bos_token)
        self.audio_eos_token_id = tokenizer.convert_tokens_to_ids(self.audio_eos_token)
        self.audio_delay_token_id = tokenizer.convert_tokens_to_ids(self.audio_delay_token)
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id

        super().__init__(
            feature_extractor,
            tokenizer,
            audio_tokenizer=audio_tokenizer,
            chat_template=chat_template,
        )

    def get_audio_tokens(self, num_audio_tokens):
        """
        Returns the audio tokens for a given number of audio tokens.
        """
        num_codebooks = self.audio_tokenizer.config.num_quantizers
        return self.audio_token * (num_audio_tokens - (num_codebooks - 1)) + self.audio_delay_token * (
            num_codebooks - 1
        )

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        **kwargs: Unpack[HiggsAudioV2ProcessorKwargs],
    ):
        output_kwargs = self._merge_kwargs(
            HiggsAudioV2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")
        n_audio_in_text = [t.count(self.audio_token) for t in text]

        n_audio = 0
        if audio is not None:
            audio = make_list_of_audio(audio)
            n_audio = len(audio)

        if sum(n_audio_in_text) > 0 and n_audio != sum(n_audio_in_text):
            if audio is None:
                raise ValueError("No audio were provided, but there are audio tokens in the prompt")
            else:
                raise ValueError(
                    f"The number of audio tokens in each text ({n_audio_in_text}) should be the same as the "
                    f"number of provided audios ({n_audio})."
                )
        elif sum(n_audio_in_text) == 0 and n_audio > 0:
            raise ValueError("Audio were provided, but there are no audio tokens in the prompt")

        if audio is not None:
            # tokenize audio
            audio_input_ids_list = []
            for audio_el in audio:
                # TODO: @eustlb, this should be batched !!!
                audio_inputs = self.feature_extractor(audio_el, **audio_kwargs)

                # TODO: @eustlb, padding_mask should be supported...
                audio_inputs.pop("padding_mask", None)
                audio_inputs.to(self.audio_tokenizer.device)
                audio_input_ids = self.audio_tokenizer.encode(**audio_inputs).audio_codes

                # add audio eos and bos
                bos_codes = audio_input_ids.new_full((*audio_input_ids.shape[:2], 1), self.audio_stream_bos_id)
                eos_codes = audio_input_ids.new_full((*audio_input_ids.shape[:2], 1), self.audio_stream_eos_id)
                audio_input_ids = torch.cat([bos_codes, audio_input_ids, eos_codes], dim=2)

                audio_input_ids = self.build_delay_pattern(audio_input_ids)
                audio_input_ids_list.append(audio_input_ids[0].transpose(0, 1))

            # expand audio tokens in text
            num_audio_tokens_iter = iter(len(audio_input_ids) for audio_input_ids in audio_input_ids_list)
            for i in range(len(text)):
                expanded = re.sub(
                    re.escape(self.audio_token), lambda _: self.get_audio_tokens(next(num_audio_tokens_iter)), text[i]
                )
                text[i] = expanded

            # convert to nested list according to n_audio_in_text
            # [audio_1, audio_2, ...] -> [[audio_1_1, audio_1_2, ...], [audio_2_1, audio_2_2, ...], ...]
            audio_input_ids_iter = iter(audio_input_ids_list)
            audio_input_ids_list = [list(islice(audio_input_ids_iter, l)) for l in n_audio_in_text]
            audio_input_ids_list = [torch.cat(batch_el, dim=0) for batch_el in audio_input_ids_list]

            # pad and stack
            lenghts = [ids.shape[0] for ids in audio_input_ids_list]
            max_length = max(lenghts)
            audio_input_ids_list = [
                F.pad(ids, (0, 0, 0, max_length - ids.shape[0]), value=self.audio_stream_eos_id)
                for ids in audio_input_ids_list
            ]
            audio_input_ids = torch.stack(audio_input_ids_list, dim=0)
            audio_input_ids_mask = torch.arange(max_length)[None, :] < torch.tensor(lenghts)[:, None]

        # tokenize text
        data = self.tokenizer(text, **text_kwargs)
        if audio is not None:
            data.update(
                {
                    "audio_input_ids": audio_input_ids,
                    "audio_input_ids_mask": audio_input_ids_mask,
                }
            )

        if output_labels:
            labels = data["input_ids"].clone()
            labels[labels == self.audio_token_id] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels[labels == self.audio_bos_token_id] = -100
            data["labels"] = labels

            if audio is not None:
                audio_labels = audio_input_ids.clone()
                audio_labels[audio_labels == self.audio_stream_bos_id] = -100
                audio_labels[audio_labels == self.audio_stream_eos_id] = -100
                data.update({"audio_labels": audio_labels})

        return BatchFeature(data=data, tensor_type="pt")

    def batch_decode(self, audio_input_ids):
        """
        Decode a batch of audio token sequences into audio waveforms.

        This method processes audio token sequences generated by the model, extracting the actual audio tokens
        between the beginning-of-stream (BOS) and end-of-stream (EOS) markers, reverting the delay pattern
        used during generation, and decoding them into audio waveforms using the audio tokenizer.

        Args:
            audio_input_ids (`torch.LongTensor`):
                Shape `(batch_size, sequence_length, num_codebooks)`
                The audio token sequences to decode. These should contain audio tokens with BOS and EOS markers
                in a delay pattern format as generated by the model.

        Returns:
            `list[torch.Tensor]`: A list of decoded audio waveforms, one for each batch element. Each waveform
            is a 1D tensor containing the audio samples.
        """
        # start idx should be the last sequence index of the audio bos tokens
        audio_bos_token_idxs = (audio_input_ids == self.audio_stream_bos_id).all(-1).nonzero()
        start_of_generation_idx = audio_bos_token_idxs[-1, -1].item()

        audio_input_ids = audio_input_ids[:, start_of_generation_idx:]

        # end idx for each batch idx should be the first sequence index of the audio eos tokens
        audio_eos_token_idxs = (audio_input_ids == self.audio_stream_eos_id).all(-1).nonzero()
        end_of_generation_idxs = [
            audio_eos_token_idxs[audio_eos_token_idxs[:, 0] == batch_idx, 1].min().item()
            if len(audio_eos_token_idxs[audio_eos_token_idxs[:, 0] == batch_idx]) > 0
            else audio_input_ids.shape[1]
            for batch_idx in range(audio_input_ids.shape[0])
        ]

        audios = []
        with torch.no_grad():
            # TODO: @eustlb, this should be batched !!!
            for batch_idx in range(audio_input_ids.shape[0]):
                audio_token_ids = audio_input_ids[batch_idx, 1 : end_of_generation_idxs[batch_idx]]
                audio_token_ids = self.revert_delay_pattern(audio_token_ids).clip(0, self.audio_stream_bos_id - 1)
                audio_i = (
                    self.audio_tokenizer.decode(audio_token_ids.transpose(0, 1).unsqueeze(0))
                    .audio_values.cpu()
                    .squeeze()
                )
                audios.append(audio_i)

        return audios

    def decode(self, audio_input_ids):
        if audio_input_ids.shape[0] != 1:
            raise ValueError(
                f"Expecting a single output to be decoded but received {audio_input_ids.shape[0]} samples instead."
            )

        return self.batch_decode(audio_input_ids)[0]

    def build_delay_pattern(self, input_ids):
        bsz, num_codebooks, seq_len = input_ids.shape
        new_seq_len = seq_len + num_codebooks - 1

        # Create output tensor with delay pattern
        output = torch.ones((bsz, num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device)

        # Create masks for different regions
        bos_mask = torch.tril(output, -1) > 0
        eos_mask = torch.triu(output, seq_len) > 0
        data_mask = ~(bos_mask | eos_mask)

        # Fill the tensor
        output[bos_mask] = self.audio_stream_bos_id
        output[data_mask] = input_ids.reshape(-1)
        output[eos_mask] = self.audio_stream_eos_id

        return output

    def revert_delay_pattern(self, input_ids):
        seq_len, num_codebooks = input_ids.shape
        # Extract diagonal slices from the delay pattern
        slices = []
        for i in range(num_codebooks):
            end_idx = seq_len - num_codebooks + 1 + i
            slices.append(input_ids[i:end_idx, i : i + 1])

        return torch.cat(slices, dim=1)

    # Copied from transformers.models.csm.processing_csm.CsmProcessor.save_audio with Csm->HiggsAudioV2
    def save_audio(
        self,
        audio: AudioInput,
        saving_path: str | Path | list[str | Path],
        **kwargs: Unpack[HiggsAudioV2ProcessorKwargs],
    ):
        # TODO: @eustlb, this should be in AudioProcessor
        if not is_soundfile_available():
            raise ImportError("Please install `soundfile` to save audio files.")

        # ensure correct audio input
        audio = make_list_of_audio(audio)

        # ensure correct saving path
        if isinstance(saving_path, (str, Path)):
            saving_path = [saving_path]
        elif not (isinstance(saving_path, (list, tuple)) and all(isinstance(p, (str, Path)) for p in saving_path)):
            raise ValueError("Invalid input path. Please provide a string, or a list of strings")

        if len(audio) != len(saving_path):
            raise ValueError("The number of audio and saving paths must be the same")

        output_kwargs = self._merge_kwargs(
            HiggsAudioV2ProcessorKwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]
        sampling_rate = audio_kwargs["sampling_rate"]

        for audio_value, p in zip(audio, saving_path):
            if isinstance(audio_value, torch.Tensor):
                audio_value = audio_value.cpu().float().numpy()
            sf.write(p, audio_value, sampling_rate)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names

        # TODO: @eustlb, to be standardized!!
        audio_tokenizer_input_names = ["audio_input_ids", "audio_input_ids_mask"]
        return tokenizer_input_names + audio_tokenizer_input_names


__all__ = ["HiggsAudioV2Processor"]
