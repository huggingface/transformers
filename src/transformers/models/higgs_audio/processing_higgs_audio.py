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
"""HiggsAudioProcessor."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ...audio_utils import make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class HiggsAudioProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@dataclass
class HiggsAudioResponse:
    audio: Optional[np.ndarray] = None
    generated_text: str = ""
    sampling_rate: Optional[int] = None


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data (:obj:`torch.Tensor`):
            The data with delay pattern applied. It will have shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret (:obj:`torch.Tensor`):
            Recovered data with delay pattern removed. It will have shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out_l, dim=0)


class HiggsAudioProcessor(ProcessorMixin):
    r"""
    Constructs a Higgs Audio processor which wraps a [`DacFeatureExtractor`], a [`AutoTokenizer`],
    and a [`HiggsAudioTokenizerModel`] into a single processor. It inherits, the audio feature extraction, tokenizer,
    and audio encode/decode functionalities.
    See [`~HiggsAudioProcessor.__call__`] and [`~HiggsAudioProcessor.decode`] for more information.

    Args:
        feature_extractor (`DacFeatureExtractor`):
            An instance of [`DacFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of [`AutoTokenizer`]. The tokenizer is a required input.
        audio_tokenizer (`HiggsAudioTokenizerModel`):
            An instance of [`HiggsAudioTokenizerModel`]. The audio tokenizer is a required input.
        chat_template (`str`, *optional*):
            A template string for chat formatting when combining text and audio interactions.
    """

    feature_extractor_class = "DacFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    audio_tokenizer_class = "HiggsAudioTokenizerModel"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_tokenizer,
        chat_template=None,
        audio_token="<|AUDIO_OUT|>",
        audio_bos_token="<|audio_out_bos|>",
        audio_eos_token="<|audio_eos|>",
    ):
        # TODO: better like that? (so llava like) or should it be more like Dia with things in the HiggsAudioProcessorKwargs
        if chat_template is None:
            chat_template = self.default_chat_template

        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token = tokenizer.audio_bos_token if hasattr(tokenizer, "audio_bos_token") else audio_bos_token
        self.audio_eos_token = tokenizer.audio_eos_token if hasattr(tokenizer, "audio_eos_token") else audio_eos_token

        super().__init__(
            feature_extractor,
            tokenizer,
            audio_tokenizer=audio_tokenizer,
            chat_template=chat_template,
        )

    def __call__(
        self,
        text: Union[str, list[str]],
        audio: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        output_labels: Optional[bool] = False,
        **kwargs: Unpack[HiggsAudioProcessorKwargs],
    ):
        output_kwargs = self._merge_kwargs(
            HiggsAudioProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
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

        # Process audio and expand audio tokens if needed
        audio_ids_list = []
        if audio is not None:
            # Ensure audio is a list
            if isinstance(audio, np.ndarray):
                audio = [audio]

            # Process each audio
            for audio_data in audio:
                # Assume audio_data comes with the correct sample rate or resample
                if hasattr(audio_data, "shape"):
                    processed_audio = audio_data
                else:
                    processed_audio = np.array(audio_data)
                # Generate audio tokens and store them in a list
                input_values = self.feature_extractor(processed_audio)
                input_values = torch.tensor(
                    input_values["input_values"][0], device=self.audio_tokenizer.device
                ).unsqueeze(0)
                audio_ids = self.audio_tokenizer.encode(input_values).audio_codes

                audio_ids = torch.cat(
                    [
                        torch.full((*audio_ids.shape[:2], 1), 1024, dtype=torch.int, device=audio_ids.device),
                        audio_ids,
                        torch.full((*audio_ids.shape[:2], 1), 1025, dtype=torch.int, device=audio_ids.device),
                    ],
                    dim=2,
                )

                audio_ids = self.build_delay_pattern_mask(
                    audio_ids,
                    bos_token_id=1024,
                    pad_token_id=1025,
                )[0].squeeze(0)
                audio_ids_list.append(audio_ids.squeeze(0).transpose(0, 1).cpu())

            # Get the actual number of audio tokens from encoded audio
            num_audio_tokens_list = [len(audio_ids) for audio_ids in audio_ids_list]
            num_audio_tokens_list_copy = num_audio_tokens_list.copy()

            # expand the text to repeat the audio token for the corresponding number of frames
            expanded_text = []
            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    num_audio_tokens = num_audio_tokens_list_copy.pop(0)
                    expanded_audio_token = self.audio_token * num_audio_tokens

                    replace_str.append(expanded_audio_token)
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)

            text = expanded_text

        encoding = self.tokenizer(text, **text_kwargs)
        data = {}
        data.update(encoding)
        if audio is not None:
            data.update(
                {
                    "audio_input_ids": torch.stack(audio_ids_list, dim=0),
                }
            )

        return BatchFeature(data=data, tensor_type="pt")

    def batch_decode(
        self,
        decoder_text_ids: "torch.Tensor",
        decoder_audio_ids: list["torch.Tensor"],
        prompt_token_length: int,
    ) -> list["torch.Tensor"]:
        raise NotImplementedError("Higgs Audio currently only supports single sample generation")

    def decode(
        self,
        decoder_audio_ids: list["torch.Tensor"],
        **kwargs: Unpack[HiggsAudioProcessorKwargs],
    ) -> HiggsAudioResponse:
        output_kwargs = self._merge_kwargs(
            HiggsAudioProcessorKwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]
        audio_codebook_size = 1024

        if len(decoder_audio_ids) > 0:
            wv_list = []
            for output_audio in decoder_audio_ids:
                vq_code = revert_delay_pattern(output_audio).clip(0, audio_codebook_size - 1)[:, 1:-1]
                wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0][0, 0]
                wv_list.append(wv_numpy.detach().cpu().numpy())
            wv_numpy = np.concatenate(wv_list)
        else:
            wv_numpy = None

        return HiggsAudioResponse(
            audio=wv_numpy,
            sampling_rate=24000
        )

    def build_delay_pattern_mask(
        self,
        input_ids: "torch.LongTensor",
        bos_token_id: int,
        pad_token_id: int,
        ) -> tuple["torch.LongTensor", "torch.LongTensor"]:

        delay_pattern = list(range(self.audio_tokenizer.config.num_quantizers))

        bsz, num_codebooks, seq_len = input_ids.shape

        new_seq_len = seq_len + num_codebooks - 1
        input_ids_with_gen_mask = torch.ones((bsz, num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device)
        bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
        eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
        input_ids_with_gen_mask[bos_mask] = bos_token_id
        input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
        input_ids = input_ids_with_gen_mask.clone()
        input_ids[eos_mask] = pad_token_id
        input_ids_with_gen_mask[eos_mask] = -1
        return input_ids, input_ids_with_gen_mask


__all__ = ["HiggsAudioProcessor"]
