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
from itertools import islice
import re

import numpy as np

from ...audio_utils import make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn.functional as F


logger = logging.get_logger(__name__)


class HiggsAudioProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "padding": False,
            "sampling_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


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
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
    ):
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token = tokenizer.audio_bos_token if hasattr(tokenizer, "audio_bos_token") else audio_bos_token
        self.audio_eos_token = tokenizer.audio_eos_token if hasattr(tokenizer, "audio_eos_token") else audio_eos_token
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id

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

        if audio is not None:
            # tokenize audio
            audio_input_ids_list = []
            for audio_el in audio:
                # TODO: @eustlb, this should be batched !!!
                audio_inputs = self.feature_extractor(audio_el, **audio_kwargs)
                audio_inputs.to(self.audio_tokenizer.device)
                audio_input_ids = self.audio_tokenizer.encode(**audio_inputs).audio_codes

                # add audio eos and bos
                bos_codes = audio_input_ids.new_full((*audio_input_ids.shape[:2], 1), self.audio_stream_bos_id)
                eos_codes = audio_input_ids.new_full((*audio_input_ids.shape[:2], 1), self.audio_stream_eos_id)
                audio_input_ids = torch.cat([bos_codes, audio_input_ids, eos_codes], dim=2)

                audio_input_ids = self.build_delay_pattern_mask(audio_input_ids)
                audio_input_ids_list.append(audio_input_ids[0].transpose(0, 1).cpu())

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
            
            # expand audio tokens in text
            num_audio_tokens_iter = iter(len(audio_input_ids) for audio_input_ids in audio_input_ids_list)
            for i in range(len(text)): 
                expanded = re.sub(
                    re.escape(self.audio_token),
                    lambda _: self.audio_token * next(num_audio_tokens_iter),
                    text[i]
                )
                text[i] = expanded

        # tokenize text
        data = self.tokenizer(text, **text_kwargs)
        if audio is not None:
            data.update(
                {
                    "audio_input_ids": audio_input_ids,
                    "audio_input_ids_mask": audio_input_ids_mask,
                }
            )

        return BatchFeature(data=data, tensor_type="pt")

    def decode(
        self,
        audio_token_ids: Union[int, list[int], np.ndarray, "torch.Tensor"],
        **kwargs,
    ) -> str:
        output_kwargs = self._merge_kwargs(
            HiggsAudioProcessorKwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]
        audio_codebook_size = 1024

        audio_token_ids = audio_token_ids.transpose(1, 2)

        wv_list = []
        for output_audio in audio_token_ids:
            vq_code = self.revert_delay_pattern(output_audio).clip(0, audio_codebook_size - 1)[:, 1:-1]
            wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0][0, 0]
            wv_list.append(wv_numpy.detach().cpu().numpy())
        wv_numpy = np.concatenate(wv_list)

        return wv_numpy

     def batch_decode(
        self,
        decoder_input_ids: "torch.Tensor",
        audio_prompt_len: Optional[int] = None,
        **kwargs: Unpack[DiaProcessorKwargs],
    ) -> list["torch.Tensor"]:
        output_kwargs = self._merge_kwargs(
            DiaProcessorKwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]

        delay_pattern = audio_kwargs.pop("delay_pattern", None)
        audio_bos_token_id = audio_kwargs.pop("bos_token_id", None)
        audio_pad_token_id = audio_kwargs.pop("pad_token_id", None)
        if audio_bos_token_id is None or audio_pad_token_id is None or delay_pattern is None:
            raise ValueError(
                "To enable decoding for Dia, we need the `bos_token_id`, `pad_token_id`, "
                "and `delay_pattern`. You may have accidentally overwritten one of those."
            )

        # either decode the whole audio sequence or only the generated parts
        if audio_prompt_len is not None:
            audio_prompt_len = torch.tensor(audio_prompt_len, device=decoder_input_ids.device, dtype=torch.long)
            start_of_generation_idx = audio_prompt_len[None].expand(decoder_input_ids.shape[0])
        else:
            start_of_generation_idx = (decoder_input_ids[:, :, 0] == audio_bos_token_id).sum(dim=-1)
        # -1 for the eos token
        end_of_generation_idx = (
            decoder_input_ids.shape[1] - (decoder_input_ids[:, :, 0] == audio_pad_token_id).sum(dim=-1) - 1
        )

        # revert delay
        
        bsz, seq_len, num_channels = decoder_input_ids.shape
        precomputed_idx = self.build_indices(
            bsz=bsz,
            seq_len=seq_len,
            num_channels=num_channels,
            delay_pattern=delay_pattern,
            revert=True,
        )

        output_sequences = self.apply_audio_delay(
            audio=decoder_input_ids,
            # We do not care about these values as we cut them out
            # with `start_of_generation_idx` and `end_of_generation_idx`
            pad_token_id=-1,
            bos_token_id=-1,
            precomputed_idx=precomputed_idx,
        ).transpose(1, 2)

        # retrieve the correct sequences each
        audios = []
        with torch.no_grad():
            for i in range(start_of_generation_idx.shape[0]):
                output_i = output_sequences[i, :, start_of_generation_idx[i] : end_of_generation_idx[i]][None, ...]
                output_i = output_i.to(self.audio_tokenizer.device)
                audio_i = self.audio_tokenizer.decode(audio_codes=output_i).audio_values.cpu().squeeze()
                audios.append(audio_i)

        return audios
    
    def decode(
        self,
        decoder_input_ids: "torch.Tensor",
        audio_prompt_len: Optional[int] = None,
        **kwargs: Unpack[DiaProcessorKwargs],
    ) -> "torch.Tensor":
        """
        Decodes a single sequence of audio codebooks into the respective audio waveform via the
        `audio_tokenizer`. See [`~DacModel.decode`] and [`~DiaProcessor.batch_decode`] for more information.
        """
        if decoder_input_ids.shape[0] != 1:
            raise ValueError(
                f"Expecting a single output to be decoded but received {decoder_input_ids.shape[0]} samples instead."
            )

        return self.batch_decode(decoder_input_ids, audio_prompt_len, **kwargs)[0]

    def build_delay_pattern_mask(self, input_ids):
        bos_token_id = self.audio_stream_bos_id
        pad_token_id = self.audio_stream_eos_id
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

        return input_ids

    def revert_delay_pattern(self, data):
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


__all__ = ["HiggsAudioProcessor"]
