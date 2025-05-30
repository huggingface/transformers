# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
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
"""
Processor class for MiniCPMO.
"""

import math
import re

from typing import Any, Dict, List, Literal, Optional, Union

import librosa
import numpy as np
import PIL
import PIL.Image
import torch
import torchaudio
from PIL import Image

from transformers import AutoImageProcessor, AutoTokenizer, WhisperFeatureExtractor
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging, TensorType

from ...feature_extraction_utils import BatchFeature
from ...image_utils import (
    ChannelDimension,
    infer_channel_dimension_format,
    is_torch_tensor,
    to_numpy_array,
    valid_images,
)
from ...utils import is_torch_device, is_torch_dtype, requires_backends, TensorType

logger = logging.get_logger(__name__)


class MiniCPMOBatchFeature(BatchFeature):
    r"""
    Extend from BatchFeature for supporting various image size
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        if tensor_type is None:
            return self

        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)

        def converter(value):
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)
                    return tensor
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        for key, value in self.items():
            self[key] = recursive_converter(converter, value)
        return self

    def to(self, *args, **kwargs) -> "MiniCPMOBatchFeature":
        requires_backends(self, ["torch"])
        import torch

        def cast_tensor(v):
            # check if v is a floating point
            if torch.is_floating_point(v):
                # cast and send to device
                return v.to(*args, **kwargs)
            elif device is not None:
                return v.to(device=device)
            else:
                return v

        new_data = {}
        device = kwargs.get("device")
        # Check if the args are a device or a dtype
        if device is None and len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if is_torch_dtype(arg):
                # The first argument is a dtype
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                # it's something else
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            new_data[k] = recursive_converter(cast_tensor, v)
        self.data = new_data
        return self
    

class MiniCPM_o_2_6Processor(ProcessorMixin):
    r"""
    Constructs a MiniCPMV processor which wraps a MiniCPMV image processor and a MiniCPMV tokenizer into a single processor.

    [`MiniCPMVProcessor`] offers all the functionalities of [`MiniCPMVImageProcessor`] and [`LlamaTokenizerWrapper`]. See the
    [`~MiniCPMVProcessor.__call__`] and [`~MiniCPMVProcessor.decode`] for more information.

    Args:
        image_processor ([`MiniCPMVImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerWrapper`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, config=None, image_processor=None, feature_extractor=None, tokenizer=None):
        if config is None:
            raise ValueError("Config must be provided.")
        if image_processor is None:
            image_processor = AutoImageProcessor.from_pretrained(config._name_or_path)
        if feature_extractor is None:
            feature_extractor = WhisperFeatureExtractor.from_pretrained(config._name_or_path)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config._name_or_path, trust_remote_code=True)
        super().__init__(image_processor, feature_extractor, tokenizer)
        self.version = image_processor.version

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: ImageInput = None,
        audios: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]] = None,
        audio_parts: Optional[list] = None,
        max_length: Optional[int] = None,
        do_pad: Optional[bool] = True,
        max_slice_nums: int = None,
        use_image_id: bool = True,
        chunk_input: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        sampling_rate: Optional[int] = 16000,
        **kwargs,
    ) -> MiniCPMOBatchFeature:
        if images is not None:
            image_inputs = self.image_processor(
                images, do_pad=do_pad, max_slice_nums=max_slice_nums, return_tensors=return_tensors
            )
        else:
            image_inputs = None

        if audios is not None:
            audio_features, audio_feature_lens, audio_phs = self.audio_feature_extract(
                audios, audio_parts, chunk_input, sampling_rate
            )
        else:
            audio_features, audio_feature_lens, audio_phs = [], [], []

        model_inputs = self._convert_omni_to_inputs(
            image_inputs,
            audio_phs,
            text,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            max_length=max_length,
            **kwargs,
        )

        model_inputs["audio_features"] = audio_features
        model_inputs["audio_feature_lens"] = audio_feature_lens

        return MiniCPMOBatchFeature(data={**model_inputs})

    def get_audio_placeholder(self, audio_lens, chunk_input, chunk_length):
        pool_step = 2
        feature_lens = math.ceil(audio_lens / self.feature_extractor.hop_length)

        feature_lens = (feature_lens - 1) // 2 + 1
        output_lens = (feature_lens - pool_step) // pool_step + 1

        if chunk_input:
            fbank_feat_in_chunk = int(chunk_length * 100)
            cnn_feat_in_chunk = (fbank_feat_in_chunk - 1) // 2 + 1
            audio_embeds_in_chunk = (cnn_feat_in_chunk - pool_step) // pool_step + 1
            num_audio_chunks = (output_lens + audio_embeds_in_chunk - 1) // audio_embeds_in_chunk

            place_holders = ""
            total_unk_len = 0
            for _ in range(num_audio_chunks):
                unk_len = min(audio_embeds_in_chunk, output_lens - total_unk_len)
                place_holders += self.tokenizer.audio_start + "<unk>" * unk_len + self.tokenizer.audio_end
                total_unk_len += unk_len
            audio_placeholder = place_holders
        else:
            audio_placeholder = self.tokenizer.audio_start + "<unk>" * output_lens + self.tokenizer.audio_end

        return audio_placeholder

    def audio_feature_extract(
        self,
        audios: Union[np.ndarray, List[np.ndarray], List[List[np.ndarray]]],
        audio_parts: Optional[list] = None,
        chunk_input: Optional[bool] = False,
        sampling_rate: Optional[int] = None,
        chunk_length: Optional[int] = 1,
        **kwargs,
    ):
        if isinstance(audios, np.ndarray):
            audios_list = [[audios]]
        elif isinstance(audios[0], np.ndarray):
            audios_list = [audios]
        else:
            audios_list = audios

        if audio_parts is not None:
            assert len(audio_parts) == len(audios_list)
            for parts, audios in zip(audio_parts, audios_list):
                assert len(parts) == len(audios)

        audio_feature_lens_list = []
        audio_ph_list = []

        audio_features_all = []

        # audio placeholder not dependent on audio_parts
        for audios in audios_list:
            if audios:
                audio_ph_list.append([self.get_audio_placeholder(len(a), chunk_input, chunk_length) for a in audios])
            else:
                audio_ph_list.append([])

        for idx, audios in enumerate(audios_list):
            if audio_parts is not None:
                # same audio part merge
                audio_part = audio_parts[idx]
                merge_audio = []
                cur_audio = []
                for aid, (part, audio) in enumerate(zip(audio_part, audios)):
                    if aid == 0 or audio_part[aid] == audio_part[aid - 1]:
                        cur_audio.append(audio)
                    else:
                        merge_audio.append(np.hstack(cur_audio))
                        cur_audio = [audio]
                if cur_audio:
                    merge_audio.append(np.hstack(cur_audio))

            else:
                merge_audio = audios

            audio_feature_lens = []

            # If the audio exceeds 30 seconds, split it into chunks every 30 seconds.
            final_merge_audio = []
            max_audio_inp_len = 30 * sampling_rate
            for audio in merge_audio:
                if len(audio) <= max_audio_inp_len:
                    final_merge_audio.append(audio)
                else:
                    for i in range(math.ceil(len(audio) / max_audio_inp_len)):
                        final_merge_audio.append(audio[i * max_audio_inp_len : (i + 1) * max_audio_inp_len])

            if audios:
                audio_inputs = self.feature_extractor(
                    final_merge_audio,
                    sampling_rate=sampling_rate,
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                    **kwargs,
                )
                audio_feature = audio_inputs["input_features"]
                actual_lens = audio_inputs["attention_mask"].sum(dim=1)

                for feat, lens in zip(audio_feature, actual_lens):
                    audio_features_all.append(feat[:, :lens])
                    audio_feature_lens.append(lens)

                audio_feature_lens = torch.hstack(audio_feature_lens)
                audio_feature_lens_list.append(audio_feature_lens)
            else:
                audio_feature_lens_list.append([])

        if audio_features_all:
            audio_features = [i.permute(1, 0) for i in audio_features_all]
            audio_features = torch.nn.utils.rnn.pad_sequence(
                audio_features, batch_first=True, padding_value=0.0
            ).permute(0, 2, 1)
        else:
            audio_features = []

        return audio_features, audio_feature_lens_list, audio_ph_list

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        output_ids = args[0]
        result_text = []
        for result in output_ids:
            result = result[result != 0]
            if result[0] == self.tokenizer.bos_id:
                result = result[1:]
            if result[-1] == self.tokenizer.eos_id:
                result = result[:-1]
            result_text.append(self.tokenizer.decode(result, *args[1:], **kwargs).strip())
        return result_text
        # return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        result = args[0]
        result = result[result != 0]
        if result[0] == self.tokenizer.bos_id:
            result = result[1:]
        if result[-1] == self.tokenizer.eos_id or (
            hasattr(self.tokenizer, "eot_id") and result[-1] == self.tokenizer.eot_id
        ):
            result = result[:-1]
        return self.tokenizer.decode(result, *args[1:], **kwargs).strip()

    def _convert(self, input_str, max_inp_length: Optional[int] = None, **kwargs):
        input_ids = self.tokenizer.encode(input_str, **kwargs)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        ## image bound
        start_cond = (input_ids == self.tokenizer.im_start_id) | (input_ids == self.tokenizer.slice_start_id)
        end_cond = (input_ids == self.tokenizer.im_end_id) | (input_ids == self.tokenizer.slice_end_id)

        image_start_idx = torch.where(start_cond)[0]
        image_start_idx += 1
        image_end_idx = torch.where(end_cond)[0]

        valid_image_nums = max(len(image_start_idx), len(image_end_idx))

        image_bounds = torch.hstack(
            [
                image_start_idx[:valid_image_nums].unsqueeze(-1),
                image_end_idx[:valid_image_nums].unsqueeze(-1),
            ]
        )

        ##  audio bound
        audio_start_idx = torch.where(input_ids == self.tokenizer.audio_start_id)[0]
        audio_end_idx = torch.where(input_ids == self.tokenizer.audio_end_id)[0]
        assert len(audio_start_idx) == len(audio_end_idx)
        audio_bounds = torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)])

        spk_start_idx = torch.where(input_ids == self.tokenizer.spk_start_id)[0]
        spk_end_idx = torch.where(input_ids == self.tokenizer.spk_end_id)[0]
        assert len(spk_start_idx) == len(spk_end_idx)
        spk_bounds = torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])

        return input_ids, image_bounds, audio_bounds, spk_bounds

    def _convert_omni_to_inputs(
        self,
        images,
        audio_phs,
        texts: Union[str, List[str]],
        truncation=None,
        max_length=None,
        max_slice_nums=None,
        use_image_id=None,
        return_tensors=None,
        **kwargs,
    ):
        if images is None and audio_phs is None:
            model_inputs = self.tokenizer(
                texts, return_tensors=return_tensors, truncation=truncation, max_length=max_length, **kwargs
            )
            return MiniCPMOBatchFeature(data={**model_inputs})

        image_tag = "(<image>./</image>)"
        image_pattern = "\(<image>./</image>\)"
        audio_tag = "(<audio>./</audio>)"
        audio_pattern = "\(<audio>./</audio>\)"
        split_pattern = f"({image_pattern}|{audio_pattern})"

        if isinstance(texts, str):
            texts = [texts]

        bs = len(texts)
        if images is not None:
            images, image_sizes, tgt_sizes = images["pixel_values"], images["image_sizes"], images["tgt_sizes"]
        else:
            images, image_sizes, tgt_sizes = [[]] * bs, [[]] * bs, [[]] * bs

        input_ids_list = []
        image_bounds_list = []
        audio_bounds_list = []
        spk_bounds_list = []

        for index, text in enumerate(texts):
            text_chunks = re.split(split_pattern, text)

            image_tags = re.findall(image_pattern, text)
            audio_tags = re.findall(audio_pattern, text)

            if image_tags:
                assert images is not None
                assert len(image_tags) == len(image_sizes[index])
            if audio_tags:
                assert audio_phs is not None
                assert len(audio_tags) == len(audio_phs[index])

            image_id = 0
            audio_id = 0
            for i, chunk in enumerate(text_chunks):
                if chunk == image_tag:
                    image_placeholder = self.image_processor.get_slice_image_placeholder(
                        image_sizes[index][image_id], image_id, max_slice_nums, use_image_id
                    )
                    image_id += 1
                    text_chunks[i] = image_placeholder
                elif chunk == audio_tag:
                    audio_placeholder = audio_phs[index][audio_id]
                    audio_id += 1
                    text_chunks[i] = audio_placeholder

            final_text = "".join(text_chunks)
            input_ids, image_bounds, audio_bounds, spk_bounds = self._convert(final_text, max_length, **kwargs)

            input_ids_list.append(input_ids)
            image_bounds_list.append(image_bounds)
            audio_bounds_list.append(audio_bounds)
            spk_bounds_list.append(spk_bounds)

        padded_input_ids, padding_lengths = self.pad(input_ids_list, padding_side="left")
        attention_mask = torch.ones_like(padded_input_ids, dtype=torch.bool)
        for i, length in enumerate(padding_lengths):
            image_bounds_list[i] = image_bounds_list[i] + length
            audio_bounds_list[i] = audio_bounds_list[i] + length
            spk_bounds_list[i] = spk_bounds_list[i] + length
            attention_mask[i, :length] = False

        data = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "pixel_values": images,
            "image_sizes": image_sizes,
            "image_bound": image_bounds_list,
            "tgt_sizes": tgt_sizes,
            "audio_bounds": audio_bounds_list,
            "spk_bounds": spk_bounds_list,
        }

        return data

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + feature_extractor_input_names))

    def pad(self, inputs, max_length=None, padding_value=0, padding_side="left"):
        items = []
        if isinstance(inputs[0], list):
            assert isinstance(inputs[0][0], torch.Tensor)
            for it in inputs:
                for tr in it:
                    items.append(tr)
        else:
            assert isinstance(inputs[0], torch.Tensor)
            items = inputs

        batch_size = len(items)
        shape = items[0].shape
        dim = len(shape)
        assert dim <= 2
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(item.shape[-1] for item in items))
        min_length = min(item.shape[-1] for item in items)
        dtype = items[0].dtype

        if dim == 0:
            return torch.stack([item for item in items], dim=0), [0]
        elif dim == 1:
            if max_length == min_length:
                return torch.stack([item for item in items], dim=0), [0] * batch_size
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        else:
            tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

        padding_length = []
        for i, item in enumerate(items):
            if dim == 1:
                if padding_side == "left":
                    tensor[i, -len(item) :] = item.clone()
                else:
                    tensor[i, : len(item)] = item.clone()
            elif dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item) :, :] = item.clone()
                else:
                    tensor[i, : len(item), :] = item.clone()
            padding_length.append(tensor.shape[-1] - len(item))

        return tensor, padding_length


class MelSpectrogramFeatures(torch.nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding: Literal["center", "same"] = "center",
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: Tensor([num_channels, num_samples])
        """
        return super().__call__(audio)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: Tensor([num_channels, num_samples])
        """
        mel: torch.Tensor = self.mel_spec(audio)
        features = torch.log(torch.clip(mel, min=1e-5))
        return features


class ChatTTSProcessor:
    def __init__(self, text_tokenizer):
        self.audio_processor = MelSpectrogramFeatures()
        self.text_tokenizer = text_tokenizer

    def __call__(self, text_list, audio_list):
        assert len(text_list) == len(audio_list)
        input_ids_varlen = []
        for text in text_list:
            input_ids_ = self.text_tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)  # [1, seq_len]
            input_ids_ = input_ids_.squeeze(0)  # [seq_len]
            input_ids_varlen.append(input_ids_)

        audio_features_varlen = []
        for audio in audio_list:
            assert audio.shape.__len__() == 1  # [seq_len]
            try:
                mel = self.audio_processor(audio)  # [100(num_mel_bins), seq_len_mel]
            except Exception as e:
                raise e
            audio_features_varlen.append(mel)

        return {
            "tts_input_ids_varlen": input_ids_varlen,  # return List[Tensor]
            "tts_input_features_varlen": audio_features_varlen,  # return List[Tensor]
        }


def is_silent(data):
    if np.abs(data).max() < 3e-3:
        return True
    else:
        return False


def sentence_end(txt):
    for c in [".", "。", "!", "?", "！", "？"]:
        if c in txt:
            if c == ".":  # check not number before it like 1.
                idx = txt.find(c)
                if idx > 0:
                    if txt[idx - 1].isdigit():
                        continue
            return c
    return ""


class NumberToTextConverter:
    r"""
    A helper class to ensure text-to-speech (TTS) systems read numeric digits
    in the desired language (Chinese or English) digit-by-digit. It forcibly
    replaces all numeric substrings in text with their language-specific
    textual representations, thereby reducing the likelihood of TTS mistakes
    on numbers.
    Note: MiniCPM-o 2.6 only use this in streaming mode.

    Attributes:
        num_to_chinese (dict):
            Mapping from digit (str) to its Chinese textual form (str).
        num_to_english (dict):
            Mapping from digit (str) to its English textual form (str).

    Example:
        >>> converter = NumberToTextConverter()
        >>> converter.replace_numbers_with_text("我有2个苹果", language="chinese")
        '我有两个苹果'
        >>> converter.replace_numbers_with_text("I have 23 books", language="english")
        'I have two three books'
    """

    def __init__(self):
        self.num_to_chinese = {
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        }
        self.num_to_english = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
        }

    def number_to_chinese_digit_by_digit(self, num_str):
        result = ""
        for char in num_str:
            if char in self.num_to_chinese:
                result += self.num_to_chinese[char]
        return result

    def number_to_english_digit_by_digit(self, num_str):
        result = []
        for char in num_str:
            if char in self.num_to_english:
                result.append(self.num_to_english[char])
        return " ".join(result)

    def detect_language(self, text):
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_count = len(re.findall(r"[a-zA-Z]", text))
        return "chinese" if chinese_count >= english_count else "english"

    def replace_numbers_with_text(self, text, language=None):
        if language is None:
            language = self.detect_language(text)
        numbers = re.findall(r"\d+", text)

        for num in numbers:
            if language == "chinese":
                replacement = self.number_to_chinese_digit_by_digit(num)
            else:
                replacement = self.number_to_english_digit_by_digit(num)
            text = text.replace(num, replacement, 1)

        return text


class VoiceChecker:
    r"""
    A simple utility class to detect silence or low variation in consecutive audio chunks by comparing
    the mel-spectrogram distances. It keeps track of consecutive zero-distance and low-distance chunks
    to decide if the audio is considered "bad" (e.g., overly silent or not changing enough).

    Attributes:
        previous_mel (`np.ndarray` or `None`):
            Holds the previously observed mel-spectrogram in decibel scale. Used to compute
            the next distance; reset via :meth:`reset`.
        consecutive_zeros (`int`):
            The number of consecutive chunks that were detected as silent (distance = 0).
        consecutive_low_distance (`int`):
            The number of consecutive chunks whose distance was below the threshold.

    Example:
        >>> checker = VoiceChecker()
        >>> # Suppose we have audio_wav (list or np.ndarray) and mel_spec (np.ndarray)
        >>> # We split them into chunks and call checker.is_bad(...)
        >>> is_audio_bad = checker.is_bad(audio_wav, mel_spec, chunk_size=2560, thresh=100.0)
        >>> if is_audio_bad:
        ...     print("Audio deemed bad!")
        >>> # Reset states if needed
        >>> checker.reset()
    """

    def __init__(self):
        self.previous_mel = None
        self.consecutive_zeros = 0
        self.consecutive_low_distance = 0

    def compute_distance(self, audio_chunk, mel_spec):
        if is_silent(audio_chunk):
            return 0.0  # 检查是否为空白片段

        mel_db = librosa.power_to_db(mel_spec)
        if self.previous_mel is None:
            self.previous_mel = mel_db
            return -1.0

        distance = np.linalg.norm(np.mean(mel_db, axis=1) - np.mean(self.previous_mel, axis=1))
        self.previous_mel = mel_db
        return distance

    def is_bad(self, audio_wav, mel_spec, chunk_size=2560, thresh=100.0):
        num_chunks = len(audio_wav) // chunk_size
        mel_chunk_size = mel_spec.shape[-1] // num_chunks
        for i in range(num_chunks):
            audio_chunk = audio_wav[i * chunk_size : (i + 1) * chunk_size]
            mel_spec_chunk = mel_spec[:, i * mel_chunk_size : (i + 1) * mel_chunk_size]

            distance = self.compute_distance(audio_chunk, mel_spec_chunk)
            logger.warning(
                f"mel dist: {distance:.1f}, zero: {self.consecutive_zeros}, low: {self.consecutive_low_distance}"
            )
            if distance == 0:
                self.consecutive_low_distance = 0  # reset
                self.consecutive_zeros += 1
                if self.consecutive_zeros >= 12:
                    logger.warning("VoiceChecker detected 1.2 s silent. Marking as failed.")
                    return True
            elif distance < thresh:
                self.consecutive_zeros = 0
                self.consecutive_low_distance += 1
                if self.consecutive_low_distance >= 5:
                    logger.warning("VoiceChecker detected 5 consecutive low distance chunks. Marking as failed.")
                    return True
            else:
                self.consecutive_low_distance = 0
                self.consecutive_zeros = 0

        return False

    def reset(self):
        self.previous_mel = None
        self.consecutive_zeros = 0
        self.consecutive_low_distance = 0


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)


class MiniCPMVImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(self, max_slice_nums=9, scale_resolution=448, patch_size=14, **kwargs):
        super().__init__(**kwargs)
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
        self.patch_size = patch_size
        self.use_image_id = kwargs.pop("use_image_id", False)
        self.image_feature_size = kwargs.pop("image_feature_size", 64)
        self.im_start_token = kwargs.pop("im_start", "<image>")
        self.im_end_token = kwargs.pop("im_end", "</image>")
        self.slice_start_token = kwargs.pop("slice_start", "<slice>")
        self.slice_end_token = kwargs.pop("slice_end", "</slice>")
        self.unk_token = kwargs.pop("unk", "<unk>")
        self.im_id_start = kwargs.pop("im_id_start", "<image_id>")
        self.im_id_end = kwargs.pop("im_id_end", "</image_id>")
        self.slice_mode = kwargs.pop("slice_mode", True)

        self.mean = np.array(kwargs.pop("norm_mean", [0.5, 0.5, 0.5]))
        self.std = np.array(kwargs.pop("norm_std", [0.5, 0.5, 0.5]))
        self.version = kwargs.pop("version", 2.0)

    def ensure_divide(self, length, patch_size):
        return max(round(length / patch_size) * patch_size, patch_size)

    def find_best_resize(self, original_size, scale_resolution, patch_size, allow_upscale=False):
        width, height = original_size
        if (width * height > scale_resolution * scale_resolution) or allow_upscale:
            r = width / height
            height = int(scale_resolution / math.sqrt(r))
            width = int(height * r)
        best_width = self.ensure_divide(width, patch_size)
        best_height = self.ensure_divide(height, patch_size)
        return (best_width, best_height)

    def get_refine_size(self, original_size, grid, scale_resolution, patch_size, allow_upscale=False):
        width, height = original_size
        grid_x, grid_y = grid

        refine_width = self.ensure_divide(width, grid_x)
        refine_height = self.ensure_divide(height, grid_y)

        grid_width = refine_width / grid_x
        grid_height = refine_height / grid_y

        best_grid_size = self.find_best_resize(
            (grid_width, grid_height), scale_resolution, patch_size, allow_upscale=allow_upscale
        )
        refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)
        return refine_size

    def split_to_patches(self, image, grid):
        patches = []
        width, height = image.size
        grid_x = int(width / grid[0])
        grid_y = int(height / grid[1])
        for i in range(0, height, grid_y):
            images = []
            for j in range(0, width, grid_x):
                box = (j, i, j + grid_x, i + grid_y)
                patch = image.crop(box)
                images.append(patch)
            patches.append(images)
        return patches

    def slice_image(self, image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False):
        original_size = image.size
        source_image = None
        best_grid = self.get_sliced_grid(original_size, max_slice_nums, never_split)
        patches = []

        if best_grid is None:
            # dont need to slice, upsample
            best_size = self.find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=True)
            source_image = image.resize(best_size, resample=Image.Resampling.BICUBIC)
        else:
            # source image, down-sampling and ensure divided by patch_size
            best_resize = self.find_best_resize(original_size, scale_resolution, patch_size)
            source_image = image.copy().resize(best_resize, resample=Image.Resampling.BICUBIC)
            refine_size = self.get_refine_size(
                original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
            )
            refine_image = image.resize(refine_size, resample=Image.Resampling.BICUBIC)
            patches = self.split_to_patches(refine_image, best_grid)

        return source_image, patches, best_grid

    def get_grid_placeholder(self, grid):
        if grid is None:
            return ""
        slice_image_placeholder = (
            self.slice_start_token + self.unk_token * self.image_feature_size + self.slice_end_token
        )

        cols = grid[0]
        rows = grid[1]
        slices = []
        for i in range(rows):
            lines = []
            for j in range(cols):
                lines.append(slice_image_placeholder)
            slices.append("".join(lines))

        slice_placeholder = "\n".join(slices)
        return slice_placeholder

    def get_image_id_placeholder(self, idx=0):
        return f"{self.im_id_start}{idx}{self.im_id_end}"

    def get_sliced_images(self, image, max_slice_nums=None):
        slice_images = []

        if not self.slice_mode:
            return [image]

        max_slice_nums = self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        assert max_slice_nums > 0
        source_image, patches, sliced_grid = self.slice_image(
            image, max_slice_nums, self.scale_resolution, self.patch_size  # default: 9  # default: 448  # default: 14
        )

        slice_images.append(source_image)
        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    slice_images.append(patches[i][j])
        return slice_images

    def get_sliced_grid(self, image_size, max_slice_nums, nerver_split=False):
        original_width, original_height = image_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (self.scale_resolution * self.scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1 or nerver_split:
            return None
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        candidate_grids = []
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        return best_grid

    def get_slice_image_placeholder(self, image_size, image_idx=0, max_slice_nums=None, use_image_id=None):
        max_slice_nums = self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        assert max_slice_nums > 0
        grid = self.get_sliced_grid(image_size=image_size, max_slice_nums=max_slice_nums)

        image_placeholder = self.im_start_token + self.unk_token * self.image_feature_size + self.im_end_token
        use_image_id = self.use_image_id if use_image_id is None else bool(use_image_id)
        if use_image_id:
            final_placeholder = self.get_image_id_placeholder(image_idx) + image_placeholder
        else:
            final_placeholder = image_placeholder

        if self.slice_mode:
            final_placeholder = final_placeholder + self.get_grid_placeholder(grid=grid)
        return final_placeholder

    def to_pil_image(self, image, rescale=None) -> PIL.Image.Image:
        """
        Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
        needed.

        Args:
            image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`):
                The image to convert to the PIL Image format.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will
                default to `True` if the image type is a floating type, `False` otherwise.
        """
        if isinstance(image, PIL.Image.Image):
            return image
        if is_torch_tensor(image):
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if rescale is None:
                # rescale default to the array being of floating type.
                rescale = isinstance(image.flat[0], np.floating)
            # If the channel as been moved to first dim, we put it back at the end.
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if rescale:
                image = image * 255
            image = image.astype(np.uint8)
            return PIL.Image.fromarray(image)
        return image

    def reshape_by_patch(self, image):
        """
        :param image: shape [3, H, W]
        :param patch_size:
        :return: [3, patch_size, HW/patch_size]
        """
        image = torch.from_numpy(image)
        patch_size = self.patch_size
        patches = torch.nn.functional.unfold(image, (patch_size, patch_size), stride=(patch_size, patch_size))

        patches = patches.reshape(image.size(0), patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(image.size(0), patch_size, -1)
        return patches.numpy()

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]],
        do_pad: Optional[bool] = True,
        max_slice_nums: int = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> MiniCPMOBatchFeature:
        if isinstance(images, Image.Image):
            images_list = [[images]]
        elif isinstance(images[0], Image.Image):
            images_list = [images]
        else:
            images_list = images

        new_images_list = []
        image_sizes_list = []
        tgt_sizes_list = []

        for _images in images_list:
            if _images is None or len(_images) == 0:
                new_images_list.append([])
                image_sizes_list.append([])
                tgt_sizes_list.append([])
                continue
            if not valid_images(_images):
                raise ValueError(
                    "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                    "torch.Tensor, tf.Tensor or jax.ndarray."
                )

            _images = [self.to_pil_image(image).convert("RGB") for image in _images]
            input_data_format = infer_channel_dimension_format(np.array(_images[0]))

            new_images = []
            image_sizes = [image.size for image in _images]
            tgt_sizes = []
            for image in _images:
                image_patches = self.get_sliced_images(image, max_slice_nums)
                image_patches = [to_numpy_array(image).astype(np.float32) / 255 for image in image_patches]
                image_patches = [
                    self.normalize(image=image, mean=self.mean, std=self.std, input_data_format=input_data_format)
                    for image in image_patches
                ]
                image_patches = [
                    to_channel_dimension_format(image, ChannelDimension.FIRST, input_channel_dim=input_data_format)
                    for image in image_patches
                ]
                for slice_image in image_patches:
                    new_images.append(self.reshape_by_patch(slice_image))
                    tgt_sizes.append(
                        np.array((slice_image.shape[1] // self.patch_size, slice_image.shape[2] // self.patch_size))
                    )

            if tgt_sizes:
                tgt_sizes = np.vstack(tgt_sizes)

            new_images_list.append(new_images)
            image_sizes_list.append(image_sizes)
            tgt_sizes_list.append(tgt_sizes)
        return MiniCPMOBatchFeature(
            data={"pixel_values": new_images_list, "image_sizes": image_sizes_list, "tgt_sizes": tgt_sizes_list},
            tensor_type=return_tensors,
        )


AutoImageProcessor.register("MiniCPMVImageProcessor", MiniCPMVImageProcessor)


__all__ = ["MiniCPM_o_2_6Processor"]
