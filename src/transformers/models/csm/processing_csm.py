# coding=utf-8
# Copyright 2025 Sesame and The HuggingFace Inc. team. All rights reserved.
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
from typing import Any, Dict, Optional

from ...utils import is_torch_available


if is_torch_available():
    import torch

from ...audio_utils import make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack


class CsmAudioKwargs(AudioKwargs, total=False):
    encoded_length_kwargs: Optional[Dict[str, Any]]


class CsmProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: CsmAudioKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "encoded_length_kwargs": {
                "kernel_sizes": [7, 3, 1, 8, 3, 1, 10, 3, 1, 12, 3, 1, 16, 3, 4],
                "strides": [1, 1, 1, 4, 1, 1, 5, 1, 1, 6, 1, 1, 8, 1, 2],
                "dilations": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "use_causal_conv": True,
            },
            "sampling_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class CsmProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    feature_extractor_class = "EncodecFeatureExtractor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
    ):
        if not hasattr(tokenizer, "audio_token"):
            self.audio_token = "<|AUDIO|>"
            self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        else:
            self.audio_token = tokenizer.audio_token
            self.audio_token_id = tokenizer.audio_token_id

        if not hasattr(tokenizer, "audio_eos_token"):
            self.audio_eos_token = "<|audio_eos|>"
            self.audio_eos_token_id = tokenizer.convert_tokens_to_ids(self.audio_eos_token)
        else:
            self.audio_eos_token = tokenizer.audio_eos_token
            self.audio_eos_token_id = tokenizer.audio_eos_token_id

        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @staticmethod
    def _get_encoded_length(audio_length, kernel_sizes=None, strides=None, dilations=None, use_causal_conv=None):
        cur_length = audio_length

        if kernel_sizes is None or strides is None or dilations is None or use_causal_conv is None:
            return cur_length

        for kernel_size, stride, dilation in zip(kernel_sizes, strides, dilations):
            effective_kernel_size = (kernel_size - 1) * dilation + 1
            padding_total = kernel_size - stride
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right

            n_frames = (cur_length - effective_kernel_size + padding_total) / stride + 1
            n_frames = math.ceil(n_frames) - 1
            ideal_length = n_frames * stride + kernel_size - padding_total
            extra_padding = ideal_length - cur_length

            if use_causal_conv:
                padding_left = padding_total
                padding_right = extra_padding
            else:
                padding_left = padding_left
                padding_right = padding_right + extra_padding

            cur_length = cur_length + padding_left + padding_right
            cur_length = (cur_length - dilation * (kernel_size - 1) - 1) // stride + 1

        return cur_length

    def __call__(
        self,
        text,
        audio=None,
        output_labels=False,
        depth_decoder_labels_ratio=1.0,
        **kwargs: Unpack[CsmProcessorKwargs],
    ):
        output_kwargs = self._merge_kwargs(
            CsmProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        return_tensors = common_kwargs.pop("return_tensors", None)
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
            encoded_length_kwargs = audio_kwargs.pop("encoded_length_kwargs", {})
            num_audio_tokens_list = [
                self._get_encoded_length(audio_array.shape[-1], **encoded_length_kwargs) for audio_array in audio
            ]
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

        data = {}
        encoding = self.tokenizer(text, **text_kwargs)
        data.update(encoding)

        if audio is not None:
            audio_kwargs.pop("return_attention_mask", None)  # not supported by the feature extractor
            audio_inputs = self.feature_extractor(audio, **audio_kwargs)
            audio_inputs["input_values_mask"] = audio_inputs.pop("padding_mask")
            data.update(audio_inputs)

        if output_labels:
            audio_frame_idxs = (data["input_ids"] == self.audio_token_id).nonzero()
            n_audio_frames = audio_frame_idxs.shape[0]

            if depth_decoder_labels_ratio <= 1.0:
                rand_idxs = torch.randperm(n_audio_frames)[: int(n_audio_frames * (1 - depth_decoder_labels_ratio))]
                skip_frames_idxs = audio_frame_idxs[rand_idxs]
            else:
                skip_frames_idxs = audio_frame_idxs

            labels = torch.where(data["input_ids"] == self.audio_token_id, data["input_ids"], -100)
            labels[skip_frames_idxs[:, 0], skip_frames_idxs[:, 1]] = -101

            data["labels"] = labels

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["CsmProcessor"]
