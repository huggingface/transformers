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
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ...utils import is_soundfile_available, is_torch_available


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class CsmAudioKwargs(AudioKwargs, total=False):
    encoded_length_kwargs: Optional[dict[str, Any]]


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
    r"""
    Constructs a Csm processor which wraps [`EncodecFeatureExtractor`] and
    [`PretrainedTokenizerFast`] into a single processor that inherits both the audio feature extraction and
    tokenizer functionalities. See the [`~CsmProcessor.__call__`] for more
    information.
    The preferred way of passing kwargs is as a dictionary per modality, see usage example below.
        ```python
        from transformers import CsmProcessor
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
        audio = ds[0]["audio"]["array"]

        processor = CsmProcessor.from_pretrained("sesame/csm-1b")

        processor(
            text=["<|begin_of_text|>[0]What are you working on?<|end_of_text|><|AUDIO|><|audio_eos|><|begin_of_text|>[1]I'm figuring out my budget.<|end_of_text|>"],
            audio=audio,
            text_kwargs = {"padding": False},
            audio_kwargs = {"sampling_rate": 16000},
            common_kwargs = {"return_tensors": "pt"},
        )
        # this should error out because EncodecFeatureExtractor expects a 24kHz audio :)
        ```

    Args:
        feature_extractor ([`EncodecFeatureExtractor`]):
            The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`, `PreTrainedTokenizerFast`]):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.

    """

    attributes = ["feature_extractor", "tokenizer"]
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
        """
        Compute the length of the encoded audio sequence.

        Args:
            audio_length (int): The length of the audio sequence.
            kernel_sizes (list[int]): The kernel sizes for the convolutional layers.
            strides (list[int]): The strides for the convolutional layers.
            use_causal_conv (bool): Whether to use causal convolutions.
        """
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

    def save_audio(
        self,
        audio: AudioInput,
        saving_path: Union[str, Path, list[Union[str, Path]]],
        **kwargs: Unpack[CsmProcessorKwargs],
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
            CsmProcessorKwargs,
            **kwargs,
        )
        audio_kwargs = output_kwargs["audio_kwargs"]
        sampling_rate = audio_kwargs["sampling_rate"]

        for audio_value, p in zip(audio, saving_path):
            if isinstance(audio_value, torch.Tensor):
                audio_value = audio_value.cpu().float().numpy()
            sf.write(p, audio_value, sampling_rate)

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]],
        audio: Optional[AudioInput] = None,
        output_labels: Optional[bool] = False,
        depth_decoder_labels_ratio: Optional[float] = 1.0,
        **kwargs: Unpack[CsmProcessorKwargs],
    ):
        r"""
        Main method to prepare text(s) and audio to be fed as input to the model. This method forwards the `text`
        arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] to encode
        the text. To prepare the audio, this method forwards the `audio` arguments to
        EncodecFeatureExtractor's [`~EncodecFeatureExtractor.__call__`]. Please refer
        to the docstring of the above two methods for more information.

        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The audio or batch of audio to be prepared. Each audio can be a NumPy array or PyTorch
                tensor.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            output_labels (bool, *optional*, default=False):
                Whether to return labels for training. Indices will be in `[config.audio_token_id, -100, -101]`.
                - `config.audio_token_id` indicates an audio frame (considering sequence length elements as frames)
                - `-100` will be ignored in the loss computation
                - `-101` indicates the audio frame will be used only for the backbone model (using the first codebook token as labels)
            depth_decoder_labels_ratio (float, *optional*, default=1.0):
                The ratio of audio frames to keep for the depth decoder labels.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return NumPy `np.ndarray` objects.
                    - `'jax'`: Return JAX `jnp.ndarray` objects.
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **input_values** -- List of audio values to be fed to a model. Returned when `audio` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **labels** -- List of labels for the audio frames. Returned when `output_labels=True`.
        """

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

        encoding = self.tokenizer(text, **text_kwargs)
        data = {}
        data.update(encoding)

        if audio is not None:
            audio_kwargs.pop("return_attention_mask", None)  # not supported by the feature extractor

            concatenated_audio, input_values_cutoffs = [], []
            offset = 0
            for n_audio in n_audio_in_text:
                if n_audio == 0:
                    concatenated_audio.append(np.zeros(0))
                    input_values_cutoffs.append(torch.tensor([-1]))
                else:
                    concatenated_audio.append(
                        np.concatenate(
                            [
                                el.cpu().numpy() if isinstance(el, torch.Tensor) else el
                                for el in audio[offset : offset + n_audio]
                            ],
                            axis=-1,
                        )
                    )
                    input_values_cutoffs.append(
                        torch.tensor([el.shape[-1] for el in audio[offset : offset + n_audio]]).cumsum(dim=-1)
                    )
                    offset += n_audio

            audio_inputs = self.feature_extractor(concatenated_audio, **audio_kwargs)
            audio_inputs.pop("padding_mask", None)  # not applicable here
            data.update(audio_inputs)

            # pad and stack the audio cut idxs
            max_len = max(cut_idxs.shape[-1] for cut_idxs in input_values_cutoffs)
            input_values_cutoffs = [
                torch.nn.functional.pad(cut_idxs, (0, max_len - cut_idxs.shape[-1]), value=-1)
                for cut_idxs in input_values_cutoffs
            ]
            data["input_values_cutoffs"] = torch.stack(input_values_cutoffs, dim=0)

        if output_labels:
            audio_frame_idxs = (data["input_ids"] == self.audio_token_id).nonzero()
            n_audio_frames = audio_frame_idxs.shape[0]

            if depth_decoder_labels_ratio <= 1.0:
                rand_idxs = torch.randperm(n_audio_frames)[: int(n_audio_frames * (1 - depth_decoder_labels_ratio))]
                skip_frames_idxs = audio_frame_idxs[rand_idxs]
            else:
                skip_frames_idxs = audio_frame_idxs

            labels = torch.where(
                (data["input_ids"] == self.audio_token_id) | (data["input_ids"] == self.audio_eos_token_id),
                data["input_ids"],
                -100,
            )
            labels[skip_frames_idxs[:, 0], skip_frames_idxs[:, 1]] = -101

            data["labels"] = labels

        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names

        # Remove `padding_mask`, it is popped and not used when processing. Make a copy of list when removing
        # otherwise `self.feature_extractor.model_input_names` is also modified
        feature_extractor_input_names = [name for name in feature_extractor_input_names if name != "padding_mask"]
        return list(tokenizer_input_names + feature_extractor_input_names + ["input_values_cutoffs"])


__all__ = ["CsmProcessor"]
