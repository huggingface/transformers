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
from typing import Any, Optional, Union

from ...utils import is_soundfile_available, is_torch_available


if is_torch_available():
    import torch

if is_soundfile_available():
    pass

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class VoxtralAudioKwargs(AudioKwargs, total=False):
    encoded_length_kwargs: Optional[dict[str, Any]]


class VoxtralProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: VoxtralAudioKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "audio_length_per_tok": 8,
            "sampling_rate": 16000,
            "padding": True,
            "truncation": False,
            "pad_to_multiple_of": 480000,
            "max_source_positions": 3000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class VoxtralProcessor(ProcessorMixin):
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
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "LlamaTokenizerFast"

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
    ):
        self.audio_token = "[AUDIO]"
        self.audio_token_id = 24

        self.audio_bos_token = "[BEGIN_AUDIO]"
        self.audio_bos_token_id = 25

        self.inst_bos_token = "[INST]"
        self.inst_bos_token_id = 3

        self.inst_eos_token = "[/INST]"
        self.inst_eos_token_id = 4

        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @staticmethod
    def _get_encoded_length(audio_length, pad_to_multiple_of, max_source_positions, audio_length_per_tok):
        next_multiple_of = math.ceil(audio_length / pad_to_multiple_of)
        num_audio_tokens = next_multiple_of * math.ceil(max_source_positions / audio_length_per_tok)

        return num_audio_tokens

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]],
        audio: Optional[AudioInput] = None,
        output_labels: Optional[bool] = False,
        depth_decoder_labels_ratio: Optional[float] = 1.0,
        **kwargs: Unpack[VoxtralProcessorKwargs],
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
            VoxtralProcessorKwargs,
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

        # TODO: @eustlb, we have a clear issue of mapping with batch!?
        if audio is not None:
            pad_to_multiple_of = audio_kwargs.get("pad_to_multiple_of", 480000)
            audio_length_per_tok = audio_kwargs.pop("audio_length_per_tok", 8)
            max_source_positions = audio_kwargs.get("max_source_positions", 3000)

            num_audio_tokens_list = [
                self._get_encoded_length(
                    audio_array.shape[-1],
                    pad_to_multiple_of,
                    max_source_positions,
                    audio_length_per_tok,
                ) for audio_array in audio
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
            # TODO: @eustlb, audio_kwargs cannot be everything here... we should warn the user
            input_features_list = []
            for audio_array in audio:
                audio_inputs = self.feature_extractor(audio_array, **audio_kwargs)

                # let's split into chunks of max_source_positions, and then stack them along batch dimension
                input_features = audio_inputs["input_features"].reshape(self.feature_extractor.feature_size, -1, max_source_positions)
                input_features_list.append(input_features.transpose(0, 1))
            data["input_features"] = torch.cat(input_features_list)

        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def get_prompt_ids(self, text: str, return_tensors="np"):
        return self.tokenizer.get_prompt_ids(text, return_tensors=return_tensors)


__all__ = ["VoxtralProcessor"]

