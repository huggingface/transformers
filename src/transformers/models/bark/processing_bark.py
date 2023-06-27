# coding=utf-8
# Copyright 2023 The Suno AI Authors and The HuggingFace Inc. team. All rights reserved.
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
Processor class for Bark
"""
import os
from typing import Optional, Union

import numpy as np


from ..auto import AutoTokenizer

from ...processing_utils import ProcessorMixin

from ...utils import (
    TensorType,
    is_flax_available,
    is_jax_tensor,
    is_numpy_array,
    is_tf_available,
    is_torch_available,
)
from ...utils.hub import get_file_from_repo


class BarkProcessor(ProcessorMixin):
    r"""
    Constructs a Bark processor which wraps a Bark voice preset and a text tokenizer into a single processor.

    Args:
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`].
        speaker_embeddings_dict (`Dict[np.ndarray]`, *optional*, defaults to `None`):
            Optional speaker embeddings dictionary. The keys follow the following pattern:
            `"{voice_preset_name}_{prompt_key}"`. For example: `"en_speaker_1_semantic_prompt"` or
            `"en_speaker_1_coarse_prompt"`.
    """
    tokenizer_class = "AutoTokenizer"
    attributes = ["tokenizer"]

    preset_shape = {
        "semantic_prompt": 1,
        "coarse_prompt": 2,
        "fine_prompt": 2,
    }

    def __init__(self, tokenizer, speaker_embeddings_dict=None):
        super().__init__(tokenizer)

        self.speaker_embeddings_dict = speaker_embeddings_dict

    @classmethod
    def from_pretrained(
        cls, pretrained_processor_name_or_path, speaker_embeddings_file_name="speaker_embeddings.npz", **kwargs
    ):
        r"""
        Args:
        Instantiate a Bark processor associated with a pretrained model.
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained [`BarkProcessor`] hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a processor saved using the [`~BarkProcessor.save_pretrained`]
                  method, e.g., `./my_model_directory/`.
            speaker_embeddings_file_name (`str`, *optional*, defaults to `"speaker_embeddings.npz"`):
                The name of the `.npz` file containing the speaker_embeddings (e.g `"speaker_embeddings.npz"`) located
                in `pretrained_model_name_or_path`. If `None`, no speaker_embeddings is loaded.
            **kwargs
                Additional keyword arguments passed along to both
                [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
        """

        if speaker_embeddings_file_name is not None:
            speaker_embeddings_dict = get_file_from_repo(
                pretrained_processor_name_or_path,
                speaker_embeddings_file_name,
                subfolder=kwargs.pop("subfolder", None),
                cache_dir=kwargs.pop("cache_dir", None),
                force_download=kwargs.pop("force_download", False),
                proxies=kwargs.pop("proxies", None),
                resume_download=kwargs.pop("resume_download", False),
                local_files_only=kwargs.pop("local_files_only", False),
                use_auth_token=kwargs.pop("use_auth_token", None),
                revision=kwargs.pop("revision", None),
            )

            speaker_embeddings_dict = np.load(speaker_embeddings_dict)
        else:
            speaker_embeddings_dict = None

        tokenizer = AutoTokenizer.from_pretrained(pretrained_processor_name_or_path, **kwargs)

        return cls(tokenizer=tokenizer, speaker_embeddings_dict=speaker_embeddings_dict)

    def save_pretrained(
        self,
        save_directory,
        speaker_embeddings_file_name="speaker_embeddings.npz",
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Saves the attributes of this processor (tokenizer...) in the specified directory so that it can be reloaded
        using the [`~BarkProcessor.from_pretrained`] method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
            speaker_embeddings_file_name (`str`, *optional*, defaults to `"speaker_embeddings.npz"`):
                The name of the `.npz` file that will contains the speaker_embeddings, if it exists, and that will be
                located in `pretrained_model_name_or_path`.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if self.speaker_embeddings_dict is not None:
            os.makedirs(save_directory, exist_ok=True)
            np.savez(os.path.join(save_directory, speaker_embeddings_file_name), **self.speaker_embeddings_dict)

        super().save_pretrained(save_directory, push_to_hub, **kwargs)

    def _validate_voice_preset_dict(self, voice_preset: Optional[dict] = None):
        if voice_preset is None:
            return
        for key in ["semantic_prompt", "coarse_prompt", "fine_prompt"]:
            if key not in voice_preset:
                raise ValueError(f"Voice preset unrecognized, missing {key} as a key.")

            if not isinstance(voice_preset[key], np.ndarray):
                raise ValueError(f"{key} voice preset must be a {str(self.preset_shape[key])}D ndarray.")

            if len(voice_preset[key].shape) != self.preset_shape[key]:
                raise ValueError(f"{key} voice preset must be a {str(self.preset_shape[key])}D ndarray.")

    def __call__(
        self,
        text=None,
        voice_preset=None,
        return_tensors="pt",
        max_length=256,
        add_special_tokens=False,
        return_attention_mask=True,
        return_token_type_ids=False,
        **kwargs,
    ):
        """
        Main method to prepare for the model one or several sequences(s). This method forwards the `text` and `kwargs`
        arguments to the AutoTokenizer's [`~AutoTokenizer.__call__`] to encode the text. The method also proposes a
        voice preset which is a dictionary of arrays that conditions `Bark`'s output. `kwargs` arguments are forwarded
        to the tokenizer and to `cached_file` method if `voice_preset` is a valid filename.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            voice_preset (`str`, `Dict[np.ndarray]`):
                The voice preset, i.e the speaker embeddings. It can either be a valid voice_preset name, e.g
                `"en_speaker_1"`, or a directly a dictionnary of embeddings for each submodel of `Bark`. Or it can be a
                valid file name of a local `.npz` single voice preset.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            Tuple([`BatchEncoding`], Dict[~utils.TensorType]): A tuple composed of a [`BatchEncoding`], i.e the output
            of the `tokenizer` and a `Dict[~utils.TensorType`]`, i.e the voice preset with the right tensors type.
        """
        if voice_preset is None or isinstance(voice_preset, dict):
            pass
        else:
            voice_preset_key = voice_preset.replace("/", "_")
            if isinstance(voice_preset, str) and f"{voice_preset_key}_semantic_prompt" in self.speaker_embeddings_dict:
                voice_preset = {}
                for prompt_type in [
                    "semantic_prompt",
                    "coarse_prompt",
                    "fine_prompt",
                ]:
                    voice_preset[prompt_type] = self.speaker_embeddings_dict[f"{voice_preset_key}_{prompt_type}"]

            else:
                if isinstance(voice_preset, str) and not voice_preset.endswith(".npz"):
                    voice_preset = voice_preset + ".npz"

                voice_preset = np.load(voice_preset)

        self._validate_voice_preset_dict(voice_preset)
        voice_preset = convert_dict_to_tensors(voice_preset, return_tensors)
        encoded_text = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

        return encoded_text, voice_preset


def convert_dict_to_tensors(tensor_dict, tensor_type: Optional[Union[str, TensorType]] = None):
    if tensor_type is None or tensor_dict is None:
        return tensor_dict

    # Convert to TensorType
    if not isinstance(tensor_type, TensorType):
        tensor_type = TensorType(tensor_type)

    # Get a function reference for the correct framework
    if tensor_type == TensorType.TENSORFLOW:
        if not is_tf_available():
            raise ImportError("Unable to convert output to TensorFlow tensors format, TensorFlow is not installed.")
        import tensorflow as tf

        as_tensor = tf.constant
        is_tensor = tf.is_tensor
    elif tensor_type == TensorType.PYTORCH:
        if not is_torch_available():
            raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
        import torch

        as_tensor = torch.tensor
        is_tensor = torch.is_tensor
    elif tensor_type == TensorType.JAX:
        if not is_flax_available():
            raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
        import jax.numpy as jnp  # noqa: F811

        as_tensor = jnp.array
        is_tensor = is_jax_tensor
    else:

        def as_tensor(value, dtype=None):
            if isinstance(value, (list, tuple)) and isinstance(value[0], (list, tuple, np.ndarray)):
                value_lens = [len(val) for val in value]
                if len(set(value_lens)) > 1 and dtype is None:
                    # we have a ragged list so handle explicitly
                    value = as_tensor([np.asarray(val) for val in value], dtype=object)
            return np.asarray(value, dtype=dtype)

        is_tensor = is_numpy_array

    new_tensor_dict = {}
    # Do the tensor conversion in batch
    for key, value in tensor_dict.items():
        try:
            if not is_tensor(value):
                tensor = as_tensor(value)

                new_tensor_dict[key] = tensor
        except Exception as e:
            if key == "overflowing_tokens":
                raise ValueError(
                    "Unable to create tensor returning overflowing tokens of different lengths. "
                    "Please see if a fast version of this tokenizer is available to have this feature available."
                ) from e
            raise ValueError(
                "Unable to create tensor, you should probably activate truncation and/or padding with"
                " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"
                f" features (`{key}` in this case) have excessive nesting (inputs type `list` where type `int` is"
                " expected)."
            ) from e

    return new_tensor_dict
