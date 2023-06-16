# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
from ...processing_utils import ProcessorMixin
from ..auto import AutoTokenizer
from typing import Optional, Tuple
import warnings
from transformers.utils.hub import cached_file
import numpy as np


class BarkProcessor(ProcessorMixin):
    r"""
    Constructs a Bark processor which wraps a Bark voice preset and a text tokenizer into a single
    processor.

    Args:
        tokenizer ([`PreTrainedTokenizer`], *optional*):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is optional. 
            By default, it will instantiated as "bert-base-multilingual-cased".
        repo_id (`str`, *optional*, defaults to ""):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
    """
    tokenizer_class = "AutoTokenizer"
    attributes = ["tokenizer"]
    
    preset_shape = {
        "semantic_prompt" : 1,
        "coarse_prompt" : 2,
        "fine_prompt" : 2,
    }

    def __init__(self, tokenizer: Optional[AutoTokenizer] = None, repo_id = "", subfolder = "",
                 **kwargs):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        super().__init__(tokenizer)
        
        
        self.repo_id = repo_id
        self.subfolder = subfolder


    @classmethod
    def from_pretrained(cls, pretrained_tokenizer_name_or_path, repo_id = None, subfolder = "speaker_embeddings", **kwargs):
        """Same initialization than __init__ except that you can specify the tokenizer name or path instead of passing a 
        [`PreTrainedTokenizer`].
        """

        tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path, **kwargs)
        

        return cls(tokenizer=tokenizer,
                    repo_id = repo_id,
                    subfolder = subfolder,
                    **kwargs)
        
    def _validate_voice_preset_dict(cls, voice_preset: Optional[dict] = None):
        if voice_preset is None:
            return
        for key in ["semantic_prompt", "coarse_prompt", "fine_prompt"]:
            if key not in voice_preset:
                raise ValueError(f"Voice preset unrecognized, missing {key} as a key.")
            
            if not isinstance(voice_preset[key], np.ndarray):
                raise ValueError(f"{key} voice preset must be a {str(cls.preset_shape[key])}D ndarray.")
            
            if len(voice_preset[key].shape) != cls.preset_shape[key]:
                raise ValueError(f"{key} voice preset must be a {str(cls.preset_shape[key])}D ndarray.")
        
    def __call__(self, text=None, voice_preset=None, return_tensors="pt", max_length = 256, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s). This method forwards the `text`
        and `kwargs` arguments to the AutoTokenizer's [`~AutoTokenizer.__call__`] to encode the text. 
        The method also proposes a voice preset which is a dictionary of arrays that conditions `Bark`'s output.  
        `kwargs` arguments are forwarded to the tokenizer and to `cached_file` if `voice_preset` 
        ... TODO

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            Tuple...
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
        """
        if voice_preset is None or isinstance(voice_preset, dict):
            pass            
        else:
            # if not dict, either 
            if isinstance(voice_preset, str) and not voice_preset.endswith(".npz"):
                voice_preset = voice_preset + ".npz"
            
            file_path = cached_file(self.repo_id, voice_preset, subfolder = self.subfolder,
                cache_dir=kwargs.pop("cache_dir", None),
                force_download=kwargs.pop("force_download", False),
                proxies=kwargs.pop("proxies", None),
                resume_download=kwargs.pop("resume_download", False),
                local_files_only=kwargs.pop("local_files_only", False),
                use_auth_token=kwargs.pop("use_auth_token", None),
                user_agent=kwargs.pop("user_agent", None),
                revision=kwargs.pop("revision", None),
                repo_type=kwargs.pop("repo_type", None),
            )            
            voice_preset = np.load(file_path)
            
                    
        self._validate_voice_preset_dict(voice_preset)
        encoded_text = self.tokenizer(text, return_tensors=return_tensors, padding = "max_length"
                                      , max_length = max_length, return_attention_mask = True, return_token_type_ids = False,
                                      **kwargs)    
        
        return encoded_text, voice_preset
