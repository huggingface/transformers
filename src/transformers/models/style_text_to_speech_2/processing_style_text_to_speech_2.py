# coding=utf-8
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
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Union, Dict

import numpy as np
import torch

from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...processing_utils import ProcessorMixin
from ...utils import logging
from ...utils.hub import get_file_from_repo
from ..auto import AutoTokenizer


logger = logging.get_logger(__name__)

VOICE_PRESETS_FILES_NAMES = {"voice_presets_path": "voice_presets.json"}
DEFAULT_VOICE_PRESET_NAME = "af_heart"


class StyleTextToSpeech2Processor(ProcessorMixin):

    tokenizer_class = "StyleTextToSpeech2Tokenizer"
    attributes = ["tokenizer"]

    def __init__(
        self, 
        tokenizer, 
        voice_presets_config: Dict[str, Union[str, Dict[str, str]]],
    ):
        super().__init__(tokenizer)
        self.voice_presets_config = voice_presets_config

    @classmethod
    def from_pretrained(
        cls, pretrained_processor_name_or_path, voice_presets_path=None, **kwargs
    ):
        if voice_presets_path is None:
            voice_presets_path = VOICE_PRESETS_FILES_NAMES["voice_presets_path"]

        voice_presets_path = get_file_from_repo(
            pretrained_processor_name_or_path,
            voice_presets_path,
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            local_files_only=kwargs.pop("local_files_only", False),
            subfolder=kwargs.pop("subfolder", ""),
        )
        with open(voice_presets_path) as voice_presets_json:
            voice_presets_config = json.load(voice_presets_json)

        tokenizer = AutoTokenizer.from_pretrained(pretrained_processor_name_or_path, **kwargs)

        return cls(tokenizer=tokenizer, voice_presets_config=voice_presets_config)
    
    def save_pretrained(
        self,
        save_directory,
        voice_presets_path=None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        if voice_presets_path is None:
            voice_presets_path = VOICE_PRESETS_FILES_NAMES["voice_presets_path"]

        voice_presets_path = os.path.join(save_directory, voice_presets_path)
        os.makedirs(os.path.dirname(voice_presets_path), exist_ok=True)

        for voice, path in self.voice_presets_config["voice_to_path"].items():
            voice_preset = self._load_voice_tensor(voice)
            save_path = os.path.join(save_directory, path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(voice_preset, save_path)
        
        with open(voice_presets_path, "w") as fp:
            voice_presets_config = self.voice_presets_config.copy()
            voice_presets_config["path_or_repo"] = save_directory
            json.dump(self.voice_presets_config, fp)

        super().save_pretrained(save_directory, push_to_hub, **kwargs)

    def _load_voice_tensor(
        self, 
        voice_name: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> torch.Tensor:
        path_or_repo = self.voice_presets_config["path_or_repo"]
        voice_preset_path = self.voice_presets_config["voice_to_path"][voice_name]

        path = get_file_from_repo(
            path_or_repo,
            voice_preset_path,
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            token=kwargs.pop("token", None),
            revision=kwargs.pop("revision", None),
            local_files_only=kwargs.pop("local_files_only", False),
            subfolder=kwargs.pop("subfolder", ""),
        )
        voice_preset = torch.load(path)

        return voice_preset

    def __call__(
        self, 
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        voice_names: Optional[Union[str, List[str]]] = None,
        return_attention_mask = True,
        **kwargs
    ):
        inputs = self.tokenizer(
            text, 
            return_attention_mask=return_attention_mask,
            **kwargs
        )

        if "attention_mask" in inputs:
            lenghts = inputs["attention_mask"].sum(dim=1).tolist()
        else:
            logger.warning("return_attention_mask is set to False. It is assumed that all texts have the same length.")
            lenghts = [inputs["input_ids"].shape[1]] * inputs["input_ids"].shape[0]

        batch_size = inputs["input_ids"].shape[0]
        if voice_names is None:
            voice_names = [DEFAULT_VOICE_PRESET_NAME] * batch_size
        elif isinstance(voice_names, str):
            voice_names = [voice_names] * batch_size
        elif isinstance(voice_names, list):
            if len(voice_names) != batch_size:
                raise ValueError(
                    f"The provided number of voice names ({len(voice_names)}) does not match the number of provided texts ({batch_size})."
                    " Use a single voice name or provide a voice name for each text."
                ) 
        
        name_to_loaded_voice = {
            voice_name: self._load_voice_tensor(voice_name)
            for voice_name in voice_names
        }

        styles = [name_to_loaded_voice[voice_name][lenght - 1] for voice_name, lenght in zip(voice_names, lenghts)]
        styles = torch.cat(styles, dim=0)
        inputs["style"] = styles

        return inputs


__all__ = ["StyleTextToSpeech2Processor"]
