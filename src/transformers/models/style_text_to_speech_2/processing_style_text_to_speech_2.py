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
from typing import Optional, List, Union

import numpy as np
import torch

from ...tokenization_utils_base import AudioInput, BatchEncoding, PreTokenizedInput, TextInput
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...utils import logging
from ...utils.hub import get_file_from_repo
from ..auto import AutoTokenizer


logger = logging.get_logger(__name__)



class StyleTextToSpeech2Processor(ProcessorMixin):

    attributes = ["tokenizer"]

    def __init__(self, tokenizer, ):
        super().__init__(tokenizer)

    def _load_voice_preset(
        self, 
        voice_preset: Optional[str, List[str]] = None,
    ):
        
        
        
        
        

    def __call__(
        self, 
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        voice_preset: Optional[Union[str, List[str], torch.Tensor]] = None, 
        **tokenizer_kwargs
    ):
        inputs = self.tokenizer(text, **tokenizer_kwargs)



        style = torch.load(voices, weights_only=True)
        style = style.repeat(len(text), 1)
        inputs["style"] = style
        return inputs
    

        

__all__ = ["StyleTextToSpeech2Processor"]
