# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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


import datetime
import gc
import math
import unittest

from transformers import CrystalCoderConfig, is_torch_available
from transformers.testing_utils import backend_empty_cache, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin
from transformers import AutoTokenizer, AutoModelForCausalLM

if is_torch_available():
    import torch

    from transformers import (
        # CRYSTALCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
        CrystalCoderLMHeadModel,
        # CrystalCoderModel,
        CrystalCoderTokenizerFast,
    )



@require_torch
class CrystalCoderModelLanguageGenerationTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        gc.collect()
        backend_empty_cache(torch_device)


    @slow
    def test_lm_generate_crystalchat(self):
        
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch_device
        tokenizer = CrystalCoderTokenizerFast.from_pretrained("LLM360/CrystalChat")
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", trust_remote_code=True)
        model = CrystalCoderLMHeadModel.from_pretrained("LLM360/CrystalChat").to(device)

        prompt = '<s> <|sys_start|> You are an AI assistant. You will be given a task. You must generate a detailed and long answer. <|sys_end|> <|im_start|> Write a python function that takes a list of integers and returns the squared sum of the list. <|im_end|>'


        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gen_tokens = model.generate(input_ids, do_sample=True, max_length=120)

        print("-"*20 + "Output for model"  + 20 * '-')

    def test_lm_generate_random(self):
        
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = torch_device
        tokenizer = AutoTokenizer.from_pretrained("LLM360/CrystalChat")
        # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(CrystalCoderConfig(n_layer=4, n_embd=64)).to(device)

        prompt = '<s> <|sys_start|> You are an AI assistant. You will be given a task. You must generate a detailed and long answer. <|sys_end|> <|im_start|> Write a python function that takes a list of integers and returns the squared sum of the list. <|im_end|>'


        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gen_tokens = model.generate(input_ids, do_sample=True, max_length=120)

        print("-"*20 + "Output for model"  + 20 * '-')
        