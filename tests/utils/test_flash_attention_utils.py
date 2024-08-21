# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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
import copy
import glob
import json
import os
import os.path
import sys
import tempfile
import threading
import unittest
import unittest.mock as mock
import uuid
from pathlib import Path

import requests
from huggingface_hub import HfApi, HfFolder, delete_repo
from requests.exceptions import HTTPError

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    OwlViTForObjectDetection,
    PretrainedConfig,
    is_torch_available,
    logging,
)
from transformers.testing_utils import (
    TOKEN,
    USER,
    CaptureLogger,
    LoggingLevel,
    TestCasePlus,
    is_staging_test,
    require_accelerate,
    require_flax,
    require_safetensors,
    require_tf,
    require_torch,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    require_usr_bin_time,
    slow,
    torch_device,
)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from transformers.utils.import_utils import (
    is_flash_attn_2_available,
    is_flax_available,
    is_tf_available,
    is_torch_sdpa_available,
    is_torchdynamo_available,
)


sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig, NoSuperInitConfig  # noqa E402

if is_torch_available():
    import torch
    
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

LOG = logging.get_logger(__name__)

@require_torch
class TestFlashAttentionForward(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda:0')
        self.dtype = torch.float16

        self.total_length = 16
        self.lengths = [3, 6, 7]
        assert sum(self.lengths) == self.total_length

        self.head_num = 4
        self.head_dim = 32

    def test_flash_attention_forward_padding_matches_padding_free_with_position_ids(self):
        query_states_padfree, key_states_padfree, value_states_padfree = torch.rand(1, self.total_length, self.head_num, self.head_dim, dtype=self.dtype, device=self.device), torch.rand(1, self.total_length, self.head_num, self.head_dim, dtype=self.dtype, device=self.device), torch.rand(1, self.total_length, self.head_num, self.head_dim, dtype=self.dtype, device=self.device)
        position_ids_padfree = torch.cat([torch.arange(length) for length in self.lengths]).long().unsqueeze(0).to(self.device)

        query_states_tuple = query_states_padfree.split(self.lengths, dim=1)
        key_states_tuple = key_states_padfree.split(self.lengths, dim=1)
        value_states_tuple = value_states_padfree.split(self.lengths, dim=1)

        query_states_padded, key_states_padded, value_states_padded = torch.rand(len(self.lengths), max(self.lengths), self.head_num, self.head_dim, dtype=self.dtype, device=self.device), torch.rand(len(self.lengths), max(self.lengths), self.head_num, self.head_dim, dtype=self.dtype, device=self.device), torch.rand(len(self.lengths), max(self.lengths), self.head_num, self.head_dim, dtype=self.dtype, device=self.device)
        attention_mask_padded = torch.zeros(len(self.lengths), max(self.lengths), dtype=torch.bool, device=self.device)
        for idx, length in enumerate(self.lengths):
            query_states_padded[idx, :length, :, :] = query_states_tuple[idx]
            key_states_padded[idx, :length, :, :] = key_states_tuple[idx]
            value_states_padded[idx, :length, :, :] = value_states_tuple[idx]
            attention_mask_padded[idx, :length] = 1

        query_length_padded = max(self.lengths)
        query_length_padfree = self.total_length
        is_causal = True
        position_ids = None

        attn_output_padded = _flash_attention_forward(
            query_states_padded,
            key_states_padded,
            value_states_padded,
            attention_mask_padded,
            query_length = query_length_padded,
            is_causal = is_causal,
            position_ids = position_ids,
        )
        
        attn_output_padfree = _flash_attention_forward(
            query_states_padfree,
            key_states_padfree,
            value_states_padfree,
            attention_mask = None,
            query_length = query_length_padfree,
            is_causal = is_causal,
            position_ids = position_ids_padfree,
        )
        
        attn_output_padded = attn_output_padded[attention_mask_padded.bool()]
        attn_output_padfree = attn_output_padfree[0]

        # acceptable numerical instability
        tol = torch.finfo(torch.float16).eps
        torch.testing.assert_close(attn_output_padded, attn_output_padfree, atol=tol, rtol=tol)

    def test_flash_attention_forward_padding_matches_padding_free_with_attention_mask(self):
        query_states_padfree, key_states_padfree, value_states_padfree = torch.rand(1, self.total_length, self.head_num, self.head_dim, dtype=self.dtype, device=self.device), torch.rand(1, self.total_length, self.head_num, self.head_dim, dtype=self.dtype, device=self.device), torch.rand(1, self.total_length, self.head_num, self.head_dim, dtype=self.dtype, device=self.device)
        attention_mask_padfree = torch.tensor([self.lengths + [0]], dtype=torch.int, device=self.device)

        query_states_tuple = query_states_padfree.split(self.lengths, dim=1)
        key_states_tuple = key_states_padfree.split(self.lengths, dim=1)
        value_states_tuple = value_states_padfree.split(self.lengths, dim=1)

        query_states_padded, key_states_padded, value_states_padded = torch.rand(len(self.lengths), max(self.lengths), self.head_num, self.head_dim, dtype=self.dtype, device=self.device), torch.rand(len(self.lengths), max(self.lengths), self.head_num, self.head_dim, dtype=self.dtype, device=self.device), torch.rand(len(self.lengths), max(self.lengths), self.head_num, self.head_dim, dtype=self.dtype, device=self.device)
        attention_mask_padded = torch.zeros(len(self.lengths), max(self.lengths), dtype=torch.bool, device=self.device)
        for idx, length in enumerate(self.lengths):
            query_states_padded[idx, :length, :, :] = query_states_tuple[idx]
            key_states_padded[idx, :length, :, :] = key_states_tuple[idx]
            value_states_padded[idx, :length, :, :] = value_states_tuple[idx]
            attention_mask_padded[idx, :length] = 1

        query_length_padded = max(self.lengths)
        query_length_padfree = self.total_length
        is_causal = True
        position_ids = None

        attn_output_padded = _flash_attention_forward(
            query_states_padded,
            key_states_padded,
            value_states_padded,
            attention_mask_padded,
            query_length = query_length_padded,
            is_causal = is_causal,
            position_ids = position_ids,
        )
        
        attn_output_padfree = _flash_attention_forward(
            query_states_padfree,
            key_states_padfree,
            value_states_padfree,
            attention_mask_padfree,
            query_length = query_length_padfree,
            is_causal = is_causal,
            position_ids = position_ids,
        )
        
        attn_output_padded = attn_output_padded[attention_mask_padded.bool()]
        attn_output_padfree = attn_output_padfree[0]

        # acceptable numerical instability
        tol = torch.finfo(torch.float16).eps
        torch.testing.assert_close(attn_output_padded, attn_output_padfree, atol=tol, rtol=tol)
