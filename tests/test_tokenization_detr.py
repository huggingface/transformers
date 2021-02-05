# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
"""Tests for the Wav2Vec2 tokenizer."""

import inspect
import json
import os
import random
import shutil
import tempfile
import unittest

import torch
import numpy as np

from PIL import Image
import requests

from transformers.models.detr.tokenization_detr import DetrTokenizer

class DetrTokenizerTest(unittest.TestCase):
    tokenizer_class = DetrTokenizer

    def setUp(self):
        super().setUp()
        
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        self.img = Image.open(requests.get(url, stream=True).raw)

    def get_tokenizer(self, **kwars):
        return DetrTokenizer()

    def test_tokenizer_no_resize(self):
        tokenizer = self.get_tokenizer()
        encoding = tokenizer(self.img, resize=False)

        self.assertEqual(encoding["pixel_values"].shape, (1,3,480,640))
        self.assertEqual(encoding["pixel_mask"].shape, (1,480,640))

    def test_tokenizer(self):
        tokenizer = self.get_tokenizer()
        encoding = tokenizer(self.img)
        
        self.assertEqual(encoding["pixel_values"].shape, (1,3,800,1066))
        self.assertEqual(encoding["pixel_mask"].shape, (1,800,1066))