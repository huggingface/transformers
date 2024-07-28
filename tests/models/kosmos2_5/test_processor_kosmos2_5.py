# coding=utf-8
# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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

import os
import shutil
import tempfile
import unittest

import numpy as np
import requests

from transformers.testing_utils import (
    require_torch,
    require_vision,
)
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image

    from transformers import (
        AutoProcessor,
        AutoTokenizer,
    )


@require_vision
class Kosmos2_5ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

    def test_can_load_various_tokenizers(self):
        # for checkpoint in ["microsoft/kosmos-2.5", "microsoft/kosmos-2.5"]:
        for checkpoint in ["kirp/kosmos2_5"]:
            processor = AutoProcessor.from_pretrained(checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    @require_torch
    def test_full_processor(self):
        url = "https://huggingface.co/kirp/kosmos2_5/resolve/main/receipt_00008.png"
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2.5")
        texts = [
            "<md>",
            "<ocr>"
        ]
        expected_input_ids = [
            [100288],
            [100282],
        ]
        expected_attention_mask = [
            [1],
            [1]
        ]

        image = Image.open(requests.get(url, stream=True).raw)
        # To match the official (microsoft) Kosmos-2 demo from which the expected values here are grabbed
        image_path = os.path.join(self.tmpdirname, "image.png")
        image.save(image_path)
        image = Image.open(image_path)

        # test single image
        outputs = processor(images=image, text=texts[0])
        self.assertListEqual(
            outputs.input_ids[0].numpy().tolist(),
            [0, 100283] + [0] * 2048 + [100284] + expected_input_ids[0],
        )
        self.assertListEqual(
            outputs.image_embeds_position_mask[0].numpy().tolist(),
            [0, -1] + [1] * 2048 + [-1] + [0] * (len(expected_input_ids[0])),
        )
        self.assertListEqual(
            outputs.attention_mask[0].numpy().tolist(),
           [1, 1] + [1] * 2048 + [1]+ expected_attention_mask[0],
        )
        EXPECTED_FP_1 = [1.0, 2.0, -2.9527735710144043, -2.672085762023926, -2.9933173656463623, -2.905944585800171, -2.5891761779785156, -2.8751866817474365, -2.962153434753418, -2.588062047958374]
        EXPECTED_FP_200 = [4.0, 45.0, 1.5713728666305542, 1.584628939628601, 1.3589054346084595, 1.6515952348709106, 1.7014952898025513, 1.3731343746185303, 1.6010395288467407, 1.6607422828674316]
        self.assertTupleEqual(outputs.flattened_patches.shape, (1, 4096, 770))
        np.testing.assert_allclose(outputs.flattened_patches[0][1][:10].numpy().tolist(), EXPECTED_FP_1, atol=1e-9)
        np.testing.assert_allclose(outputs.flattened_patches[0][200][:10].numpy().tolist(), EXPECTED_FP_200, atol=1e-9)

        # test a batch of images and texts, right padding
        outputs = processor(images=[image, image], text=texts)
        self.assertListEqual(
            outputs.input_ids[1].numpy().tolist(),
            [0, 100283] + [0] * 2048 + [100284] + expected_input_ids[1],
        )
        self.assertListEqual(
            outputs.image_embeds_position_mask[1].numpy().tolist(),
            [0, -1] + [1] * 2048 + [-1] + [0] * (len(expected_input_ids[1])),
        )
        self.assertListEqual(
            outputs.attention_mask[1].numpy().tolist(),
           [1, 1] + [1] * 2048 + [1]+ expected_attention_mask[1],
        )
        self.assertTupleEqual(outputs.flattened_patches.shape, (2, 4096, 770))
        np.testing.assert_allclose(outputs.flattened_patches[1][1][:10].numpy().tolist(), EXPECTED_FP_1, atol=1e-9)
        np.testing.assert_allclose(outputs.flattened_patches[1][200][:10].numpy().tolist(), EXPECTED_FP_200, atol=1e-9)

