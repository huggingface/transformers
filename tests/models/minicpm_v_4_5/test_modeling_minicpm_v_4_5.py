# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
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
"""Testing suite for the PyTorch MiniCPM-V-4_5 model."""

import math
import os
import unittest
from io import BytesIO

import numpy as np
import requests
from moviepy.editor import VideoFileClip
from PIL import Image

from transformers import (
    AutoModel,
    AutoTokenizer,
    is_torch_available,
)
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_vision,
    slow,
    torch_device,
)


if is_torch_available():
    import torch


@require_torch
class MiniCPM_V_4_5ModelIngestionTest(unittest.TestCase):
    """Test for MiniCPM_V_4_5Model."""

    def setUp(self):
        """initial test environment"""
        self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        os.makedirs(self.assets_dir, exist_ok=True)

        self.video_path = os.path.join(self.assets_dir, "Skiing.mp4")

        if not os.path.exists(self.video_path):
            video_url = "https://huggingface.co/openbmb/MiniCPM-V-4_5/resolve/main/assets/Skiing.mp4"
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            with open(self.video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        self.model = AutoModel.from_pretrained(
            "openbmb/MiniCPM-V-4_5",
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=True,
        )
        self.model = self.model.eval().to(torch_device)
        self.tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-4_5", trust_remote_code=True)

    def tearDown(self):
        """clean up test environment"""
        if os.path.exists(self.video_path):
            os.remove(self.video_path)
        if os.path.exists(self.assets_dir):
            os.rmdir(self.assets_dir)

    @slow
    def test_MiniCPM_V_4_5_model_base(self):
        """test base model loading"""
        base_model = AutoModel.from_pretrained("openbmb/MiniCPM-V-4_5")
        self.assertIsNotNone(base_model)

    def _get_video_chunk_content(self, video_path, flatten=True):
        """process video content, extract frames"""
        video = VideoFileClip(video_path)

        num_units = math.ceil(video.duration)
        contents = []

        for i in range(num_units):
            frame = video.get_frame(i + 1)
            image = Image.fromarray(frame.astype(np.uint8))

            if flatten:
                contents.extend([image])
            else:
                contents.append([image])

        video.close()
        return contents

    @slow
    @require_vision
    @require_sentencepiece
    @require_tokenizers
    def test_single_image_inference(self):
        try:
            image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content)).convert("RGB")
            question = "What is in the image?"

            msgs = [{"role": "user", "content": [image, question]}]

            res = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)

            self.assertIsNotNone(res, "Normal inference response should not be empty")
            self.assertTrue(len(res) > 0, "Normal inference response text should not be empty")

            res = self.model.chat(msgs=msgs, tokenizer=self.tokenizer, sampling=True, stream=True)

            generated_text = ""
            for new_text in res:
                generated_text += new_text
                self.assertIsNotNone(new_text, "Each part of streaming reasoning should not be empty")

            self.assertTrue(len(generated_text) > 0, "Text should not be empty")

        except requests.exceptions.RequestException as e:
            self.skipTest(f"Failed to download image: {str(e)}")
        except Exception as e:
            self.fail(f"Single image inference test failed: {str(e)}")
