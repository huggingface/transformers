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
""" Testing suite for the PyTorch MiniCPM-o-2.6 model. """

import unittest
import os # Added import os
from io import BytesIO
from urllib.request import urlopen

import librosa
import requests

import math
import numpy as np
from moviepy.editor import VideoFileClip
import tempfile
import soundfile as sf

from transformers import (
    MiniCPM_o_2_6Config,
    MiniCPM_o_2_6Model,
    MiniCPM_o_2_6ForConditionalGeneration,
    AutoProcessor,
    AutoModel,
    AutoTokenizer, # Added AutoTokenizer
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_torch, 
    slow, 
    torch_device,
    require_vision,
    require_soundfile,
    require_sentencepiece,
    require_tokenizers,
    require_flash_attn
)

if is_torch_available():
    import torch

if is_torch_available():
    from PIL import Image

@require_torch
class MiniCPM_o_2_6ModelIngestionTest(unittest.TestCase):
    """Test for MiniCPM_o_2_6Model."""

    def setUp(self):
        # Initialize model and tokenizer as in omni_chat.py
        self.model = AutoModel.from_pretrained(
            "OpenBMB/MiniCPM-o-2.6",
            trust_remote_code=True,
            attn_implementation='sdpa',  # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True
        )
        self.model = self.model.eval().to(torch_device)
        self.tokenizer = AutoTokenizer.from_pretrained('OpenBMB/MiniCPM-o-2.6', trust_remote_code=True)
        self.model.init_tts() # Initialize TTS

        # Placeholder for assets - in a real test, manage these properly
        # For now, assuming they exist relative to where the test is run or are handled externally
        self.video_path = "assets/Skiing.mp4" # May need adjustment for test environment
        self.ref_audio_path = "assets/demo.wav" # May need adjustment for test environment

    @slow
    def test_minicpm_o_2_6_model_base(self):
        """Test for base MiniCPM_o_2_6Model loading and a simple check."""
        # This is a simplified version of the original test_minicpm_o_2_6_model
        # to ensure the base model still loads if needed for other tests.
        # Or, it can be removed if all tests use the AutoModel setup.
        base_model = MiniCPM_o_2_6Model.from_pretrained("OpenBMB/MiniCPM-o-2.6")
        self.assertIsNotNone(base_model)

    def _get_video_chunk_content(self, video_path, flatten=True):
        video = VideoFileClip(video_path)
        # print('video_duration:', video.duration) # Avoid print in tests, use assertions or logging
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
            temp_audio_file_path = temp_audio_file.name
            video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000, logger=None) # Suppress moviepy logger
            audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
        num_units = math.ceil(video.duration)
        
        contents= []
        for i in range(num_units):
            frame = video.get_frame(i+1) # moviepy frames are 1-indexed for get_frame
            image = Image.fromarray((frame).astype(np.uint8))
            audio_segment = audio_np[sr*i:sr*(i+1)]
            if flatten:
                contents.extend(["<unit>", image, audio_segment])
            else:
                contents.append(["<unit>", image, audio_segment])
        video.close() # Ensure video file is closed
        return contents

    @slow
    @require_vision
    @require_soundfile # For librosa and sf
    @require_sentencepiece
    @require_tokenizers
    # @require_flash_attn # if flash_attention_2 is used and required
    def test_omni_chat(self):
        """Test for omni chat functionality based on omni_chat.py."""
        # For tests, ensure asset paths are correct or use mocks/small test files
        # For now, using paths from setUp
        try:
            ref_audio, _ = librosa.load(self.ref_audio_path, sr=16000, mono=True)
            sys_msg = self.model.get_sys_prompt(ref_audio=ref_audio, mode='omni', language='en')
            
            contents = self._get_video_chunk_content(self.video_path)
            msg = {"role":"user", "content": contents}
            msgs = [sys_msg, msg]

            output_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

            res = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=0.5,
                max_new_tokens=128, # Reduced for faster testing
                omni_input=True,
                use_tts_template=True,
                generate_audio=True,
                output_audio_path=output_audio_path,
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True
            )
            self.assertIsNotNone(res, "Chat response should not be None")
            self.assertIn("text", res, "Chat response should contain text")
            self.assertTrue(len(res["text"]) > 0, "Chat response text should not be empty")
            if res.get("audio_path"): # audio_path might not always be in res if generate_audio=False or error
                 self.assertTrue(sf.info(output_audio_path).frames > 0, "Generated audio file should not be empty")
        except FileNotFoundError:
            self.skipTest(f"Asset file not found (e.g., {self.video_path} or {self.ref_audio_path}). Skipping omni_chat test.")
        finally:
            if 'output_audio_path' in locals() and os.path.exists(output_audio_path):
                os.remove(output_audio_path)
        