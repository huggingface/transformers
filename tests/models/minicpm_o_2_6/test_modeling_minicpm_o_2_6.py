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
"""Testing suite for the PyTorch MiniCPM-o-2.6 model."""

import unittest
import os
import math
import tempfile
import numpy as np
import librosa
import soundfile as sf
from PIL import Image
import requests
from io import BytesIO

from transformers import (
    AutoModel,
    AutoProcessor,
)
from transformers.utils.import_utils import is_torch_available, is_soundfile_available, _is_package_available
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
    require_vision,
    require_sentencepiece,
)

if is_torch_available():
    import torch

    from transformers import MiniCPM_o_2_6ForConditionalGeneration

if is_soundfile_available():
    import soundfile as sf

if _is_package_available("moviepy"):
    from moviepy import VideoFileClip


@require_torch
class MiniCPM_o_2_6ModelIngestionTest(unittest.TestCase):
    """Test for MiniCPM_o_2_6Model."""

    all_model_classes = (MiniCPM_o_2_6ForConditionalGeneration,) if is_torch_available() else ()

    def setUp(self):
        """initial test environment"""
        self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        os.makedirs(self.assets_dir, exist_ok=True)

        self.video_path = os.path.join(self.assets_dir, "Skiing.mp4")
        self.ref_audio_path = os.path.join(self.assets_dir, "demo.wav")

        if not os.path.exists(self.video_path):
            video_url = "https://huggingface.co/openbmb/MiniCPM-o-2_6/resolve/main/assets/Skiing.mp4"
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            with open(self.video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        if not os.path.exists(self.ref_audio_path):
            audio_url = "https://huggingface.co/openbmb/MiniCPM-o-2_6/resolve/main/assets/demo.wav"
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            with open(self.ref_audio_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        self.model_path = "openbmb/MiniCPM-o-2_6"
        self.model = AutoModel.from_pretrained(self.model_path, attn_implementation="sdpa", torch_dtype=torch.bfloat16)
        self.model = self.model.eval().to(torch_device)
        self.model.init_tts()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def tearDown(self):
        """clean up test environment"""
        if os.path.exists(self.video_path):
            os.remove(self.video_path)
        if os.path.exists(self.ref_audio_path):
            os.remove(self.ref_audio_path)
        if os.path.exists(self.assets_dir):
            os.rmdir(self.assets_dir)

    def _get_video_chunk_content(self, video_path, flatten=True):
        """process video content, extract frames and audio"""
        video = VideoFileClip(video_path)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
            video.audio.write_audiofile(temp_audio_file.name, codec="pcm_s16le", fps=16000, logger=None)
            audio_np, sr = librosa.load(temp_audio_file.name, sr=16000, mono=True)

        num_units = math.ceil(video.duration)
        contents = []

        for i in range(num_units):
            frame = video.get_frame(i + 1)
            image = Image.fromarray(frame.astype(np.uint8))
            audio_segment = audio_np[sr * i : sr * (i + 1)]

            if flatten:
                contents.extend(["<unit>", image, audio_segment])
            else:
                contents.append(["<unit>", image, audio_segment])

        video.close()
        return contents

    @slow
    @require_vision
    @require_sentencepiece
    def test_omni_generate(self):
        if not is_soundfile_available():
            self.skipTest("test requires soundfile")
        if not _is_package_available("moviepy"):
            self.skipTest("test requires moviepy")

        try:
            ref_audio, _ = librosa.load(self.ref_audio_path, sr=16000, mono=True)
            sys_msg = self.processor.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")

            contents = self._get_video_chunk_content(self.video_path)
            msg = {"role": "user", "content": contents}
            msgs = [sys_msg, msg]
            inputs = self.processor.apply_chat_template(msgs=msgs).to(self.model.device)

            output_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

            res = self.model.generate(
                **inputs,
                processor=self.processor,
                sampling=True,
                temperature=0.5,
                max_new_tokens=128,
                omni_input=True,
                use_tts_template=True,
                generate_audio=True,
                output_audio_path=output_audio_path,
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True,
            )
            res = self.processor.decode(res.outputs.sequences)

            self.assertIsNotNone(res, "Chat response should not be empty")
            self.assertTrue(len(res) > 0, "Chat response text should not be empty")
            self.assertTrue(sf.info(output_audio_path).frames > 0, "Generated audio file should not be empty")

        except FileNotFoundError:
            self.skipTest(f"资源文件未找到: {self.video_path} 或 {self.ref_audio_path}")
        finally:
            if "output_audio_path" in locals() and os.path.exists(output_audio_path):
                os.remove(output_audio_path)

    @slow
    @require_vision
    @require_sentencepiece
    def test_streaming_inference(self):
        if not is_soundfile_available():
            self.skipTest("test requires soundfile")
        if not _is_package_available("moviepy"):
            self.skipTest("test requires moviepy")

        try:
            self.model.reset_session()

            ref_audio, _ = librosa.load(self.ref_audio_path, sr=16000, mono=True)
            sys_msg = self.processor.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")

            contents = self._get_video_chunk_content(self.video_path, flatten=False)
            session_id = "test_session"
            generate_audio = True

            self.model.streaming_prefill(
                session_id=session_id,
                msgs=[sys_msg],
                processor=self.processor,
            )

            for content in contents:
                msgs = [{"role": "user", "content": content}]
                self.model.streaming_prefill(
                    session_id=session_id,
                    msgs=msgs,
                    processor=self.processor,
                )

            res = self.model.streaming_generate(
                session_id=session_id, processor=self.processor, temperature=0.5, generate_audio=generate_audio
            )

            audios = []
            text = ""
            output_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

            if generate_audio:
                for r in res:
                    audio_wav = r.audio_wav
                    sampling_rate = r.sampling_rate
                    txt = r.text

                    audios.append(audio_wav)
                    text += txt

                res_audio = np.concatenate(audios)
                sf.write(output_audio_path, res_audio, samplerate=sampling_rate)

                self.assertTrue(len(text) > 0, "Generated text should not be empty")
                self.assertTrue(sf.info(output_audio_path).frames > 0, "Generated audio file should not be empty")
            else:
                for r in res:
                    text += r["text"]
                self.assertTrue(len(text) > 0, "Generated text should not be empty")

        except FileNotFoundError:
            self.skipTest(f"Resource file not found: {self.video_path} or {self.ref_audio_path}")
        finally:
            if "output_audio_path" in locals() and os.path.exists(output_audio_path):
                os.remove(output_audio_path)

    @slow
    @require_sentencepiece
    def test_audio_mimick(self):
        if not is_soundfile_available():
            self.skipTest("test requires soundfile")

        try:
            self.model.init_tts()

            mimick_prompt = "Please repeat each user's speech, including voice style and speech content."

            audio_urls = [
                "https://huggingface.co/openbmb/MiniCPM-o-2_6/resolve/main/assets/input_examples/Trump_WEF_2018_10s.mp3",
                "https://huggingface.co/openbmb/MiniCPM-o-2_6/resolve/main/assets/input_examples/cxk_original.wav",
                "https://huggingface.co/openbmb/MiniCPM-o-2_6/resolve/main/assets/input_examples/fast-pace.wav",
                "https://huggingface.co/openbmb/MiniCPM-o-2_6/resolve/main/assets/input_examples/chi-english-1.wav",
                "https://huggingface.co/openbmb/MiniCPM-o-2_6/resolve/main/assets/input_examples/exciting-emotion.wav",
            ]

            for audio_url in audio_urls:
                try:
                    response = requests.get(audio_url, stream=True)
                    response.raise_for_status()

                    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_url)[1], delete=False) as temp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            temp_file.write(chunk)
                        temp_file_path = temp_file.name

                    audio_input, _ = librosa.load(temp_file_path, sr=16000, mono=True)

                    msgs = [{"role": "user", "content": [mimick_prompt, audio_input]}]
                    inputs = self.processor.apply_chat_template(msgs=msgs).to(self.model.device)

                    output_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

                    res = self.model.generate(
                        **inputs,
                        processor=self.processor,
                        sampling=True,
                        max_new_tokens=128,
                        use_tts_template=True,
                        temperature=0.3,
                        generate_audio=True,
                        output_audio_path=output_audio_path,
                    )
                    res = self.processor.decode(res.outputs.sequences)

                    self.assertIsNotNone(res, "Mimic response should not be empty")
                    self.assertTrue(os.path.exists(output_audio_path), "Output audio file should exist")
                    self.assertTrue(sf.info(output_audio_path).frames > 0, "Generated audio file should not be empty")

                except requests.exceptions.RequestException as e:
                    self.skipTest(f"Failed to download audio file: {str(e)}")
                except Exception as e:
                    self.fail(f"Failed to process audio file: {str(e)}")
                finally:
                    if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    if "output_audio_path" in locals() and os.path.exists(output_audio_path):
                        os.remove(output_audio_path)

        except Exception as e:
            self.fail(f"Audio mimick test failed: {str(e)}")

    @slow
    @require_vision
    @require_sentencepiece
    def test_single_image_inference(self):
        try:
            image_url = "https://bkimg.cdn.bcebos.com/pic/d043ad4bd11373f082022267f9585cfbfbedaa64aeb6"
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content)).convert("RGB")
            question = "What is in the image?"

            msgs = [{"role": "user", "content": [image, question]}]
            inputs = self.processor.apply_chat_template(msgs=msgs).to(self.model.device)

            res = self.model.generate(
                **inputs,
                processor=self.processor,
            )
            res = self.processor.decode(res.sequences)

            self.assertIsNotNone(res, "Normal inference response should not be empty")
            self.assertTrue(len(res) > 0, "Normal inference response text should not be empty")

            res = self.model.generate(**inputs, processor=self.processor, sampling=True, stream=True)

            generated_text = ""
            for new_text in res:
                generated_text += new_text
                self.assertIsNotNone(new_text, "Each part of streaming reasoning should not be empty")

            self.assertTrue(len(generated_text) > 0, "Text should not be empty")

        except requests.exceptions.RequestException as e:
            self.skipTest(f"Failed to download image: {str(e)}")
        except Exception as e:
            self.fail(f"Single image inference test failed: {str(e)}")
