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

import inspect
import shutil
import tempfile
import unittest

import numpy as np
import pytest
import librosa
import requests

from transformers import (
    AutoProcessor,
    MiniCPM_o_2_6Processor,
    AutoTokenizer,
    WhisperFeatureExtractor,
)
from transformers.testing_utils import require_av, require_librosa, require_torch, require_torchaudio, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    from transformers import AutoImageProcessor


@require_vision
@require_torch
@require_torchaudio
class MiniCPM_o_2_6ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MiniCPM_o_2_6Processor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor_kwargs = self.prepare_processor_dict()
        processor = MiniCPM_o_2_6Processor.from_pretrained("openbmb/MiniCPM-O-2.6", **processor_kwargs)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).feature_extractor

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
        }

    def tearDown(self):
        shutil.rmtree(self.tmpdirname, ignore_errors=True)

    def prepare_image_inputs(self):
        """Preparing the image input for testing"""
        image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open(requests.get(image_url, stream=True).raw)
        return image

    def prepare_audio_inputs(self):
        """Preparing the audio input for testing"""
        audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
        try:
            audio, _ = librosa.load(audio_url, sr=16000)
            return audio
        except:
            # 如果下载失败，使用随机生成的音频
            return np.random.rand(160000) * 2 - 1  # 模拟1秒的音频

    def test_save_load_pretrained_default(self):
        """Test saving and loading pretrained models"""
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        processor = self.processor_class(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
        )

        processor.save_pretrained(self.tmpdirname)
        processor = MiniCPM_o_2_6Processor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.image_processor.to_json_string(), image_processor.to_json_string())
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())

    def test_image_processor(self):
        """Testing image processing functions"""
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        processor = self.processor_class(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
        )

        image_input = self.prepare_image_inputs()
        input_image_proc = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, text="dummy", return_tensors="np")

        for key in input_image_proc.keys():
            self.assertAlmostEqual(input_image_proc[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_processor(self):
        """Testing overall processing functions"""
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        processor = self.processor_class(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
        )

        input_str = "Test text"
        image_input = self.prepare_image_inputs()
        audio_input = self.prepare_audio_inputs()
        inputs = processor(text=input_str, images=image_input, audios=audio_input)
        
        keys = list(inputs.keys())
        self.assertListEqual(
            keys,
            [
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_sizes",
                "image_bound",
                "tgt_sizes",
                "audio_features",
                "audio_feature_lens",
                "audio_bounds",
                "spk_bounds",
            ],
        )

        # no input  
        with pytest.raises(ValueError):
            processor()

        # no text input
        with pytest.raises(ValueError):
            processor(images=image_input)

    def test_model_input_names(self):
        """Testing model input names"""
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        processor = self.processor_class(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
        )

        input_str = "Test text"
        image_input = self.prepare_image_inputs()
        audio_input = self.prepare_audio_inputs()

        inputs = processor(text=input_str, images=image_input, audios=audio_input)
        self.assertListEqual(sorted(inputs.keys()), sorted(processor.model_input_names))

    def test_apply_chat_template(self):
        """Testing chat template application"""
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor does not have a chat template")

        messages = [
            [
                {
                    "role": "user",
                    "content": "Please describe this image.",
                },
                {
                    "role": "assistant",
                    "content": "This is an image.",
                },
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        expected_output = processor.tokenizer(formatted_prompt, return_tensors=None).input_ids
        self.assertListEqual(expected_output, formatted_prompt_tokenized)

    def test_audio_processing(self):
        """Testing audio processing functions"""
        processor = self.get_processor()
        audio_input = self.prepare_audio_inputs()
        
        # test audio feature extraction
        audio_features = processor.feature_extractor(
            audio_input,
            sampling_rate=16000,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        self.assertIn("input_features", audio_features)
        self.assertIn("attention_mask", audio_features)
        
        # test audio and text combination processing
        inputs = processor(
            text="Describe this audio",
            audios=audio_input,
            return_tensors="pt",
        )
        
        self.assertIn("audio_features", inputs)
        self.assertIn("audio_feature_lens", inputs)
        self.assertIn("audio_bounds", inputs)

    def test_number_to_text_converter(self):
        """Testing number to text conversion functions"""
        processor = self.get_processor()
        converter = processor.NumberToTextConverter()
        
        # test chinese number conversion
        chinese_text = converter.replace_numbers_with_text("我有2个苹果", language="chinese")
        self.assertEqual(chinese_text, "我有两个苹果")
        
        # test english number conversion
        english_text = converter.replace_numbers_with_text("I have 23 books", language="english")
        self.assertEqual(english_text, "I have two three books")
        
        # test automatic language detection
        mixed_text = converter.replace_numbers_with_text("我有3个苹果和4个梨")
        self.assertIn("三", mixed_text)
        self.assertIn("四", mixed_text)

    def test_voice_checker(self):
        """Testing voice checking functions"""
        processor = self.get_processor()
        checker = processor.VoiceChecker()
        
        # test silent detection
        silent_audio = np.zeros(16000)  # 1 second of silence
        mel_spec = np.random.rand(100, 100)  # simulate mel spectrogram
        is_bad = checker.is_bad(silent_audio, mel_spec)
        self.assertTrue(is_bad)
        
        # test normal audio
        normal_audio = np.random.rand(16000) * 2 - 1  # 1 second of normal audio
        is_bad = checker.is_bad(normal_audio, mel_spec)
        self.assertFalse(is_bad)
        
        # test reset function
        checker.reset()
        self.assertIsNone(checker.previous_mel)
        self.assertEqual(checker.consecutive_zeros, 0)
        self.assertEqual(checker.consecutive_low_distance, 0)

    def test_image_processor_basic(self):
        """Testing basic image processor functions"""
        image_processor = self.get_image_processor()
        
        # test image preprocessing
        image = self.prepare_image_inputs()
        processed = image_processor.preprocess(image)
        
        self.assertIn("pixel_values", processed)
        self.assertIn("image_sizes", processed)
        self.assertIn("tgt_sizes", processed)
        
        # test image slicing function
        sliced_images = image_processor.get_sliced_images(image)
        self.assertIsInstance(sliced_images, list)
        self.assertTrue(len(sliced_images) > 0)
        
        # test image placeholder generation
        placeholder = image_processor.get_slice_image_placeholder(image.size)
        self.assertIsInstance(placeholder, str)
        self.assertTrue(len(placeholder) > 0)

    def test_image_processor_edge_cases(self):
        """Testing image processor edge cases"""
        image_processor = self.get_image_processor()
        
        # test empty image list
        with self.assertRaises(ValueError):
            image_processor.preprocess([])
            
        # test invalid image
        invalid_image = np.zeros((100, 100))
        with self.assertRaises(ValueError):
            image_processor.preprocess(invalid_image)
            
        # test large image
        large_image = Image.new('RGB', (2000, 2000))
        processed = image_processor.preprocess(large_image)
        self.assertIn("pixel_values", processed)

    def test_chat_tts_processor(self):
        """Testing ChatTTS processor functions"""
        processor = self.get_processor()
        tts_processor = processor.ChatTTSProcessor(processor.tokenizer)
        
        # prepare test data
        text_list = ["Test text 1", "Test text 2"]
        audio_list = [np.random.rand(16000) for _ in range(2)]
        
        # test processing function
        outputs = tts_processor(text_list, audio_list)
        
        self.assertIn("tts_input_ids_varlen", outputs)
        self.assertIn("tts_input_features_varlen", outputs)
        self.assertEqual(len(outputs["tts_input_ids_varlen"]), len(text_list))
        self.assertEqual(len(outputs["tts_input_features_varlen"]), len(audio_list))

    def test_mel_spectrogram_features(self):
        """Testing mel spectrogram feature extraction functions"""
        processor = self.get_processor()
        mel_processor = processor.MelSpectrogramFeatures()
        
        # prepare test audio
        audio = torch.randn(1, 16000)  # 1 second of audio
        
        # test feature extraction
        features = mel_processor(audio)
        
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.dim(), 2)  # should be 2D tensor
        self.assertEqual(features.shape[0], 100)  # mel filter number

    def test_batch_processing(self):
        """Testing batch processing functions"""
        processor = self.get_processor()
        
        # prepare batch processing data
        texts = ["Text 1", "Text 2", "Text 3"]
        images = [self.prepare_image_inputs() for _ in range(3)]
        audios = [np.random.rand(16000) for _ in range(3)]
        
        # test batch processing
        batch_outputs = processor(
            text=texts,
            images=images,
            audios=audios,
            return_tensors="pt"
        )
        
        # verify outputs
        self.assertIn("input_ids", batch_outputs)
        self.assertIn("attention_mask", batch_outputs)
        self.assertIn("pixel_values", batch_outputs)
        self.assertIn("audio_features", batch_outputs)
        
        # verify batch size
        self.assertEqual(batch_outputs["input_ids"].shape[0], len(texts))
        self.assertEqual(len(batch_outputs["pixel_values"]), len(images))
        self.assertEqual(len(batch_outputs["audio_features"]), len(audios))

    def test_error_handling(self):
        """Testing error handling functions"""
        processor = self.get_processor()
        
        # test invalid input
        with self.assertRaises(ValueError):
            processor(text=None, images=None, audios=None)
            
        # test mismatch input length
        with self.assertRaises(AssertionError):
            processor(
                text=["Text 1", "Text 2"],
                images=[self.prepare_image_inputs()],
                audios=[np.random.rand(16000)]
            )
            
        # test invalid audio sampling rate
        with self.assertRaises(ValueError):
            processor(
                text="Test text",
                audios=np.random.rand(16000),
                sampling_rate=0
            )
