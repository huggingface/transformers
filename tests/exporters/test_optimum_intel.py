# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Loading optimum-intel's OpenVINO IRs from the Hub with `ExportedModel.from_pretrained`, against the eager models
they were exported from (or, for a multimodal model, asking what a COCO image shows)."""

import unittest

import numpy as np
import torch

from tests.exporters.test_utils import _check_encoder, _check_names_the_cats, _load
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    GenerationMixin,
)
from transformers.exporters import (
    ExportedModelForCausalLM,
    ExportedModelForSeq2SeqLM,
    ExportedMultimodalModelForCausalLM,
)
from transformers.testing_utils import (
    require_openvino,
    require_soundfile,
    require_torch,
    slow,
)


@slow
@require_torch
@require_openvino
class OptimumIntelTest(unittest.TestCase):
    def test_causal_lm(self):
        repo = "OpenVINO/Qwen3-0.6B-int4-ov"
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = _load(self, repo, "optimum-intel")
        self.assertIsInstance(model, ExportedModelForCausalLM)
        inputs = tokenizer(["The capital of France is"], return_tensors="pt")
        generated = model.generate(**inputs, generation_config=GenerationConfig(max_new_tokens=8, do_sample=False))
        self.assertIn("Paris", tokenizer.decode(generated[0], skip_special_tokens=True))

    def test_encoder(self):
        _check_encoder(self, "OpenVINO/bge-base-en-v1.5-fp16-ov", "optimum-intel")

    def test_encoder_decoder(self):
        from transformers import WhisperForConditionalGeneration

        processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
        time = np.linspace(0, 3, 48000, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * time) * (np.sin(2 * np.pi * 2 * time) > 0)
        features = processor(audio, sampling_rate=16000, return_tensors="pt")
        generation_config = GenerationConfig.from_pretrained("openai/whisper-tiny")
        generation_config.max_new_tokens = 16
        model = _load(self, "OpenVINO/whisper-tiny-fp16-ov", "optimum-intel")
        self.assertIsInstance(model, ExportedModelForSeq2SeqLM)
        generated = model.generate(**features, generation_config=generation_config)
        eager = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").eval()
        with torch.no_grad():
            # `generate` itself, as the runtime drives it, rather than Whisper's own override.
            expected = GenerationMixin.generate(eager, **features, generation_config=generation_config)
        self.assertEqual(generated.tolist(), expected.tolist())

    def test_multimodal(self):
        """Text embeddings, one vision graph and the language model, as this runtime splits Gemma 3: run as the
        multimodal split (an int4 export, so it answers rather than matching eager's logits)."""
        _check_names_the_cats(self, "OpenVINO/gemma-3-4b-it-int4-cw-ov", "optimum-intel", "google/gemma-3-4b-it")

    def test_multimodal_with_a_vision_merger(self):
        """Qwen2.5-VL's vision tower, split in two graphs, is fused into one, its merger fed the rotary embeddings,
        attention masks and window index the image grid gives."""
        _check_names_the_cats(self, "OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov", "optimum-intel")

    def test_multimodal_with_vision_position_embeddings(self):
        """Qwen3-VL's vision tower, split in three graphs, is fused into one: its position embeddings at the bilinear
        indices of the image grid, weighted and summed, and its merger's deepstack features fed to the language model."""
        _check_names_the_cats(self, "OpenVINO/Qwen3-VL-2B-Instruct-int4-ov", "optimum-intel")

    def test_multimodal_packing_crops(self):
        """LLaVA-NeXT's crops, packed with the `image_newline` its config holds."""
        _check_names_the_cats(self, "OpenVINO/llava-v1.6-mistral-7b-hf-int4-ov", "optimum-intel")

    def test_multimodal_with_raster_patches(self):
        """LFM3-VL's vision graph, run per image on its patches in raster order (its processor is remote code)."""
        _check_names_the_cats(
            self, "OpenVINO/LFM3-VL-2B-int4-gq-ov", "optimum-intel", max_new_tokens=60, trust_remote_code=True
        )

    def test_multimodal_with_crops_glue(self):
        """MiniCPM-V 4.7's vision graph, taking the position, window and merge indices of each image's crops."""
        _check_names_the_cats(self, "OpenVINO/MiniCPM-V-4.7-1B-int4-ov", "optimum-intel", max_new_tokens=60)

    def test_multimodal_with_projected_crops(self):
        """Phi-3.5-vision's vision graph fused with its projection, its crops laid out per image in Python, its
        placeholders negative ids (its config and processor are remote code, its prompt its own)."""
        from transformers.image_utils import load_image

        repo = "OpenVINO/Phi-3.5-vision-instruct-int4-ov"
        processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
        prompt = "<|image_1|>\nWhat animal is in this picture? Answer in one word."
        text = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, images=[image], return_tensors="pt")
        model = _load(self, repo, "optimum-intel", trust_remote_code=True)
        self.assertIsInstance(model, ExportedMultimodalModelForCausalLM)
        generated = model.generate(**inputs, generation_config=GenerationConfig(max_new_tokens=16, do_sample=False))
        answer = processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertIn("cat", answer.lower())

    def test_multimodal_with_projected_crops_and_audio(self):
        """Phi-4-multimodal's vision graph fused with its projection, its crops cut to what their mask keeps (its config
        and processor are remote code, its prompt its own)."""
        from transformers.image_utils import load_image

        repo = "OpenVINO/Phi-4-multimodal-instruct-int4-cw-ov"
        processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
        prompt = "<|image_1|>What animal is in this picture? Answer in one word."
        text = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, images=[image], return_tensors="pt")
        model = _load(self, repo, "optimum-intel", trust_remote_code=True)
        self.assertIsInstance(model, ExportedMultimodalModelForCausalLM)
        generated = model.generate(**inputs, generation_config=GenerationConfig(max_new_tokens=16, do_sample=False))
        answer = processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertIn("cat", answer.lower())

    @require_soundfile
    def test_audio_tower_fused(self):
        """Phi-4-multimodal's audio tower fused into one graph, the encoder run on 500-frame chunks, transcribing a
        LibriSpeech clip."""
        from transformers.audio_utils import load_audio

        repo = "OpenVINO/Phi-4-multimodal-instruct-int4-cw-ov"
        processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        audio = load_audio("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac", sampling_rate=16000)
        prompt = "<|audio_1|>Transcribe the audio clip into text."
        text = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, audios=[(audio, 16000)], return_tensors="pt")
        model = _load(self, repo, "optimum-intel", trust_remote_code=True)
        generated = model.generate(**inputs, generation_config=GenerationConfig(max_new_tokens=16, do_sample=False))
        answer = processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertIn("stew for dinner", answer.lower())

    def test_multimodal_with_resampler(self):
        """MiniCPM-V 4.5's vision graph fused with its resampler, run per crop (its config and processor are remote
        code, its prompt its own)."""
        from transformers.image_utils import load_image

        repo = "OpenVINO/MiniCPM-V-4_5-int4-cw-ov"
        processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
        prompt = "(<image>./</image>)\nWhat animal is in this picture? Answer in one word."
        text = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[[image]], return_tensors="pt")
        model = _load(self, repo, "optimum-intel", trust_remote_code=True)
        self.assertIsInstance(model, ExportedMultimodalModelForCausalLM)
        generated = model.generate(**inputs, generation_config=GenerationConfig(max_new_tokens=16, do_sample=False))
        answer = processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertIn("cat", answer.lower())
