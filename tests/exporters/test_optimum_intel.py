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
"""Loading optimum-intel OpenVINO exports from the Hub (the OpenVINO organization's) with `ExportedModel.from_pretrained`:
a causal LM and an encoder-decoder through `generate`, and an encoder, against the eager models they were exported from."""

import unittest

import numpy as np

from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig, GenerationMixin
from transformers.exporters import (
    ExportedModel,
    ExportedModelForCausalLM,
    ExportedModelForSeq2SeqLM,
    optimum_intel,
)
from transformers.testing_utils import CaptureLogger, require_openvino, require_torch, slow
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


@slow
@require_torch
@require_openvino
class OptimumIntelLoadingTest(unittest.TestCase):
    def _load(self, repo):
        with CaptureLogger(optimum_intel.logger) as captured:
            model = ExportedModel.from_pretrained(repo, device="cpu")
        self.assertIn("legacy format", captured.out)
        return model

    def test_causal_lm(self):
        repo = "OpenVINO/Qwen3-0.6B-int4-ov"
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = self._load(repo)
        self.assertIsInstance(model, ExportedModelForCausalLM)
        inputs = tokenizer(["The capital of France is"], return_tensors="pt")
        generated = model.generate(**inputs, generation_config=GenerationConfig(max_new_tokens=8, do_sample=False))
        self.assertIn("Paris", tokenizer.decode(generated[0], skip_special_tokens=True))

    def test_encoder(self):
        repo = "OpenVINO/bge-base-en-v1.5-fp16-ov"
        tokenizer = AutoTokenizer.from_pretrained(repo)
        inputs = dict(tokenizer(["The capital of France is Paris."], return_tensors="pt"))
        model = self._load(repo)
        self.assertIs(type(model), ExportedModel)
        with torch.no_grad():
            expected = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").eval()(**inputs).last_hidden_state[:, 0]
        # The IR records no output class: outputs come back by the IR's names.
        embedding = model(**inputs)["last_hidden_state"][:, 0].float()
        self.assertGreater(torch.nn.functional.cosine_similarity(embedding, expected, dim=-1).item(), 0.999)

    def test_encoder_decoder(self):
        from transformers import WhisperForConditionalGeneration

        processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
        time = np.linspace(0, 3, 48000, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * time) * (np.sin(2 * np.pi * 2 * time) > 0)
        features = processor(audio, sampling_rate=16000, return_tensors="pt")
        generation_config = GenerationConfig.from_pretrained("openai/whisper-tiny")
        generation_config.max_new_tokens = 16
        model = self._load("OpenVINO/whisper-tiny-fp16-ov")
        self.assertIsInstance(model, ExportedModelForSeq2SeqLM)
        generated = model.generate(**features, generation_config=generation_config)
        eager = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").eval()
        with torch.no_grad():
            # `generate` itself, as the runtime drives it, rather than Whisper's own override.
            expected = GenerationMixin.generate(eager, **features, generation_config=generation_config)
        self.assertEqual(generated.tolist(), expected.tolist())

    def test_multimodal_is_refused(self):
        with self.assertRaisesRegex(ValueError, "multimodal export"):
            ExportedModel.from_pretrained("OpenVINO/Qwen3-VL-2B-Instruct-fp16-ov", device="cpu")
