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
"""Loading optimum-executorch's programs from the Hub with `ExportedModel.from_pretrained`, against the eager models
they were exported from."""

import unittest

import numpy as np
import torch

from tests.exporters.test_utils import _check_names_the_cats, _load
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    GenerationMixin,
)
from transformers.exporters import (
    ExportedModel,
    ExportedModelForCausalLM,
    ExportedModelForSeq2SeqLM,
)
from transformers.image_utils import load_image
from transformers.testing_utils import (
    require_executorch,
    require_torch,
    slow,
)


@slow
@require_torch
@require_executorch
class OptimumExecutorchTest(unittest.TestCase):
    def test_causal_lm(self):
        """A program keeping its cache in its buffers generates the eager model's tokens, and starts over each call."""
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        inputs = tokenizer(["The capital of France is"], return_tensors="pt")
        generation_config = GenerationConfig(max_new_tokens=8, do_sample=False)
        model = _load(self, "executorch-community/SmolLM2-135M", "optimum-executorch")
        self.assertIsInstance(model, ExportedModelForCausalLM)
        eager = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32).eval()
        with torch.no_grad():
            expected = eager.generate(**inputs, generation_config=generation_config)
        for _ in range(2):
            generated = model.generate(**inputs, generation_config=generation_config)
            self.assertEqual(generated.tolist(), expected.tolist())

    def test_encoder_decoder(self):
        """`encoder` and `text_decoder` methods of one program, through `generate` as eager's."""
        from transformers import WhisperForConditionalGeneration

        processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
        time = np.linspace(0, 3, 48000, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * time) * (np.sin(2 * np.pi * 2 * time) > 0)
        features = processor(audio, sampling_rate=16000, return_tensors="pt")
        generation_config = GenerationConfig.from_pretrained("openai/whisper-tiny")
        generation_config.max_new_tokens = 16
        model = _load(self, "larryliu0820/whisper-tiny-ExecuTorch-XNNPACK", "optimum-executorch")
        self.assertIsInstance(model, ExportedModelForSeq2SeqLM)
        eager = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").eval()
        with torch.no_grad():
            # `generate` itself, as the runtime drives it, rather than Whisper's own override.
            expected = GenerationMixin.generate(eager, **features, generation_config=generation_config)
        self.assertEqual(model.generate(**features, generation_config=generation_config).tolist(), expected.tolist())

    def test_multimodal(self):
        """`token_embedding`, `vision_encoder` and `text_decoder` methods, run as the multimodal split (an int4
        program, so it answers rather than matching eager's logits)."""
        _check_names_the_cats(self, "pytorch/gemma-3-4b-it-HQQ-INT8-INT4", "optimum-executorch")

    def test_fill_mask(self):
        """A text encoder's `forward(input_ids, attention_mask)`, its logits by name."""
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        inputs = tokenizer(["The capital of France is [MASK]."], return_tensors="pt")
        model = _load(self, "IlyasMoutawwakil/distilbert-base-uncased-fill-mask-executorch", "optimum-executorch")
        self.assertIs(type(model), ExportedModel)
        eager = AutoModelForMaskedLM.from_pretrained("distilbert/distilbert-base-uncased").eval()
        with torch.no_grad():
            expected = eager(**inputs).logits
        torch.testing.assert_close(model(**inputs)["logits"], expected, atol=1e-4, rtol=1e-4)

    def test_image_classification(self):
        """An image encoder's `forward(pixel_values)`, its logits by name."""
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        inputs = processor(load_image("http://images.cocodataset.org/val2017/000000039769.jpg"), return_tensors="pt")
        model = _load(
            self, "IlyasMoutawwakil/vit-base-patch16-224-image-classification-executorch", "optimum-executorch"
        )
        eager = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").eval()
        with torch.no_grad():
            expected = eager(**inputs).logits
        torch.testing.assert_close(model(**inputs)["logits"], expected, atol=1e-4, rtol=1e-4)
