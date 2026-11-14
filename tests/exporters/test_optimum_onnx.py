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
"""Loading optimum-onnx's ONNX graphs from the Hub with `ExportedModel.from_pretrained`, against the eager models they
were exported from."""

import tempfile
import unittest

import torch
from huggingface_hub import snapshot_download
from parameterized import parameterized

from tests.exporters.test_utils import _check_encoder, _load
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GenerationMixin,
)
from transformers.exporters import (
    ExportedModel,
    ExportedModelForCausalLM,
    ExportedModelForSeq2SeqLM,
)
from transformers.testing_utils import (
    require_onnxruntime,
    require_onnxscript,
    require_torch,
    slow,
)


@slow
@require_torch
@require_onnxscript
@require_onnxruntime
class OptimumOnnxTest(unittest.TestCase):
    @parameterized.expand(
        [
            # One graph taking and returning the cache, its weights in an external `model.onnx_data`
            ("onnx-community/Qwen3-0.6B-ONNX", "Qwen/Qwen3-0.6B"),
            ("HuggingFaceTB/SmolLM2-135M-Instruct", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        ]
    )
    def test_causal_lm(self, repo, eager_repo):
        model = _load(self, repo, "optimum-onnx")
        self.assertIsInstance(model, ExportedModelForCausalLM)
        self._check_generates_as_eager(model, eager_repo)

    def test_causal_lm_merged_decoder(self):
        """A merged decoder, taking a `use_cache_branch` flag (alone, as the repo also holds a `model.onnx`)."""
        with tempfile.TemporaryDirectory() as directory:
            snapshot_download(
                "Xenova/distilgpt2",
                local_dir=directory,
                allow_patterns=["config.json", "generation_config.json", "onnx/decoder_model_merged.onnx"],
            )
            model = _load(self, directory, "optimum-onnx")
            self._check_generates_as_eager(model, "distilbert/distilgpt2")

    def _check_generates_as_eager(self, model, eager_repo):
        tokenizer = AutoTokenizer.from_pretrained(eager_repo)
        inputs = tokenizer(["The capital of France is"], return_tensors="pt")
        generation_config = GenerationConfig(max_new_tokens=8, do_sample=False)
        eager = AutoModelForCausalLM.from_pretrained(eager_repo, dtype=torch.float32).eval()
        with torch.no_grad():
            expected = eager.generate(**inputs, generation_config=generation_config)
        self.assertEqual(model.generate(**inputs, generation_config=generation_config).tolist(), expected.tolist())

    def test_encoder(self):
        _check_encoder(self, "Xenova/bge-base-en-v1.5", "optimum-onnx")

    @parameterized.expand(
        [
            ("Xenova/t5-small", "google-t5/t5-small", "translate English to German: The house is wonderful."),
            ("Xenova/opus-mt-en-de", "Helsinki-NLP/opus-mt-en-de", "The house is wonderful."),
            ("Xenova/blenderbot_small-90M", "facebook/blenderbot_small-90M", "The house is wonderful."),
        ]
    )
    def test_encoder_decoder(self, repo, eager_repo, text):
        """An encoder and a merged decoder, which fills the cross-attention cache on the first step, through
        `generate` as eager's."""
        from transformers import AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(eager_repo)
        inputs = tokenizer([text], return_tensors="pt")
        generation_config = GenerationConfig.from_pretrained(eager_repo)
        generation_config.update(max_new_tokens=16, do_sample=False, num_beams=1)
        model = _load(self, repo, "optimum-onnx")
        self.assertIsInstance(model, ExportedModelForSeq2SeqLM)
        eager = AutoModelForSeq2SeqLM.from_pretrained(eager_repo).eval()
        with torch.no_grad():
            expected = GenerationMixin.generate(eager, **inputs, generation_config=generation_config)
        self.assertEqual(model.generate(**inputs, generation_config=generation_config).tolist(), expected.tolist())

    def test_multimodal(self):
        """onnx-community's multimodal split (text embeddings, a vision encoder, a merged decoder on the embeddings), run as the
        multimodal split: the eager model's answer, token for token."""
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers.exporters import ExportedMultimodalModelForCausalLM
        from transformers.image_utils import load_image

        repo = "HuggingFaceTB/SmolVLM-256M-Instruct"
        processor = AutoProcessor.from_pretrained(repo)
        content = [{"type": "image"}, {"type": "text", "text": "What animals are in the picture?"}]
        text = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
        inputs = processor(text=text, images=image, return_tensors="pt")
        generation_config = GenerationConfig(max_new_tokens=12, do_sample=False)
        model = _load(self, repo, "optimum-onnx")
        self.assertIsInstance(model, ExportedMultimodalModelForCausalLM)
        eager = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.float32).eval()
        with torch.no_grad():
            expected = eager.generate(**inputs, generation_config=generation_config)
        self.assertEqual(model.generate(**inputs, generation_config=generation_config).tolist(), expected.tolist())

    def test_split_decoder_is_refused(self):
        with self.assertRaisesRegex(ValueError, "split in two graphs"):
            ExportedModel.from_pretrained("fxmarty/onnx-tiny-random-gpt2-without-merge")
