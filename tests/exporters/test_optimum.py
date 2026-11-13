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
"""Loading the optimum libraries' exports from the Hub with `ExportedModel.from_pretrained`: optimum-intel's OpenVINO
IRs, optimum-onnx's ONNX graphs and optimum-executorch's programs, against the eager models they were exported from (or, for
tiny random ONNX graphs, the tokens ONNX Runtime generates from them)."""

import unittest

import numpy as np
from parameterized import parameterized

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    GenerationMixin,
)
from transformers.exporters import (
    ExportedModel,
    ExportedModelForCausalLM,
    ExportedModelForSeq2SeqLM,
    ExportedMultimodalModelForCausalLM,
    optimum,
)
from transformers.testing_utils import (
    CaptureLogger,
    require_executorch,
    require_onnxruntime,
    require_onnxscript,
    require_openvino,
    require_torch,
    slow,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


def _load(test, repo, library):
    with CaptureLogger(optimum.logger) as captured:
        model = ExportedModel.from_pretrained(repo, device="cpu")
    test.assertIn(f"is an {library} export", captured.out)
    return model


def _check_encoder(test, repo, library):
    """A BGE export's [CLS] embedding is the eager model's; the graph records no output class, so outputs come back by
    its names."""
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    inputs = dict(tokenizer(["The capital of France is Paris."], return_tensors="pt"))
    model = _load(test, repo, library)
    test.assertIs(type(model), ExportedModel)
    with torch.no_grad():
        expected = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").eval()(**inputs).last_hidden_state[:, 0]
    embedding = model(**inputs)["last_hidden_state"][:, 0].float()
    test.assertGreater(torch.nn.functional.cosine_similarity(embedding, expected, dim=-1).item(), 0.999)


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

    def test_multimodal_is_refused(self):
        with self.assertRaisesRegex(ValueError, "multimodal export"):
            ExportedModel.from_pretrained("OpenVINO/Qwen3-VL-2B-Instruct-fp16-ov", device="cpu")


@slow
@require_torch
@require_onnxscript
@require_onnxruntime
class OptimumOnnxTest(unittest.TestCase):
    @parameterized.expand(
        [
            # One graph taking and returning the cache, its weights split over several `model.onnx_data*` files
            (
                "onnx-internal-testing/tiny-random-LlamaForCausalLM-ONNX_external",
                [19053, 25470, 2323, 21742, 13486, 13934],
            ),
            ("onnx-internal-testing/tiny-random-GPT2LMHeadModel-ONNX", [13, 13, 882, 882, 882, 882]),
            # A merged decoder, taking a `use_cache_branch` flag
            ("fxmarty/onnx-tiny-random-gpt2-with-merge", [13, 13, 13, 13, 13, 469]),
        ]
    )
    def test_causal_lm(self, repo, expected):
        """Tokens a plain ONNX Runtime loop generates from the graph, feeding the cache back by optimum-onnx's names."""
        model = _load(self, repo, "optimum-onnx")
        self.assertIsInstance(model, ExportedModelForCausalLM)
        input_ids = torch.tensor([[1, 5, 7, 9, 11, 13]])
        generation_config = GenerationConfig(max_new_tokens=6, do_sample=False, eos_token_id=None)
        generated = model.generate(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), generation_config=generation_config
        )
        self.assertEqual(generated[0, 6:].tolist(), expected)

    def test_encoder(self):
        _check_encoder(self, "Xenova/bge-base-en-v1.5", "optimum-onnx")

    def test_encoder_decoder(self):
        """An encoder and a merged decoder, which fills the cross-attention cache on the first step, through
        `generate` as eager's."""
        from transformers import T5ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        inputs = tokenizer(["translate English to German: The house is wonderful."], return_tensors="pt")
        generation_config = GenerationConfig(max_new_tokens=16, do_sample=False, decoder_start_token_id=0)
        model = _load(self, "Xenova/t5-small", "optimum-onnx")
        self.assertIsInstance(model, ExportedModelForSeq2SeqLM)
        eager = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").eval()
        with torch.no_grad():
            expected = GenerationMixin.generate(eager, **inputs, generation_config=generation_config)
        self.assertEqual(model.generate(**inputs, generation_config=generation_config).tolist(), expected.tolist())

    def test_split_decoder_is_refused(self):
        with self.assertRaisesRegex(ValueError, "split in two graphs"):
            ExportedModel.from_pretrained("fxmarty/onnx-tiny-random-gpt2-without-merge")


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
        """`token_embedding`, `vision_encoder` and `text_decoder` methods, run as the multimodal split: an int4
        program, so it answers rather than matching eager's logits."""
        from transformers.image_utils import load_image

        repo = "pytorch/gemma-3-4b-it-HQQ-INT8-INT4"
        processor = AutoProcessor.from_pretrained(repo)
        image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
        content = [{"type": "image"}, {"type": "text", "text": "What animal is in this picture? Answer in one word."}]
        text = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        inputs = processor(text=text, images=image, return_tensors="pt")
        model = _load(self, repo, "optimum-executorch")
        self.assertIsInstance(model, ExportedMultimodalModelForCausalLM)
        generated = model.generate(**inputs, generation_config=GenerationConfig(max_new_tokens=6, do_sample=False))
        answer = processor.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertIn("cat", answer.lower())
