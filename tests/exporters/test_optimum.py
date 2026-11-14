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
IRs, optimum-onnx's ONNX graphs and optimum-executorch's programs, against the eager models they were exported from."""

import tempfile
import unittest

import numpy as np
from huggingface_hub import snapshot_download
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


def _load(test, repo, library, trust_remote_code=False):
    with CaptureLogger(optimum.logger) as captured:
        model = ExportedModel.from_pretrained(repo, device="cpu", trust_remote_code=trust_remote_code)
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


def _check_names_the_cats(test, repo, library, processor_repo=None, max_new_tokens=6, trust_remote_code=False):
    """A multimodal export, asked what animal a COCO image shows, answers cats."""
    from transformers.image_utils import load_image

    processor = AutoProcessor.from_pretrained(processor_repo or repo, trust_remote_code=trust_remote_code)
    image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
    content = [{"type": "image"}, {"type": "text", "text": "What animal is in this picture? Answer in one word."}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt")
    model = _load(test, repo, library)
    test.assertIsInstance(model, ExportedMultimodalModelForCausalLM)
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    generated = model.generate(**inputs, generation_config=generation_config)
    answer = processor.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    test.assertIn("cat", answer.lower())


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
        """`token_embedding`, `vision_encoder` and `text_decoder` methods, run as the multimodal split (an int4
        program, so it answers rather than matching eager's logits)."""
        _check_names_the_cats(self, "pytorch/gemma-3-4b-it-HQQ-INT8-INT4", "optimum-executorch")
