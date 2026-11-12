# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Inkling2 model."""

import unittest

from parameterized import parameterized

from transformers import (
    AutoProcessor,
    Inkling2Config,
    Inkling2TextConfig,
    Inkling2VisionConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    require_deterministic_for_xpu,
    require_torch,
    require_torch_accelerator,
    require_torch_accelerator_memory,
    require_vision,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION,
    _test_eager_matches_batched_and_grouped_inference,
    floats_tensor,
)
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Inkling2ForCausalLM,
        Inkling2ForConditionalGeneration,
        Inkling2Model,
        Inkling2TextModel,
    )


class Inkling2TextModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Inkling2TextConfig
        base_model_class = Inkling2TextModel
        causal_lm_class = Inkling2ForCausalLM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_types = ["hybrid_sliding", "hybrid"]
        self.swa_num_attention_heads = self.num_attention_heads
        self.swa_num_key_value_heads = self.num_key_value_heads
        self.swa_head_dim = self.head_dim
        self.n_routed_experts = 8


@require_torch
class Inkling2TextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Inkling2TextModelTester

    def _get_conv_state_shape(self, batch_size: int, config):
        # key and value convolutions, then the attention output and MLP output convolutions
        kv_size = config.num_key_value_heads * config.head_dim
        return [(batch_size, kv_size, config.conv_kernel_size)] * 2 + [
            (batch_size, config.hidden_size, config.conv_kernel_size)
        ] * 2

    @unittest.skip("Chained renames leave no intermediate key after the reverse mapping")
    def test_reverse_loading_mapping(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_batched_and_grouped_inference(self, name, dtype):
        # batched_mm runs one matrix-vector product per token, whose float32 sums round differently in low precision
        if dtype in ("fp16", "bf16"):
            self.skipTest("Low-precision rounding differences across expert kernels exceed the 1e-4 tolerance")
        _test_eager_matches_batched_and_grouped_inference(self, name, dtype)


class Inkling2VisionText2TextModelTester(VLMModelTester):
    base_model_class = Inkling2Model
    config_class = Inkling2Config
    text_config_class = Inkling2TextConfig
    vision_config_class = Inkling2VisionConfig
    conditional_generation_class = Inkling2ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 4)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("spatial_merge_size", 2)
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("posemb_grid_size", 4)
        kwargs.setdefault("projection_hidden_dim", 32)
        kwargs.setdefault("layer_types", ["hybrid_sliding", "hybrid"])
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("n_routed_experts", 8)
        kwargs.setdefault("swa_num_attention_heads", 2)
        kwargs.setdefault("swa_num_key_value_heads", 2)
        kwargs.setdefault("swa_head_dim", 16)
        super().__init__(parent, **kwargs)
        self.text_hidden_size = self.hidden_size

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        side = self.patch_size * self.spatial_merge_size
        return floats_tensor([batch_size * self.num_image_tokens, side, side, self.num_channels])

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {"image_grid_thw": torch.tensor([[1, 2, 2]] * batch_size, device=torch_device)}


@require_torch
class Inkling2VisionText2TextModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Inkling2VisionText2TextModelTester
    # each image adds a class token and `spatial_merge_size**2` vision tokens per `pixel_values` row
    skip_test_image_features_output_shape = True

    def _get_conv_state_shape(self, batch_size: int, config):
        # key and value convolutions, then the attention output and MLP output convolutions
        kv_size = config.num_key_value_heads * config.head_dim
        return [(batch_size, kv_size, config.conv_kernel_size)] * 2 + [
            (batch_size, config.hidden_size, config.conv_kernel_size)
        ] * 2

    @unittest.skip("Chained renames leave no intermediate key after the reverse mapping")
    def test_reverse_loading_mapping(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_batched_and_grouped_inference(self, name, dtype):
        # batched_mm runs one matrix-vector product per token, whose float32 sums round differently in low precision
        if dtype in ("fp16", "bf16"):
            self.skipTest("Low-precision rounding differences across expert kernels exceed the 1e-4 tolerance")
        _test_eager_matches_batched_and_grouped_inference(self, name, dtype)

    @unittest.skip(
        "Inkling2Model does not support SDPA: text attention adds a relative position bias and a per-token sink"
    )
    def test_can_set_attention_dynamically_composite_model(self):
        pass


# Differences to the original checkpoint:
# - decoder layers: 12 instead of 48 (original layers 0-5 with NVFP4 experts and 42-47 with MXFP8 experts)
# - text: hidden size 256 instead of 6144, 16 experts (top-4) instead of 384
# - vision: hidden size 128, 2 layers instead of 23
#
# Expectations are gibberish as the model is not trained.
@slow
@require_torch_accelerator
class Inkling2SmallModelIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "hf-internal-testing/tiny-inkling2"

    MAX_NEW_TOKENS = 20

    image_url = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
    )

    @classmethod
    def setUpClass(cls):
        cls.model = None

    @classmethod
    def get_model(cls):
        # loaded on first use and shared by the whole class
        if cls.model is None:
            cls.model = Inkling2ForConditionalGeneration.from_pretrained(
                cls.model_id, dtype=torch.bfloat16, device_map=torch_device
            )
        return cls.model

    def setUp(self):
        super().setUp()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _generate(self, messages, **kwargs):
        model = self.get_model()
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **kwargs
        )
        inputs = inputs.to(model.device).to(model.dtype)
        output = model.generate(
            **inputs,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )
        completion = self.processor.decode(
            output.sequences[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        return output, completion

    @require_deterministic_for_xpu
    def test_small_model_integration_test_text(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Explain in one sentence why the sky is usually blue."}],
            }
        ]
        output, completion = self._generate(messages, reasoning_effort="none")

        EXPECTED_LOGITS = Expectations(
            {
                ("cuda", 8): [0.4901907444000244, 0.7303244471549988, 1.3194607496261597, 0.2734299600124359, 0.4550432562828064],
                ("cuda", 10): [0.49409592151641846, 0.7420361638069153, 1.3116633892059326, 0.27733585238456726, 0.47066453099250793],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(output.logits[0][0, :5].float(), expected_logits, atol=1e-4, rtol=1e-4)

        EXPECTED_COMPLETION = Expectations(
            {
                ("cuda", 10): "Susie積臧臧臧臧芃Susie積臧臧臧芃Susie積臧臧臧芃Susie",
            }
        )  # fmt: skip
        self.assertEqual(completion, EXPECTED_COMPLETION.get_expectation())

    @require_vision
    @require_deterministic_for_xpu
    def test_small_model_integration_test_image(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url_to_local_path(self.image_url)},
                    {"type": "text", "text": "What animal is in this image, and where is it?"},
                ],
            }
        ]
        output, completion = self._generate(messages)

        EXPECTED_LOGITS = Expectations(
            {
                ("cuda", 8): [0.8591400384902954, 0.5780534744262695, 1.2648746967315674, -0.16796699166297913, 0.6171004772186279],
                ("cuda", 10): [0.8747519850730896, 0.5780534744262695, 1.27267324924469, -0.16992005705833435, 0.6171004772186279],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(output.logits[0][0, :5].float(), expected_logits, atol=1e-4, rtol=1e-4)

        EXPECTED_COMPLETION = Expectations(
            {
                ("cuda", 8): "audreyانہAdelaide盅природ瑮YanYanYanYanYanYanYanYanYanYanYanYanYanYan",
                ("cuda", 10): " Emb Emb Emb Emb Emb Emb Emb Emb Emb Emb Emb Emb Emb Emb Emb Emb Emb Emb Emb Emb",
            }
        )  # fmt: skip
        self.assertEqual(completion, EXPECTED_COMPLETION.get_expectation())


@slow
@require_torch_accelerator
@require_torch_accelerator_memory(memory=1400)  # 8x B200, with the experts dequantized to bfloat16
class Inkling2IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    r"""Greedy completions agree with the reference implementation's (sglang, TP=8, EP=8) on the same prompts: the
    image one for all 30 tokens, the text one for the first 24 ("... than longer red wavelengths"), where sglang ends
    the sentence and Transformers continues (", a process known as") on a near tie. Their logits differ by ~0.2 on
    average (bfloat16 rounding, amplified through the 48 layers), so the logits expectations are Transformers' own.

    Run from the repository root on a node with eight B200 GPUs::

        RUN_SLOW=1 pytest tests/models/inkling2/test_modeling_inkling2.py::Inkling2IntegrationTest -s
    """

    model_id = "thinkingmachines/Inkling-2-Small-MXFP8"

    MAX_NEW_TOKENS = 30

    image_url = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
    )

    @classmethod
    def setUpClass(cls):
        cls.model = Inkling2ForConditionalGeneration.from_pretrained(
            cls.model_id, dtype=torch.bfloat16, device_map="auto"
        ).eval()
        cls.processor = AutoProcessor.from_pretrained(cls.model_id)

    def _generate(self, messages, **kwargs):
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **kwargs
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )
        completion = self.processor.decode(
            output.sequences[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        return output, completion

    def test_integration_text(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Explain in one sentence why the sky is usually blue."}],
            }
        ]
        output, completion = self._generate(messages, reasoning_effort="none")

        EXPECTED_LOGITS = Expectations(
            {
                ("cuda", 10): [-16.551239013671875, -15.015604972839355, -17.647777557373047, -13.765021324157715, -17.565834045410156],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(output.logits[0][0, :5].float(), expected_logits, atol=1e-4, rtol=1e-4)

        EXPECTED_COMPLETION = Expectations(
            {
                ("cuda", 10): (
                    "<|message_model|><|content_text|>The sky appears blue because air molecules scatter shorter blue "
                    "wavelengths of sunlight more effectively than longer red wavelengths, a process known as "
                ),
            }
        )  # fmt: skip
        self.assertEqual(completion, EXPECTED_COMPLETION.get_expectation())

    @require_vision
    def test_integration_image(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url_to_local_path(self.image_url)},
                    {"type": "text", "text": "What animal is in this image, and where is it?"},
                ],
            }
        ]
        output, completion = self._generate(messages)

        EXPECTED_LOGITS = Expectations(
            {
                ("cuda", 10): [-17.81046485900879, -17.400745391845703, -12.554040908813477, -14.590080261230469, -13.467277526855469],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(output.logits[0][0, :5].float(), expected_logits, atol=1e-4, rtol=1e-4)

        EXPECTED_COMPLETION = Expectations(
            {
                ("cuda", 10): (
                    "<|message_model|><|content_thinking|>The user is asking to identify the animal and location in "
                    "the image. Need answer based on visible content. The animal looks like a Pallas"
                ),
            }
        )  # fmt: skip
        self.assertEqual(completion, EXPECTED_COMPLETION.get_expectation())
