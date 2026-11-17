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
"""Testing suite for the PyTorch Inkling2Loop model."""

import copy
import unittest

from parameterized import parameterized

from transformers import (
    AutoProcessor,
    Inkling2LoopConfig,
    Inkling2LoopTextConfig,
    Inkling2VisionConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
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
        Inkling2LoopForCausalLM,
        Inkling2LoopForConditionalGeneration,
        Inkling2LoopModel,
        Inkling2LoopTextModel,
    )


# Four layers with one attention residual block each: a prelude (layer 0), a core of one sliding-window and one
# full-attention layer (layers 1-2) that runs three times, and a coda (layer 3). Three passes, so that the last pass
# reads the messages of the second pass, not of the first.
LAYER_TYPES = ["hybrid_sliding", "hybrid_sliding", "hybrid", "hybrid_sliding"]
LOOP_CONFIG = {"core_start": 1, "core_end": 3, "num_passes": 3, "gate_rank": 8}
# hidden states are recorded at every layer visit: the 4 layers and 2 more passes through the 2 core layers
NUM_LAYER_VISITS = len(LAYER_TYPES) + 2 * 2


def check_loop_cache(test, batch_size, cache, seq_length, config, check_cache):
    """The cache has the layers of the physical decoder layers, then one per layer of every extra core pass, shaped like
    the cache layer of the same core layer."""
    text_config = config.get_text_config(decoder=True)
    loop_config = text_config.loop_config
    core = range(loop_config["core_start"], loop_config["core_end"])
    pass_layers = [layer_idx for _ in range(loop_config["num_passes"] - 1) for layer_idx in core]
    test.assertEqual(len(cache), text_config.num_hidden_layers + len(pass_layers))

    physical_cache = copy.copy(cache)
    physical_cache.layers = cache.layers[: text_config.num_hidden_layers]
    check_cache(batch_size, physical_cache, seq_length, config)
    for pass_layer, layer_idx in zip(cache.layers[text_config.num_hidden_layers :], pass_layers):
        test.assertIs(type(pass_layer), type(cache.layers[layer_idx]))
        test.assertEqual(pass_layer.keys.shape, cache.layers[layer_idx].keys.shape)
        test.assertEqual(
            [state.shape for state in pass_layer.conv_states.values()],
            [state.shape for state in cache.layers[layer_idx].conv_states.values()],
        )


class Inkling2LoopTextModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Inkling2LoopTextConfig
        base_model_class = Inkling2LoopTextModel
        causal_lm_class = Inkling2LoopForCausalLM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hidden_layers = len(LAYER_TYPES)
        self.layer_types = LAYER_TYPES
        self.n_layers_per_attn_res_block = 1
        self.loop_config = LOOP_CONFIG
        self.expected_num_hidden_layers = NUM_LAYER_VISITS + 1
        self.swa_num_attention_heads = self.num_attention_heads
        self.swa_num_key_value_heads = self.num_key_value_heads
        self.swa_head_dim = self.head_dim
        self.n_routed_experts = 8


@require_torch
class Inkling2LoopTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Inkling2LoopTextModelTester

    @unittest.skip("Attentions are returned per layer visit: the loop core runs several times per forward")
    def test_attention_outputs(self):
        pass

    def _check_cache_for_generate(self, batch_size, cache, seq_length, config):
        check_loop_cache(self, batch_size, cache, seq_length, config, super()._check_cache_for_generate)

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


class Inkling2LoopVisionText2TextModelTester(VLMModelTester):
    base_model_class = Inkling2LoopModel
    config_class = Inkling2LoopConfig
    text_config_class = Inkling2LoopTextConfig
    vision_config_class = Inkling2VisionConfig
    conditional_generation_class = Inkling2LoopForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 4)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("spatial_merge_size", 2)
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("posemb_grid_size", 4)
        kwargs.setdefault("projection_hidden_dim", 32)
        kwargs.setdefault("text_num_hidden_layers", len(LAYER_TYPES))
        kwargs.setdefault("expected_num_hidden_layers", NUM_LAYER_VISITS + 1)
        kwargs.setdefault("layer_types", LAYER_TYPES)
        kwargs.setdefault("n_layers_per_attn_res_block", 1)
        kwargs.setdefault("loop_config", LOOP_CONFIG)
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
class Inkling2LoopVisionText2TextModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Inkling2LoopVisionText2TextModelTester

    @unittest.skip("Attentions are returned per layer visit: the loop core runs several times per forward")
    def test_attention_outputs(self):
        pass

    # each image adds a class token and `spatial_merge_size**2` vision tokens per `pixel_values` row
    skip_test_image_features_output_shape = True

    def _check_cache_for_generate(self, batch_size, cache, seq_length, config):
        check_loop_cache(self, batch_size, cache, seq_length, config, super()._check_cache_for_generate)

    def _image_features_get_expected_num_attentions(self, model_tester=None):
        return self.model_tester.num_hidden_layers

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
        "Inkling2LoopModel does not support SDPA: text attention adds a relative position bias and a per-token sink"
    )
    def test_can_set_attention_dynamically_composite_model(self):
        pass


@slow
@require_torch_accelerator
@require_torch_accelerator_memory(memory=80)  # ~50 GiB of weights with the experts dequantized to bfloat16
class Inkling2LoopIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    r"""Greedy completions and first-step logits of the reference implementation (sglang, one B200, experts
    dequantized to bfloat16) on the prompts of the Inkling2 integration tests.

    The logits are sglang's, hence the loose tolerance: on Inkling2, Transformers' and sglang's first-step logits[:5]
    differed by up to 1.0 (bfloat16 rounding, amplified through 48 layer visits). Replace them with Transformers' own
    values and a 1e-4 tolerance once the implementation agrees with the reference. The text completion has a near tie
    at its 9th token ("explanation", ahead of the runner-up by 0.12), where Transformers may legitimately diverge.

    Run from the repository root on a GPU with at least 80 GiB::

        RUN_SLOW=1 pytest tests/models/inkling2_loop/test_modeling_inkling2_loop.py::Inkling2LoopIntegrationTest -s
    """

    model_id = "thinkingmachines/inkling-1116-share"

    MAX_NEW_TOKENS = 30

    image_url = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
    )

    @classmethod
    def setUpClass(cls):
        cls.model = Inkling2LoopForConditionalGeneration.from_pretrained(
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
                ("cuda", 10): [-14.193360328674316, -12.197671890258789, -17.577695846557617, -18.56708335876465, -21.122817993164062],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(output.logits[0][0, :5].float(), expected_logits, atol=1.5, rtol=0)

        EXPECTED_COMPLETION = Expectations(
            {
                ("cuda", 10): (
                    "<|message_model|><|content_thinking|>The user asks for one sentence explanation why sky is "
                    "usually blue. Need answer concise. Need ensure accuracy: Rayleigh scattering: shorter blue"
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
                ("cuda", 10): [-3.9177968502044678, -3.2600231170654297, -4.5490851402282715, -9.422932624816895, -11.752306938171387],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(output.logits[0][0, :5].float(), expected_logits, atol=1.5, rtol=0)

        EXPECTED_COMPLETION = Expectations(
            {
                ("cuda", 10): (
                    "<|message_model|><|content_thinking|>The user is asking what animal is in the image and where it "
                    "is. Let me analyze the image.\n\nThe image shows a stocky"
                ),
            }
        )  # fmt: skip
        self.assertEqual(completion, EXPECTED_COMPLETION.get_expectation())
