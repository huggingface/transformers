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
        Inkling2LoopForCausalLM,
        Inkling2LoopForConditionalGeneration,
        Inkling2LoopModel,
        Inkling2LoopTextModel,
    )
    from transformers.cache_utils import DynamicCache, StaticCache
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopPastReader


def check_loop_cache(test, batch_size, cache, seq_length, config, check_cache):
    """The cache has the layers of the physical decoder layers, then one per layer of every extra pass through the loop
    layers, shaped like the cache layer of the same loop layer."""
    text_config = config.get_text_config(decoder=True)
    loop_layers = range(text_config.loop_layer_index_start, text_config.loop_layer_index_end)
    pass_layers = [layer_idx for _ in range(text_config.loop_num_passes - 1) for layer_idx in loop_layers]
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

    small_config_overrides = {
        "num_hidden_layers": 4,
        "layer_types": ["hybrid_sliding", "hybrid_sliding", "hybrid", "hybrid_sliding"],
        "n_layers_per_attn_res_block": 1,
        "loop_layer_index_start": 1,
        "loop_layer_index_end": 3,
        "loop_num_passes": 3,
        "loop_gate_rank": 8,
        "n_routed_experts": 8,
        # hidden states are recorded at every layer visit (4 layers + 2 more passes through the 2 loop layers) and
        # for the embeddings
        "expected_num_hidden_layers": 4 + 2 * 2 + 1,
    }

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        values["swa_num_attention_heads"] = values["num_attention_heads"]
        values["swa_num_key_value_heads"] = values["num_key_value_heads"]
        values["swa_head_dim"] = values["head_dim"]
        values["unpadded_vocab_size"] = values["vocab_size"]
        return values


@require_torch
class Inkling2LoopTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Inkling2LoopTextModelTester
    # `copy.deepcopy` of the cache layers appended for the loop layers in `forward` fails on export's fake tensors
    test_torch_exportable = False

    @unittest.skip("Attentions are returned per layer visit: the loop layers run several times per forward")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Chained renames leave no intermediate key after the reverse mapping")
    def test_reverse_loading_mapping(self):
        pass

    def _check_cache_for_generate(self, batch_size, cache, seq_length, config):
        check_loop_cache(self, batch_size, cache, seq_length, config, super()._check_cache_for_generate)

    def _get_conv_state_shape(self, batch_size: int, config):
        # key and value convolutions, then the attention output and MLP output convolutions
        kv_size = config.num_key_value_heads * config.head_dim
        return [(batch_size, kv_size, config.conv_kernel_size)] * 2 + [
            (batch_size, config.hidden_size, config.conv_kernel_size)
        ] * 2

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_batched_and_grouped_inference(self, name, dtype):
        # batched_mm runs one matrix-vector product per token, whose float32 sums round differently in low precision
        if dtype in ("fp16", "bf16"):
            self.skipTest("Low-precision rounding differences across expert kernels exceed the 1e-4 tolerance")
        _test_eager_matches_batched_and_grouped_inference(self, name, dtype)

    def test_last_loop_single_forward_matches_per_loop(self):
        # With no past, the readers return the passes' own states, so the first forward matches across modes
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["input_ids"]
        model = Inkling2LoopForCausalLM(config).to(torch_device).eval()
        with torch.no_grad():
            logits = {}
            for mode in ("per_loop", "last_loop"):
                config.recurrent_kv_cache_mode = mode
                logits[mode] = model(input_ids, cache=DynamicCache(config=config)).logits
        torch.testing.assert_close(logits["last_loop"], logits["per_loop"], rtol=1e-4, atol=1e-4)

    def test_static_cache_matches_dynamic_with_loops(self):
        # `last_loop` passes before the last read the cache without writing, which must match across cache types,
        # also past the sliding window
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.sliding_window_size = 4
        input_ids = inputs_dict["input_ids"][:1].repeat(1, 2)[:, :12]
        for mode in ("per_loop", "last_loop"):
            with self.subTest(mode=mode):
                config.recurrent_kv_cache_mode = mode
                model = Inkling2LoopForCausalLM(config).to(torch_device).eval()
                dynamic = DynamicCache(config=config)
                static = StaticCache(config=config, max_cache_len=16)
                with torch.no_grad():
                    for start, end in ((0, 2), (2, 4), (4, 7), (7, 8), (8, 12)):
                        expected = model(input_ids[:, start:end], cache=dynamic).logits
                        actual = model(input_ids[:, start:end], cache=static).logits
                        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_generate_with_static_cache_past_sliding_window(self):
        # `generate` prebuilds the causal masks: the sliding layers of hybrid models must get a sliding mask, not a
        # full one, or the static buffer length breaks the attention on prefills longer than the window
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.sliding_window_size = 4
        input_ids = inputs_dict["input_ids"][:, :12]
        model = Inkling2LoopForCausalLM(config).to(torch_device).eval()
        dynamic = model.generate(input_ids, max_new_tokens=8, do_sample=False, cache=DynamicCache(config=config))
        for mode in ("per_loop", "last_loop"):
            with self.subTest(mode=mode):
                config.recurrent_kv_cache_mode = mode
                static = model.generate(
                    input_ids,
                    max_new_tokens=8,
                    do_sample=False,
                    cache=StaticCache(config=config, max_cache_len=20),
                )
                torch.testing.assert_close(static, dynamic, rtol=0, atol=0)

    def test_last_loop_readers_hold_no_kv(self):
        # The extra cache layers are read-only views of the loop layers: pass-local conv states, no KV of their own
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.recurrent_kv_cache_mode = "last_loop"
        text_config = config.get_text_config(decoder=True)
        num_loop_layers = text_config.loop_layer_index_end - text_config.loop_layer_index_start
        model = Inkling2LoopForCausalLM(config).to(torch_device).eval()
        cache = DynamicCache(config=config)
        with torch.no_grad():
            model(inputs_dict["input_ids"], cache=cache)
        readers = cache.layers[text_config.num_hidden_layers :]
        self.assertEqual(len(readers), (text_config.loop_num_passes - 1) * num_loop_layers)
        for reader, layer_idx in zip(
            readers, list(range(text_config.loop_layer_index_start, text_config.loop_layer_index_end)) * 2
        ):
            self.assertIs(type(reader), Inkling2LoopPastReader)
            self.assertIsNone(reader.keys)
            self.assertIs(reader.layer, cache.layers[layer_idx])
            self.assertTrue(any(state is not None for state in reader.conv_states.values()))


class Inkling2LoopVisionText2TextModelTester(VLMModelTester):
    base_model_class = Inkling2LoopModel
    config_class = Inkling2LoopConfig
    text_config_class = Inkling2LoopTextConfig
    vision_config_class = Inkling2VisionConfig
    conditional_generation_class = Inkling2LoopForConditionalGeneration

    small_config_overrides = {
        "image_token_id": 4,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "num_image_tokens": 4,
        "posemb_grid_size": 4,
        "projection_hidden_dim": 32,
        "text_num_hidden_layers": 4,
        "layer_types": ["hybrid_sliding", "hybrid_sliding", "hybrid", "hybrid_sliding"],
        "n_layers_per_attn_res_block": 1,
        "loop_layer_index_start": 1,
        "loop_layer_index_end": 3,
        "loop_num_passes": 3,
        "loop_gate_rank": 8,
        "expected_num_hidden_layers": 4 + 2 * 2 + 1,
        "moe_intermediate_size": 16,
        "n_routed_experts": 8,
        "swa_num_attention_heads": 2,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 16,
    }

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        values["text_hidden_size"] = values["hidden_size"]
        values["unpadded_vocab_size"] = values["vocab_size"]
        return values

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
    # `copy.deepcopy` of the cache layers appended for the loop layers in `forward` fails on export's fake tensors
    test_torch_exportable = False

    # `last_hidden_state` holds `spatial_merge_size**2` patch states per `pixel_values` row plus a class token per image,
    # but the common test expects one state per `pixel_values` row
    skip_test_image_features_output_shape = True

    @unittest.skip("Attentions are returned per layer visit: the loop layers run several times per forward")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Chained renames leave no intermediate key after the reverse mapping")
    def test_reverse_loading_mapping(self):
        pass

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
class Inkling2LoopSmallModelIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    r"""Greedy completions and first-step logits of a tiny checkpoint sliced from the original one.
    Differences to the original checkpoint:

    - decoder layers: 18 instead of 24 (original layers 0-11 and 18-23), with a single attn-res block of loop layers
      (6-11) instead of two (6-17); NVFP4 experts on layers 0-13 and MXFP8 experts on layers 14-17
    - text: hidden size 256 instead of 2304, 16 experts (top-4) instead of 128
    - vision: hidden size 128, 2 layers instead of 23

    Expectations are gibberish as the model is not trained.
    """

    model_id = "hf-internal-testing/tiny-inkling2-mini"

    MAX_NEW_TOKENS = 20

    image_url = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
    )

    @classmethod
    def setUpClass(cls):
        cls.model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = Inkling2LoopForConditionalGeneration.from_pretrained(
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
                ("cuda", 8): [1.9270305633544922, 0.287100613117218, 3.4533743858337402, 2.711331605911255, 0.878654956817627],
                ("cuda", 10): [1.9970425367355347, 0.26171210408210754, 3.4687917232513428, 2.6803338527679443, 0.9059745073318481],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(output.logits[0][0, :5].float(), expected_logits, atol=1e-4, rtol=1e-4)

        EXPECTED_COMPLETION = Expectations(
            {
                ("cuda", 8): "floursigne턽 перек перек перек перек перек перек перек перек перек перек перек перек перек перек перек перек перек",
                ("cuda", 10): "floursigne턽 перек перек перек перек перек перек перек перек перек перек перек перек перек перек перек перек перек",
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
                ("cuda", 8): [0.4433271288871765, -1.3506488800048828, 3.853593111038208, 0.593672513961792, 1.5766711235046387],
                ("cuda", 10): [0.4452798366546631, -1.3506488800048828, 3.8689591884613037, 0.6053866147994995, 1.5688800811767578],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(output.logits[0][0, :5].float(), expected_logits, atol=1e-4, rtol=1e-4)

        EXPECTED_COMPLETION = Expectations(
            {
                ("cuda", 8): " Studyпсихијатventovécudostępучитель�SenStudieпсихијатventoperdere● Studyпсихијатventoperdere",
                ("cuda", 10): " Studyпсихијатventovécudostępучитель�SenStudieпсихијатventoperdere● Studyпсихијатventoperdere",
            }
        )  # fmt: skip
        self.assertEqual(completion, EXPECTED_COMPLETION.get_expectation())


@slow
@require_torch_accelerator
@require_torch_accelerator_memory(memory=80)  # ~50 GiB of weights with the experts dequantized to bfloat16
class Inkling2LoopIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    r"""Greedy completions and first-step logits of the reference implementation (sglang, one B200, experts
    dequantized to bfloat16) on the prompts of the Inkling2 integration tests.

    Run from the repository root on a GPU with at least 80 GiB::

        RUN_SLOW=1 pytest tests/models/inkling2_loop/test_modeling_inkling2_loop.py::Inkling2LoopIntegrationTest -s
    """

    model_id = "thinkingmachines/Inkling-2-Mini"

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
                ("cuda", 10): [-12.755270957946777, -11.535194396972656, -16.171504974365234, -17.790428161621094, -19.750545501708984],
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
                ("cuda", 10): [-3.7679390907287598, -3.2080488204956055, -4.347489356994629, -9.261019706726074, -11.941338539123535],
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
