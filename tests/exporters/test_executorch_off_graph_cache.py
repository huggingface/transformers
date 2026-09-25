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
"""Contract tests for ExecuTorch off-graph cache export."""

import unittest
from unittest import mock

from parameterized import parameterized

from transformers.exporters import utils as exporter_utils
from transformers.exporters.configs import ExecutorchConfig
from transformers.exporters.exporter_executorch import _BACKEND_PREPARE, _OFF_GRAPH_CACHE_BACKENDS
from transformers.testing_utils import require_executorch, require_torch
from transformers.utils.import_utils import is_torch_available


if is_torch_available():
    import torch

    from transformers import GenerationConfig


@require_torch
@require_executorch
class ExecutorchOffGraphCacheRejectionTest(unittest.TestCase):
    def setUp(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers.exporters import exporter_executorch

        self.exporter_module = exporter_executorch
        self.model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                attn_implementation="eager",
            )
        ).eval()

    def test_rejects_attention_sinks_before_cache_update(self):
        from transformers import GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention

        attention = GptOssAttention(
            GptOssConfig(
                hidden_size=16,
                head_dim=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                layer_types=["full_attention"],
            ),
            layer_idx=0,
        ).eval()
        query = torch.zeros(1, 2, 1, 8)
        key = torch.zeros(1, 1, 1, 8)
        with mock.patch.object(
            torch.ops.kvcache, "update_and_attend", return_value=torch.zeros_like(query), create=True
        ) as update_and_attend:
            with self.assertRaisesRegex(ValueError, "sink|s_aux"):
                self.exporter_module._executorch_off_graph_cache_attention_forward(
                    attention,
                    query,
                    key,
                    torch.ones_like(key),
                    attention_mask=None,
                    position_ids=torch.zeros(1, 1, dtype=torch.long),
                    scaling=attention.scaling,
                    s_aux=attention.sinks,
                )
            update_and_attend.assert_not_called()

    def test_rejects_noncausal_model_config(self):
        validate_model = self.exporter_module._validate_executorch_off_graph_cache_model
        try:
            validate_model(self.model)
        except ImportError:
            self.skipTest("Requires ExecuTorch's off-graph cache extension")

        self.model.config.is_causal = False
        # Llama's attention module still reports True; the config controls its mask.
        self.assertTrue(self.model.model.layers[0].self_attn.is_causal)
        with self.assertRaisesRegex(ValueError, "causal"):
            validate_model(self.model)

    @parameterized.expand(sorted(_OFF_GRAPH_CACHE_BACKENDS))
    def test_rejects_masked_inputs_alias_before_capture(self, backend):
        exporter = self.exporter_module.ExecutorchExporter()
        inputs = {
            "inputs": torch.tensor([[3, 4, 5]]),
            "attention_mask": torch.tensor([[0, 1, 1]]),
            "position_ids": torch.tensor([[0, 1, 2]]),
        }
        config = ExecutorchConfig(backend=backend, cache_implementation="executorch_off_graph_cache")
        with mock.patch.object(exporter_utils, "decompose_for_generation", return_value={}) as capture:
            with self.assertRaisesRegex(ValueError, "mask|padding|input"):
                exporter.export_for_generation(
                    self.model,
                    inputs,
                    config=config,
                    generation_config=GenerationConfig(cache_implementation="dynamic", do_sample=False),
                )
            capture.assert_not_called()

    def _assert_rejected_before_capture(self, backend, inputs, message, **generation_kwargs):
        config = ExecutorchConfig(backend=backend, cache_implementation="executorch_off_graph_cache")
        generation_config = GenerationConfig(cache_implementation="dynamic", **generation_kwargs)
        with mock.patch.object(exporter_utils, "decompose_for_generation", return_value={}) as capture:
            with self.assertRaisesRegex(ValueError, message):
                self.exporter_module.ExecutorchExporter().export_for_generation(
                    self.model, inputs, config=config, generation_config=generation_config
                )
            capture.assert_not_called()

    @parameterized.expand(sorted(_OFF_GRAPH_CACHE_BACKENDS))
    def test_rejects_unsupported_model_contract_before_capture(self, backend):
        inputs = {"input_ids": torch.tensor([[3, 4, 5]])}
        cases = [
            (self.model, "_supports_attention_backend", False, "attention interface"),
            (self.model, "_supports_sdpa", False, "SDPA"),
            (self.model.config, "is_causal", False, "causal"),
        ]
        for owner, name, value, message in cases:
            with self.subTest(name=name), mock.patch.object(owner, name, value, create=True):
                self._assert_rejected_before_capture(backend, inputs, message)
        with mock.patch.multiple(
            self.model.config, layer_types=["chunked_attention"], attention_chunk_size=4, create=True
        ):
            self._assert_rejected_before_capture(backend, inputs, "full or sliding")

    @parameterized.expand(sorted(_OFF_GRAPH_CACHE_BACKENDS))
    def test_rejects_unsupported_requests_before_capture(self, backend):
        from transformers import DynamicCache

        base = {"input_ids": torch.tensor([[3, 4, 5]])}
        cases = [
            ({"inputs": base["input_ids"]}, "inputs alias"),
            ({}, "input_ids or inputs_embeds"),
            ({"input_ids": torch.tensor([[3, 4], [5, 6]])}, "single input sequence"),
            ({"input_ids": torch.tensor([3, 4, 5])}, "single input sequence"),
            ({**base, "attention_mask": torch.tensor([[0, 1, 1]])}, "padding or custom"),
            ({**base, "attention_mask": torch.ones(1, 1, 3, 3)}, "padding or custom"),
            ({**base, "position_ids": torch.tensor([[0, 2, 3]])}, "contiguous position_ids"),
            ({**base, "position_ids": torch.tensor([[0.0, 1.0, 2.0]])}, "contiguous position_ids"),
            ({**base, "past_key_values": DynamicCache(config=self.model.config)}, "existing cache"),
        ]
        for inputs, message in cases:
            with self.subTest(inputs=inputs):
                self._assert_rejected_before_capture(backend, inputs, message)

    @parameterized.expand(sorted(_OFF_GRAPH_CACHE_BACKENDS))
    def test_rejects_unsupported_generation_modes_before_capture(self, backend):
        inputs = {"input_ids": torch.tensor([[3, 4, 5]])}
        cases = [
            ({"num_beams": 2}, "beam expansion"),
            ({"num_return_sequences": 2, "do_sample": True}, "one sequence"),
            ({"is_assistant": True}, "speculative generation"),
            ({"use_cache": False}, "use_cache=True"),
            ({"output_attentions": True}, "attention weights"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                self._assert_rejected_before_capture(backend, inputs, message, **kwargs)

    @parameterized.expand(sorted(_OFF_GRAPH_CACHE_BACKENDS))
    def test_accepts_canonical_generation_requests(self, backend):
        config = ExecutorchConfig(backend=backend, cache_implementation="executorch_off_graph_cache")
        for name, tokens in (
            ("input_ids", torch.tensor([[3, 4, 5]])),
            ("inputs_embeds", torch.zeros(1, 3, self.model.config.hidden_size)),
        ):
            for do_sample in (False, True):
                with self.subTest(name=name, do_sample=do_sample):
                    inputs = {name: tokens, "attention_mask": torch.ones(1, 3, dtype=torch.long)}
                    generation_config = GenerationConfig(cache_implementation="dynamic", do_sample=do_sample)
                    with mock.patch.object(exporter_utils, "decompose_for_generation", return_value={}) as capture:
                        self.exporter_module.ExecutorchExporter().export_for_generation(
                            self.model,
                            inputs,
                            config=config,
                            generation_config=generation_config,
                            multi_token_decode=True,
                        )
                        capture.assert_called_once_with(
                            self.model, inputs, generation_config=generation_config, multi_token_decode=True
                        )

    def _call_attention(self, attention=None, **kwargs):
        query = torch.zeros(1, 2, 1, 8)
        key = torch.zeros(1, 1, 1, 8)
        attention = self.model.model.layers[0].self_attn if attention is None else attention
        return self.exporter_module._executorch_off_graph_cache_attention_forward(
            attention,
            query,
            key,
            torch.ones_like(key),
            attention_mask=None,
            position_ids=torch.zeros(1, 1, dtype=torch.long),
            scaling=attention.scaling,
            **kwargs,
        )

    def test_rejects_unsupported_attention_arguments_before_cache_update(self):
        cases = [
            ({"dropout": 0.1}, "dropout"),
            ({"softcap": 1.0}, "softcap"),
            ({"head_mask": torch.ones(2)}, "head masks"),
            ({"position_bias": torch.zeros(1, 2, 1, 1)}, "position_bias"),
            ({"is_causal": False}, "causal"),
            ({"use_cache": True}, "HF cache updates"),
            ({"output_attentions": True}, "attention-weight outputs"),
            ({"sliding_window": 4}, "cache layer policy"),
            ({"unknown_attention_option": None}, "unknown_attention_option"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with mock.patch.object(torch.ops.kvcache, "update_and_attend", create=True) as update:
                    with self.assertRaisesRegex(ValueError, message):
                        self._call_attention(**kwargs)
                    update.assert_not_called()

    def test_accepts_supported_attention_arguments(self):
        expected = torch.arange(16, dtype=torch.float32).reshape(1, 2, 1, 8)
        with mock.patch.object(torch.ops.kvcache, "update_and_attend", return_value=expected, create=True) as update:
            output, weights = self._call_attention(
                is_causal=True,
                sliding_window=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                output_router_logits=False,
            )
            update.assert_called_once()
            torch.testing.assert_close(output, expected.transpose(1, 2))
            self.assertIsNone(weights)

    def test_checks_sliding_window_against_layer_policy(self):
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        attention = Qwen3Attention(
            Qwen3Config(
                hidden_size=16,
                head_dim=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                layer_types=["sliding_attention"],
                use_sliding_window=True,
                sliding_window=4,
            ),
            layer_idx=0,
        ).eval()
        for window in (None, 3, 4):
            with self.subTest(window=window):
                with mock.patch.object(
                    torch.ops.kvcache, "update_and_attend", return_value=torch.zeros(1, 2, 1, 8), create=True
                ) as update:
                    if window == 4:
                        self._call_attention(attention, sliding_window=window)
                        update.assert_called_once()
                    else:
                        with self.assertRaisesRegex(ValueError, "cache layer policy"):
                            self._call_attention(attention, sliding_window=window)
                        update.assert_not_called()


class ExecutorchOffGraphCacheConfigTest(unittest.TestCase):
    @parameterized.expand(sorted(_OFF_GRAPH_CACHE_BACKENDS))
    def test_selection_roundtrip(self, backend):
        from transformers.exporters import ExecutorchConfig

        config = ExecutorchConfig(backend=backend, cache_implementation="executorch_off_graph_cache")
        self.assertEqual(ExecutorchConfig.from_dict(config.to_dict()), config)
        self.assertIsNone(ExecutorchConfig().cache_implementation)
        with self.assertRaises(ValueError):
            ExecutorchConfig(cache_implementation="dynamic")


@require_torch
@require_executorch
class ExecutorchOffGraphCacheSafetyTest(unittest.TestCase):
    def setUp(self):
        try:
            from executorch.extension.llm.cache.update_and_attend import update_and_attend  # noqa: F401
            from executorch.extension.llm.export.model_metadata import write_cache_geometry  # noqa: F401
        except ImportError:
            self.skipTest("Requires ExecuTorch's off-graph cache extension")
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig

        self.model = Gemma4ForCausalLM(
            Gemma4TextConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                vocab_size=64,
                num_kv_shared_layers=2,
                layer_types=["sliding_attention", "full_attention"] * 2,
                sliding_window=3,
                per_layer_config={},
                vocab_size_per_layer_input=64,
                hidden_size_per_layer_input=8,
            )
        ).eval()
        self.inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    @parameterized.expand(sorted(_OFF_GRAPH_CACHE_BACKENDS))
    def test_padding_and_custom_masks_rejected(self, backend):
        from transformers.exporters import ExecutorchExporter

        config = ExecutorchConfig(backend=backend, cache_implementation="executorch_off_graph_cache")
        for mask in (torch.tensor([[0, 1, 1, 1]]), torch.ones(1, 1, 4, 4), {"full_attention": None}):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "padding or custom attention masks"):
                ExecutorchExporter().export(self.model, {**self.inputs, "attention_mask": mask}, config)
        # Padding inferred from the original request must also be rejected before capture.
        with self.assertRaisesRegex(ValueError, "padding or custom attention masks"):
            ExecutorchExporter().export_for_generation(
                self.model, self.inputs, config, GenerationConfig(pad_token_id=1, eos_token_id=2)
            )

    def test_export_context_restores_model_and_attention_registry_on_error(self):
        from transformers.exporters.exporter_executorch import _executorch_off_graph_cache_export
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        original_attention = self.model.config._attn_implementation
        original_mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with _executorch_off_graph_cache_export(self.model, self.inputs):
                self.assertEqual(self.model.config._attn_implementation, "executorch_off_graph_cache")
                raise RuntimeError("test failure")
        self.assertEqual(self.model.config._attn_implementation, original_attention)
        self.assertIs(ALL_ATTENTION_FUNCTIONS._global_mapping, original_mapping)

    def test_shared_layers_reuse_producer_cache_ids_and_kv(self):
        from transformers.exporters.exporter_executorch import _executorch_off_graph_cache_export

        # Exercise the real attention adapter without allocating runtime cache state.
        with (
            _executorch_off_graph_cache_export(self.model, self.inputs) as inputs,
            mock.patch("torch.ops.kvcache.update_and_attend", side_effect=lambda query, *args: query) as attend,
            torch.no_grad(),
        ):
            self.model(**inputs)
        calls = attend.call_args_list
        self.assertEqual([call.args[4] for call in calls], [0, 1, 0, 1])
        for consumer, producer in ((2, 0), (3, 1)):
            self.assertIs(calls[consumer].args[1], calls[producer].args[1])
            self.assertIs(calls[consumer].args[2], calls[producer].args[2])
        self.assertEqual([layer.self_attn.layer_idx for layer in self.model.model.layers], [0, 1, 2, 3])

    @parameterized.expand(sorted(_BACKEND_PREPARE.keys() - _OFF_GRAPH_CACHE_BACKENDS), skip_on_empty=True)
    def test_unsupported_backends_reject_before_modifying_model(self, backend):
        from transformers.exporters import ExecutorchConfig, ExecutorchExporter

        with mock.patch.object(self.model, "to") as move_model:
            with self.assertRaisesRegex(ValueError, f"not supported by the ExecuTorch {backend.upper()} backend"):
                ExecutorchExporter().export(
                    self.model,
                    self.inputs,
                    ExecutorchConfig(backend=backend, cache_implementation="executorch_off_graph_cache"),
                )
            move_model.assert_not_called()
            self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))
