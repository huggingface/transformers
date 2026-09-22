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

import copy
import importlib.util
import unittest
from unittest.mock import patch

from transformers import GenerationConfig
from transformers.testing_utils import require_executorch, require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=128,
        )
    ).eval()


class NativeCacheConfigTest(unittest.TestCase):
    def test_selection_roundtrip(self):
        from transformers.exporters import ExecutorchConfig

        config = ExecutorchConfig(backend="mlx", cache_implementation="executorch_native")
        self.assertEqual(ExecutorchConfig.from_dict(config.to_dict()), config)
        self.assertIsNone(ExecutorchConfig().cache_implementation)
        self.assertFalse(hasattr(config, "max_cache_len"))
        with self.assertRaises(ValueError):
            ExecutorchConfig(cache_implementation="dynamic")
        with self.assertRaises(ValueError):
            GenerationConfig(cache_implementation="executorch_native")


@require_torch
@require_executorch
class NativeCacheExportTest(unittest.TestCase):
    def setUp(self):
        try:
            from executorch.extension.llm.cache.update_and_attend import REGISTRY
        except ImportError:
            self.skipTest("Requires ExecuTorch's native cache extension")
        self.registry = REGISTRY

    def _export(self, model=None, inputs=None, strict=True, **generation_kwargs):
        from transformers.exporters import DynamoConfig, DynamoExporter, ExecutorchConfig, ExecutorchExporter
        from transformers.integrations.executorch_native import native_cache_export

        model = tiny_model() if model is None else model
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])} if inputs is None else inputs
        captured_inputs = {}

        def export_ep(exporter, submodel, inputs, config):
            self.assertEqual(config.cache_implementation, "executorch_native")
            with native_cache_export(submodel, inputs) as inputs:
                # The real kernel must not run or be needed during tracing.
                with self.assertRaisesRegex(RuntimeError, "no active cache"):
                    self.registry.current()
                name = "prefill" if not captured_inputs else "decode"
                captured_inputs[name] = copy.deepcopy(inputs)
                return DynamoExporter().export(submodel, inputs, DynamoConfig(dynamic=True, strict=strict))

        with patch.object(ExecutorchExporter, "export", export_ep):
            artifacts = ExecutorchExporter().export_for_generation(
                model,
                inputs,
                ExecutorchConfig(backend="mlx", dynamic=True, cache_implementation="executorch_native"),
                generation_config=GenerationConfig(do_sample=False, **generation_kwargs),
                multi_token_decode=True,
            )
        return artifacts, captured_inputs

    def test_export_contains_native_ops_without_cache_io(self):
        model = tiny_model()
        original_attention = model.config._attn_implementation
        artifacts, inputs = self._export(model)
        self.assertEqual(model.config._attn_implementation, original_attention)
        with self.assertRaisesRegex(RuntimeError, "no active cache"):
            self.registry.current()
        self.assertEqual(set(artifacts), {"prefill", "decode"})
        self.assertEqual(inputs["decode"]["position_ids"].tolist(), [[4, 5]])
        for ep in artifacts.values():
            ops = [node for node in ep.graph.nodes if node.target == torch.ops.kvcache.update_and_attend.default]
            self.assertEqual(len(ops), 2)
            self.assertFalse(ep.graph_signature.buffers_to_mutate)
            self.assertFalse(ep.graph_signature.user_inputs_to_mutate)
            self.assertEqual(set(ep.graph_signature.user_inputs), {"input_ids", "position_ids"})
            self.assertEqual(len(ep.graph_signature.user_outputs), 1)

    def test_geometry_matches_cache_slots_and_has_no_capacity(self):
        from executorch.extension.llm.export.model_metadata import write_cache_geometry

        from transformers.integrations.executorch_native import native_cache_geometry

        artifacts, _ = self._export()
        expected = write_cache_geometry([2, 2], [8, 8], [0, 0])
        for ep in artifacts.values():
            metadata = native_cache_geometry(ep)
            self.assertEqual(set(metadata), set(expected))
            for name, value in expected.items():
                torch.testing.assert_close(metadata[name], value)

    def test_geometry_survives_pte_serialization(self):
        from executorch.extension.llm.export.model_metadata import write_cache_geometry
        from executorch.runtime import Runtime

        from transformers.exporters import ExecutorchConfig, ExecutorchExporter

        if importlib.util.find_spec("executorch.backends.mlx") is None:
            self.skipTest("The initially supported ExecuTorch backend is not installed")
        artifacts = ExecutorchExporter().export_for_generation(
            tiny_model(),
            {"input_ids": torch.tensor([[1, 2, 3, 4]])},
            ExecutorchConfig(backend="mlx", dynamic=True, cache_implementation="executorch_native"),
            GenerationConfig(cache_implementation="dynamic"),
            multi_token_decode=True,
        )
        expected = write_cache_geometry([2, 2], [8, 8], [0, 0])
        for artifact in artifacts.values():
            self.assertEqual(artifact.config_methods, set(expected))
            program = Runtime.get().load_program(artifact.buffer)
            for name, value in expected.items():
                actual = program.load_method(name).execute([])[0]
                torch.testing.assert_close(actual, value)

    def test_prompt_decode_chunk_and_reset_parity(self):
        from executorch.extension.llm.cache.reference_cache import CacheConfig, SequenceReferenceCache

        from transformers import DynamicCache

        for strict in (False, True):
            with self.subTest(strict=strict):
                model = tiny_model()
                artifacts, _ = self._export(model, strict=strict)
                prefill, decode = artifacts["prefill"].module(), artifacts["decode"].module()
                reference_cache = SequenceReferenceCache(
                    CacheConfig(n_layers=2, n_kv_heads=2, head_dim=8, capacity=32)
                )
                self.registry.install("transformers-native-test", reference_cache)
                try:
                    with self.registry.active("transformers-native-test"), torch.no_grad():
                        # Replay after reset, also using decode for the empty-cache prompt.
                        for first_method in (prefill, decode):
                            reference_cache.reset()
                            eager_cache = DynamicCache(config=model.config)
                            start = 0
                            for method, ids in ((first_method, [1, 2, 3, 4]), (decode, [7]), (decode, [8, 9, 10])):
                                tokens = torch.tensor([ids])
                                positions = torch.arange(start, start + len(ids)).unsqueeze(0)
                                expected = model(
                                    input_ids=tokens, position_ids=positions, past_key_values=eager_cache
                                ).logits
                                actual = method(input_ids=tokens, position_ids=positions)
                                logits = actual.logits if hasattr(actual, "logits") else actual[0]
                                torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-4)
                                start += len(ids)
                finally:
                    self.registry.uninstall("transformers-native-test")

    def test_unsupported_backend_rejected_before_capture(self):
        from transformers.exporters import ExecutorchConfig, ExecutorchExporter

        model = tiny_model()
        with patch.object(model, "generate") as generate:
            with self.assertRaisesRegex(ValueError, "requires.*mlx"):
                ExecutorchExporter().export_for_generation(
                    model,
                    {"input_ids": torch.tensor([[1, 2, 3]])},
                    ExecutorchConfig(backend="xnnpack", cache_implementation="executorch_native"),
                    GenerationConfig(cache_implementation="dynamic"),
                )
            generate.assert_not_called()

    def test_padding_custom_masks_and_positions_rejected(self):
        for extra in (
            {"attention_mask": torch.tensor([[0, 1, 1, 1]])},
            {"attention_mask": torch.ones(1, 1, 4, 4)},
            {"position_ids": torch.tensor([[0, 1, 0, 1]])},
            {"input_ids": torch.ones(2, 4, dtype=torch.long)},
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "executorch_native"):
                self._export(inputs={"input_ids": torch.tensor([[1, 2, 3, 4]]), **extra})

    def test_unsupported_generation_modes_rejected(self):
        for kwargs in (
            {"num_beams": 2},
            {"prompt_lookup_num_tokens": 2},
            {"use_cache": False},
            {"output_attentions": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, RuntimeError)):
                self._export(**kwargs)

    def test_export_failure_restores_model_and_registry(self):
        from transformers.exporters import DynamoExporter, ExecutorchConfig, ExecutorchExporter
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        model = tiny_model()
        original_attention = model.config._attn_implementation
        original_mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
        outer_cache = object()
        self.registry.install("transformers-native-outer", outer_cache)
        try:
            with self.registry.active("transformers-native-outer"):
                with (
                    patch.object(DynamoExporter, "export", side_effect=ValueError("trace failed")),
                    patch.object(self.registry, "install", wraps=self.registry.install) as install,
                    self.assertRaisesRegex(RuntimeError, "component 'prefill'"),
                ):
                    ExecutorchExporter().export_for_generation(
                        model,
                        {"input_ids": torch.tensor([[1, 2, 3, 4]])},
                        ExecutorchConfig(backend="mlx", cache_implementation="executorch_native"),
                        GenerationConfig(cache_implementation="dynamic"),
                    )
                install.assert_not_called()
                self.assertIs(self.registry.current(), outer_cache)
                self.assertEqual(model.config._attn_implementation, original_attention)
                self.assertIs(ALL_ATTENTION_FUNCTIONS._global_mapping, original_mapping)
        finally:
            self.registry.uninstall("transformers-native-outer")

    def test_missing_extension_reports_actionable_error(self):
        with patch.dict("sys.modules", {"executorch.extension.llm.cache.update_and_attend": None}):
            with self.assertRaisesRegex(ImportError, "executorch_native requires an ExecuTorch build"):
                self._export()

    def test_capture_tokens_match_ordinary_generation(self):
        model = tiny_model()
        prompt = torch.tensor([[1, 2, 3, 4]])
        expected = model.generate(
            prompt,
            generation_config=GenerationConfig(do_sample=False, max_new_tokens=3, min_new_tokens=3),
        )
        _, inputs = self._export(model)
        torch.testing.assert_close(inputs["decode"]["input_ids"], expected[:, 4:6])

    def test_unsupported_layer_policy_rejected(self):
        model = tiny_model()
        model.config.layer_types = ["sliding_attention", "full_attention"]
        model.config.sliding_window = 4
        with self.assertRaisesRegex(ValueError, "full attention"):
            self._export(model)

    def test_capture_cache_selection_and_capacity_do_not_change_native_graph(self):
        from transformers import DynamicCache, StaticCache
        from transformers.exporters.utils import decompose_for_generation

        model = tiny_model()
        original_attention = model.config._attn_implementation
        first, first_inputs = self._export(model, cache_implementation="dynamic")
        for implementation, capacity in (("dynamic", None), ("static", 16), ("static", 32)):
            with self.subTest(implementation=implementation, capacity=capacity):

                def capture(*args, **kwargs):
                    self.assertEqual(model.config._attn_implementation, original_attention)
                    stages = decompose_for_generation(*args, **kwargs)
                    expected_type = StaticCache if implementation == "static" else DynamicCache
                    for _, inputs in stages.values():
                        self.assertIsInstance(inputs["past_key_values"], expected_type)
                    if capacity is not None:
                        self.assertEqual(stages["decode"][1]["past_key_values"].get_max_length(), capacity)
                    return stages

                with (
                    patch("transformers.exporters.base.decompose_for_generation", side_effect=capture),
                    patch.object(self.registry, "install", wraps=self.registry.install) as install,
                ):
                    second, second_inputs = self._export(
                        model, cache_implementation=implementation, max_cache_len=capacity
                    )
                install.assert_not_called()
                for name in first:
                    self.assertEqual(str(first[name].graph), str(second[name].graph))
                    torch.testing.assert_close(first_inputs[name]["input_ids"], second_inputs[name]["input_ids"])
                    torch.testing.assert_close(first_inputs[name]["position_ids"], second_inputs[name]["position_ids"])

    def test_component_config_consistency(self):
        from transformers.exporters import ExecutorchConfig, ExecutorchExporter

        model = tiny_model()
        native = ExecutorchConfig(backend="mlx", cache_implementation="executorch_native")
        with patch.object(model, "generate") as generate:
            with self.assertRaisesRegex(ValueError, "same cache implementation"):
                ExecutorchExporter().export_for_generation(
                    model,
                    {"input_ids": torch.tensor([[1, 2, 3, 4]])},
                    {"prefill": native, "decode": ExecutorchConfig(backend="mlx")},
                )
            generate.assert_not_called()

    def test_captured_custom_mask_rejected(self):
        from transformers import StaticCache
        from transformers.integrations.executorch_native import prepare_native_inputs

        model = tiny_model()
        cache = StaticCache(config=model.config, max_cache_len=8)
        allowed = torch.arange(8) <= torch.arange(4).unsqueeze(-1)
        mask = allowed.unsqueeze(0).unsqueeze(0)
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "past_key_values": cache, "attention_mask": mask}
        prepared = prepare_native_inputs(inputs)
        self.assertNotIn("attention_mask", prepared)
        self.assertIs(inputs["past_key_values"], cache)
        mask[0, 0, 3, 0] = False
        with self.assertRaisesRegex(ValueError, "custom attention masks"):
            prepare_native_inputs(inputs)
