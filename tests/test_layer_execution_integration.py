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

"""End-to-end checks for cache backends, native state, structured streams, and execution dependencies."""

import copy
import tempfile
import unittest
from types import SimpleNamespace

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torch_gpu


if is_torch_available():
    import torch

    from transformers import (
        DecoderLayerExecutionAdapter,
        Gemma3nForCausalLM,
        Gemma3nTextConfig,
        LayerExecutionCache,
        LayerExecutionPlan,
        LayerExecutionState,
        RecurrentGemmaConfig,
        RecurrentGemmaForCausalLM,
        register_layer_execution_adapter,
    )

    from . import test_layer_execution as execution_tests

    class _TwoStreamAdapter(DecoderLayerExecutionAdapter):
        def prepare(self, decoder, inputs_embeds, attention_mask, position_ids, cache, **kwargs):
            from transformers.layer_execution.adapters import _LlamaAdapter

            hidden, context = _LlamaAdapter().prepare(
                decoder, inputs_embeds, attention_mask, position_ids, cache, **kwargs
            )
            return LayerExecutionState(hidden, {"auxiliary": torch.zeros_like(hidden)}), context

        def execute_step(self, layer, state, kwargs, cache, use_cache):
            value = self.execute_layer(layer, state.hidden_states, kwargs, cache, use_cache)
            auxiliary = state.streams["auxiliary"]
            return LayerExecutionState(value + auxiliary * 0.125, {"auxiliary": auxiliary + value * 0.25})

        def extract_state(self, output, previous):
            return output

        def finalize(self, decoder, state, cache, context):
            return super().finalize(decoder, state.hidden_states + state.streams["auxiliary"] * 0.5, cache, context)


def make_stateful_model(family):
    torch.manual_seed(17)
    common = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": None,
    }
    if family == "recurrent_gemma":
        return RecurrentGemmaForCausalLM(
            RecurrentGemmaConfig(
                **common,
                intermediate_size=32,
                num_hidden_layers=3,
                lru_width=16,
                conv1d_width=3,
                attention_window_size=16,
            )
        ).eval()
    return Gemma3nForCausalLM(
        Gemma3nTextConfig(
            **common,
            vocab_size_per_layer_input=32,
            hidden_size_per_layer_input=8,
            intermediate_size=[32] * 4,
            num_hidden_layers=4,
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["full_attention", "sliding_attention"] * 2,
            num_kv_shared_layers=2,
            altup_num_inputs=2,
            laurel_rank=4,
            sliding_window=8,
        )
    ).eval()


class _IndependentStepsAdapter:
    """Use separately registered parameter copies as the shared-gradient reference."""

    def __init__(self, native, layers):
        self.native, self.layers = native, layers

    def __getattr__(self, name):
        return getattr(self.native, name)

    def step_kwargs(self, decoder, step, context, shared_states):
        return {
            **self.native.step_kwargs(decoder, step, context, shared_states),
            "independent_step": step.execution_index,
        }

    def execute_step(self, layer, state, kwargs, cache, use_cache):
        return self.native.execute_step(self.layers[kwargs.pop("independent_step")], state, kwargs, cache, use_cache)


@require_torch
class LayerExecutionIntegrationTest(unittest.TestCase):
    @parameterized.expand(
        [(family, backend) for family in ("llama", "qwen3_5") for backend in ("static", "offloaded_static")]
    )
    def test_sdpa_static_capacity_remains_masked_without_compilation(self, family, backend):
        model = execution_tests.LayerExecutionModelTest().make_model(family)
        model.set_attn_implementation("sdpa")
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        cache = LayerExecutionCache(model.config, cache_implementation=backend, max_cache_len=16)
        self.assertTrue(cache.requires_explicit_mask)
        self.assertEqual(cache.is_compileable, backend == "static")
        prefix = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            model(prefix, past_key_values=cache)
            from transformers.layer_execution.cache import _ExecutionCacheView
            from transformers.masking_utils import create_causal_mask

            view = _ExecutionCacheView(cache, source_index=1, execution_index=1)
            # Custom blocks can build their own masks from the execution-local view.
            mask = create_causal_mask(
                model.config,
                model.get_input_embeddings()(torch.tensor([[4]])),
                None,
                view,
                position_ids=torch.tensor([[3]]),
                layer_idx=1,
            )
            self.assertIsNotNone(mask)
            self.assertFalse(mask[..., 4:].any())
            for token in (4, 5):
                query = torch.tensor([[token]])
                actual = model(query, past_key_values=cache).logits
                prefix = torch.cat((prefix, query), dim=1)
                torch.testing.assert_close(actual, model(prefix, use_cache=False).logits[:, -1:], atol=3e-5, rtol=3e-4)

    def test_paged_grouping_preserves_native_metadata_and_skips_consumers(self):
        from transformers.generation.continuous_batching.cache import group_layers_by_attn_type
        from transformers.layer_execution.paged import execution_cache_config

        config = SimpleNamespace(num_hidden_layers=8, layer_types=["full_attention", "sliding_attention"])
        self.assertEqual(group_layers_by_attn_type(config), {"full_attention": [0], "sliding_attention": [1]})
        model = make_stateful_model("gemma3n")
        model.set_layer_execution_plan(LayerExecutionPlan(tuple(range(4)) * 2, kv_sharing="native"))
        metadata = execution_cache_config(model.config)
        groups = group_layers_by_attn_type(metadata)
        self.assertEqual(sorted(index for group in groups.values() for index in group), [0, 1, 4, 5])
        self.assertEqual(model.config.num_hidden_layers, 4)

    def test_native_paged_pipeline_requires_portable_scheduler(self):
        model = execution_tests.LayerExecutionModelTest().make_model("llama")
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        model._pp_stage = SimpleNamespace()
        config = copy.deepcopy(model.generation_config)
        config.cache_implementation = "paged"
        with self.assertRaisesRegex(ValueError, "portable continuous batching"):
            model.init_continuous_batching(generation_config=config)

    @parameterized.expand(
        [
            (family, backend)
            for family in ("llama", "qwen3_5", "recurrent_gemma")
            for backend in ("offloaded", "offloaded_static")
        ]
    )
    @require_torch_gpu
    def test_cuda_offload_attention_and_external_state(self, family, backend):
        model = (
            make_stateful_model(family)
            if family == "recurrent_gemma"
            else execution_tests.LayerExecutionModelTest().make_model(family)
        ).cuda()
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        cache = LayerExecutionCache(model.config, cache_implementation=backend, max_cache_len=16)
        prefix = torch.tensor([[1, 2, 3]], device="cuda")
        with torch.no_grad():
            model(prefix, past_key_values=cache)
            for token in (4, 5):
                query = torch.tensor([[token]], device="cuda")
                actual = model(query, past_key_values=cache).logits
                prefix = torch.cat((prefix, query), dim=1)
                torch.testing.assert_close(actual, model(prefix, use_cache=False).logits[:, -1:], atol=3e-5, rtol=3e-4)
            # Transfers are real: attention, recurrent, and adapter-owned tensors reside on the CPU after each forward.
            from torch.utils._pytree import tree_flatten

            for layer in cache.layers:
                for name in ("keys", "values", "conv_states", "recurrent_states"):
                    leaves, _ = tree_flatten(getattr(layer, name, None))
                    for value in leaves:
                        if isinstance(value, torch.Tensor):
                            self.assertEqual(value.device.type, "cpu")
            for value in tree_flatten(cache.layer_states)[0]:
                self.assertEqual(value.device.type, "cpu")
            cache.reset()
            torch.testing.assert_close(
                model(prefix, past_key_values=cache).logits,
                model(prefix, use_cache=False).logits,
                atol=3e-5,
                rtol=3e-4,
            )

    def test_named_stream_state_matches_expanded_forward_and_gradients(self):
        from transformers.layer_execution.adapters import _ADAPTERS

        from .test_layer_execution_distributed import _source_name

        model = execution_tests.LayerExecutionModelTest().make_model("llama")
        order = (0, 1, 0, 1, 2)
        reference = execution_tests.LayerExecutionModelTest().expanded_reference(model, order).train()
        model.config.model_type = "test_two_streams"
        register_layer_execution_adapter(model.config.model_type, _TwoStreamAdapter)
        self.addCleanup(_ADAPTERS.pop, model.config.model_type)
        model.set_layer_execution_plan(order)
        model.train()
        model.gradient_checkpointing_enable()
        inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        actual = model(inputs, use_cache=False).logits
        hidden = reference.model.embed_tokens(inputs)
        positions = torch.arange(inputs.shape[1])[None]
        rotary = reference.model.rotary_emb(hidden, positions)
        from transformers.masking_utils import create_causal_mask

        mask = create_causal_mask(reference.config, hidden, None, None, position_ids=positions)
        auxiliary = torch.zeros_like(hidden)
        for layer in reference.model.layers:
            value = layer(
                hidden, attention_mask=mask, position_embeddings=rotary, position_ids=positions, use_cache=False
            )
            hidden, auxiliary = value + auxiliary * 0.125, auxiliary + value * 0.25
        expected = reference.lm_head(reference.model.norm(hidden + auxiliary * 0.5))
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        expected.square().mean().backward()
        gradients = {}
        for name, parameter in reference.named_parameters():
            if parameter.grad is not None:
                name = _source_name(name, order)
                gradients[name] = gradients.get(name, 0) + parameter.grad
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-5, rtol=3e-4, msg=name)

    @parameterized.expand(
        [
            (family, checkpointing, device)
            for family in ("gemma3n", "recurrent_gemma")
            for checkpointing in (False, True)
            for device in ("cpu", "cuda")
        ]
    )
    def test_stateful_shared_gradients_against_independent_parameters(self, family, checkpointing, device):
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("Requires CUDA")
        model = make_stateful_model(family).to(device)
        count = model.config.num_hidden_layers
        plan = LayerExecutionPlan(
            tuple(range(count)) * 2, kv_sharing="native" if family == "gemma3n" else "independent"
        )
        reference = copy.deepcopy(model)
        model.set_layer_execution_plan(plan)
        reference.set_layer_execution_plan(plan)
        decoder = reference.get_decoder()
        decoder._independent_steps = torch.nn.ModuleList(
            [copy.deepcopy(decoder.layers[index]) for index in plan.layer_order]
        )
        decoder._layer_execution_adapter = _IndependentStepsAdapter(
            decoder._layer_execution_adapter, decoder._independent_steps
        )
        model.train()
        reference.train()
        if checkpointing:
            model.gradient_checkpointing_enable()
            reference.gradient_checkpointing_enable()
        inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device)
        actual = model(inputs, labels=inputs, use_cache=False)
        expected = reference(inputs, labels=inputs, use_cache=False)
        torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-4)
        actual.loss.backward()
        expected.loss.backward()
        gradients = {}
        for name, parameter in reference.named_parameters():
            if parameter.grad is None:
                continue
            if "._independent_steps." in name:
                prefix, suffix = name.split("._independent_steps.")
                slot, suffix = suffix.split(".", 1)
                name = f"{prefix}.layers.{plan.layer_order[int(slot)]}.{suffix}"
            gradients[name] = gradients.get(name, 0) + parameter.grad
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-5, rtol=3e-4, msg=name)
            else:
                self.assertIsNone(parameter.grad, name)

    def test_continuous_admission_cancel_streaming_stop_and_reconfigure(self):
        from transformers import ContinuousBatchingConfig

        model = execution_tests.LayerExecutionModelTest().make_model("qwen3_5")
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        manager = model.init_continuous_batching(
            continuous_batching_config=ContinuousBatchingConfig(max_requests_per_batch=1)
        )
        with manager.pause():
            manager.start()
            manager.add_request([1, 2, 3], request_id="first", max_new_tokens=3, streaming=True)
            manager.add_request([4, 5], request_id="cancelled", max_new_tokens=3)
            manager.cancel_request("cancelled")
            manager.add_request([6, 7], request_id="bad", max_new_tokens=2, unknown_option=True)
            with self.assertRaisesRegex(ValueError, "Stop continuous"):
                model.set_layer_execution_plan((0, 1, 2))
            with self.assertRaisesRegex(RuntimeError, "pause"):
                manager.stop()
        outputs = []
        finished = set()
        while len(finished) < 3:
            result = manager.get_result(timeout=5)
            self.assertIsNotNone(result)
            outputs.append(result)
            if result.is_finished():
                finished.add(result.request_id)
        self.assertIsNotNone(next(output for output in outputs if output.request_id == "cancelled").error)
        self.assertIsNotNone(next(output for output in outputs if output.request_id == "bad").error)
        first = [output for output in outputs if output.request_id == "first"]
        self.assertEqual([len(output.generated_tokens) for output in first], [1, 2, 3])
        # Normal stop drains queued work, whereas hard stop reports cancellation for pending work.
        manager.add_request([8, 9], request_id="late", max_new_tokens=2)
        manager.stop(timeout=5, keep_for_next_session=True)
        self.assertIsNone(manager.get_result(timeout=1).error)
        self.assertFalse(manager._caches)
        manager.start()
        with manager.pause():
            manager.add_request([1, 2], request_id="hard", max_new_tokens=10)
        manager.stop(hard_stop=True, timeout=5)
        self.assertIsNotNone(manager.get_result(timeout=1).error)
        model.set_layer_execution_plan((0, 1, 2))
        with self.assertRaisesRegex(RuntimeError, "destroyed"):
            manager.start()

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_looped_assistant_and_bounded_rollback_history(self, family):
        model = execution_tests.LayerExecutionModelTest().make_model(family)
        assistant = copy.deepcopy(model)
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        assistant.set_layer_execution_plan((0, 0, 1, 2))
        inputs = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            expected = model.generate(inputs, max_new_tokens=12)
            actual = model.generate(inputs, assistant_model=assistant, max_new_tokens=12, return_dict_in_generate=True)
        torch.testing.assert_close(actual.sequences, expected)
        self.assertLessEqual(len(actual.past_key_values._history), 2)

    def test_speculative_output_capture_and_mask_slicing(self):
        model = execution_tests.LayerExecutionModelTest().make_model("llama")
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        model.set_attn_implementation("eager")
        cache = LayerExecutionCache(model.config)
        cache.activate_past_recording()
        inputs = torch.tensor([[1, 2, 3, 4]])
        mask = torch.zeros(1, 1, 4, 4).masked_fill(
            ~torch.tril(torch.ones(4, 4, dtype=torch.bool)), torch.finfo(torch.float32).min
        )
        with torch.no_grad():
            actual = model(
                inputs, past_key_values=cache, attention_mask=mask, output_hidden_states=True, output_attentions=True
            )
            expected = model(
                inputs, attention_mask=mask, use_cache=False, output_hidden_states=True, output_attentions=True
            )
        torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-4)
        self.assertEqual(len(actual.hidden_states), len(expected.hidden_states))
        for actual_value, expected_value in zip(actual.hidden_states, expected.hidden_states):
            torch.testing.assert_close(actual_value, expected_value, atol=2e-5, rtol=2e-4)
        for actual_value, expected_value in zip(actual.attentions, expected.attentions):
            torch.testing.assert_close(actual_value, expected_value, atol=2e-5, rtol=2e-4)

    @parameterized.expand(
        [
            (family, backend)
            for family in ("llama", "qwen3_5")
            for backend in ("dynamic", "static", "offloaded", "offloaded_static")
        ]
    )
    def test_backend_forward_beams_reset_and_checkpoint(self, family, backend):
        helper = execution_tests.LayerExecutionModelTest()
        model = helper.make_model(family)
        order = (0, 1, 0, 1, 2)
        reference = helper.expanded_reference(model, order)
        model.set_layer_execution_plan(order)
        cache = LayerExecutionCache(model.config, cache_implementation=backend, max_cache_len=16)
        inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        with torch.no_grad():
            actual = model(inputs, past_key_values=cache, use_cache=True)
            torch.testing.assert_close(actual.logits, reference(inputs, use_cache=False).logits, atol=2e-5, rtol=2e-4)
            tail = torch.tensor([[7, 8], [8, 9]])
            actual = model(tail, past_key_values=cache, use_cache=True)
            expected = reference(torch.cat((inputs, tail), dim=1), use_cache=False)
            torch.testing.assert_close(actual.logits, expected.logits[:, -2:], atol=2e-5, rtol=2e-4)
            expected_ids = reference.generate(inputs, num_beams=2, max_new_tokens=3)
            actual_ids = model.generate(inputs, num_beams=2, max_new_tokens=3, cache_implementation=backend)
            torch.testing.assert_close(actual_ids, expected_ids)
            cache.reset()
            self.assertEqual(cache.get_seq_length(), 0)
            torch.testing.assert_close(
                model(inputs, past_key_values=cache).logits, reference(inputs).logits, atol=2e-5, rtol=2e-4
            )

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_speculative_rollback_matches_greedy(self, family):
        helper = execution_tests.LayerExecutionModelTest()
        model = helper.make_model(family)
        assistant = helper.make_model(family)
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        inputs = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            greedy = model.generate(inputs, max_new_tokens=6)
            assisted = model.generate(inputs, assistant_model=assistant, max_new_tokens=6)
        torch.testing.assert_close(assisted, greedy)

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_continuous_batching_native_api(self, family):
        model = execution_tests.LayerExecutionModelTest().make_model(family)
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        prompts = [[1, 2, 3], [4, 5, 6], [1, 2]]
        with torch.no_grad():
            expected = [
                model.generate(torch.tensor([prompt]), max_new_tokens=4)[0, len(prompt) :].tolist()
                for prompt in prompts
            ]
        results = model.generate_batch(prompts, max_new_tokens=4, warmup=False)
        self.assertEqual(len(results), len(prompts))
        for index, tokens in enumerate(expected):
            result = results[f"req_{index}"]
            self.assertIsNone(result.error)
            self.assertEqual(result.generated_tokens, tokens)

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_quantized_cache_isolated_buffers_beams_and_reset(self, family):
        model = execution_tests.LayerExecutionModelTest().make_model(family)
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        options = {"backend": "torch", "nbits": 4, "q_group_size": 8, "residual_length": 2}
        cache = LayerExecutionCache(model.config, cache_implementation="quantized", cache_config=options)
        cache.batch_repeat_interleave(2)
        for layer in cache.layers:
            if hasattr(layer, "_quantize"):
                layer.batch_repeat_interleave(2)
        inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        with torch.no_grad():
            prefill = model(inputs, past_key_values=cache).logits
            torch.testing.assert_close(prefill, model(inputs, use_cache=False).logits)
            actual = model(torch.tensor([[7], [8]]), past_key_values=cache).logits
            expected = model(torch.tensor([[1, 2, 3, 7], [4, 5, 6, 8]]), use_cache=False).logits[:, -1:]
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.2)
            generated = model.generate(
                inputs, num_beams=2, max_new_tokens=3, cache_implementation="quantized", cache_config=options
            )
            self.assertEqual(generated.shape, (2, 6))
            cache.reset()
            torch.testing.assert_close(model(inputs, past_key_values=cache).logits, prefill)

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_static_cache_fullgraph_decode(self, family):
        model = execution_tests.LayerExecutionModelTest().make_model(family)
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        cache = LayerExecutionCache(model.config, cache_implementation="static", max_cache_len=16)
        prefix = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            model(prefix, past_key_values=cache)
            compiled = torch.compile(model, backend="eager", fullgraph=True)
            for token in (4, 5):
                actual = compiled(torch.tensor([[token]]), past_key_values=cache).logits
                prefix = torch.cat((prefix, torch.tensor([[token]])), dim=1)
                expected = model(prefix, use_cache=False).logits[:, -1:]
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        torch._dynamo.reset()

    @parameterized.expand([("llama",), ("gemma3n",)])
    def test_native_paged_continuous_batching(self, family):
        from transformers import ContinuousBatchingConfig

        if family == "llama":
            model = execution_tests.LayerExecutionModelTest().make_model("llama")
            model.set_layer_execution_plan((0, 1, 0, 1, 2))
        else:
            model = make_stateful_model("gemma3n")
            model.set_layer_execution_plan(LayerExecutionPlan(tuple(range(4)) * 2, kv_sharing="native"))
        count = model.config.num_hidden_layers
        prompts = [list(range(1, 12)), [4, 5], list(range(1, 12))]
        with torch.no_grad():
            expected = [
                model.generate(torch.tensor([prompt]), max_new_tokens=3)[0, len(prompt) :].tolist()
                for prompt in prompts
            ]
        config = copy.deepcopy(model.generation_config)
        config.cache_implementation = "paged"
        results = model.generate_batch(
            prompts,
            generation_config=config,
            max_new_tokens=3,
            warmup=False,
            continuous_batching_config=ContinuousBatchingConfig(
                num_blocks=32,
                block_size=8,
                max_batch_tokens=16,
                max_requests_per_batch=4,
                use_cuda_graph=False,
                use_async_batching=False,
                auto_switch_to_flash=False,
            ),
        )
        self.assertEqual(len(results), len(prompts))
        for index, tokens in enumerate(expected):
            self.assertEqual(results[f"req_{index}"].generated_tokens, tokens)
        self.assertEqual(model.config.num_hidden_layers, count)

    def test_recurrent_gemma_external_state_and_interleaved_requests(self):
        torch.manual_seed(17)
        model = RecurrentGemmaForCausalLM(
            RecurrentGemmaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=3,
                num_attention_heads=2,
                lru_width=16,
                conv1d_width=3,
                attention_window_size=16,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=None,
            )
        ).eval()
        reference = copy.deepcopy(model)
        inputs = torch.tensor([[1, 2, 3]])
        model.set_layer_execution_plan((0, 1, 2))
        with torch.no_grad():
            torch.testing.assert_close(
                model(inputs, use_cache=False).logits, reference(inputs, use_cache=False).logits
            )
        model.set_layer_execution_plan((0, 1, 0, 1, 2))
        prefixes = [inputs, torch.tensor([[4, 5]])]
        caches = []
        with torch.no_grad():
            for prefix in prefixes:
                caches.append(model(prefix, use_cache=True).past_key_values)
            for index, prefix in enumerate(prefixes):
                tail = torch.tensor([[6, 7]])
                actual = model(tail, past_key_values=caches[index]).logits
                expected = model(torch.cat((prefix, tail), dim=1), use_cache=False).logits[:, -2:]
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
            self.assertIsNot(caches[0].layer_states[0]["recurrent"], caches[0].layer_states[2]["recurrent"])
            for layer in model.model.layers:
                if hasattr(layer.temporal_block, "conv1d_state"):
                    self.assertIsNone(layer.temporal_block.conv1d_state)
                    self.assertIsNone(layer.temporal_block.rg_lru.recurrent_states)
            ids = model.generate(inputs, max_new_tokens=3)
            torch.testing.assert_close(ids, model.generate(inputs, max_new_tokens=3, use_cache=False))

    def test_gemma3n_native_dependencies_streams_save_load(self):
        torch.manual_seed(17)
        model = Gemma3nForCausalLM(
            Gemma3nTextConfig(
                vocab_size=32,
                vocab_size_per_layer_input=32,
                hidden_size=16,
                hidden_size_per_layer_input=8,
                intermediate_size=[32] * 4,
                num_hidden_layers=4,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                layer_types=["full_attention", "sliding_attention"] * 2,
                num_kv_shared_layers=2,
                altup_num_inputs=2,
                laurel_rank=4,
                sliding_window=8,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=None,
            )
        ).eval()
        inputs = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            expected = model(inputs, use_cache=False).logits
        with self.assertRaisesRegex(ValueError, "kv_sharing"):
            model.set_layer_execution_plan((0, 1, 2, 3))
        model.set_layer_execution_plan(LayerExecutionPlan((0, 1, 2, 3), kv_sharing="native"))
        with torch.no_grad():
            torch.testing.assert_close(model(inputs, use_cache=False).logits, expected)
        plan = LayerExecutionPlan(tuple(range(4)) * 2, kv_sharing="native")
        model.set_layer_execution_plan(plan)
        self.assertEqual(
            [step.kv_producer for step in model.model._layer_execution_steps], [None, None, 0, 1, None, None, 4, 5]
        )
        with torch.no_grad():
            cache = model(inputs, use_cache=True).past_key_values
            actual = model(torch.tensor([[4]]), past_key_values=cache).logits
            expected = model(torch.tensor([[1, 2, 3, 4]]), use_cache=False).logits[:, -1:]
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = Gemma3nForCausalLM.from_pretrained(directory).eval()
            self.assertEqual(restored.get_layer_execution_plan(), plan)
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs).logits, model(inputs).logits)
