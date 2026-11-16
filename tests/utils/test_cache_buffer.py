# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Tests for `GenerationConfig(use_cache_buffer=True)`: `generate` keeps its static cache as the model's
non-persistent buffers, and `decompose_for_generation` / `export_for_generation` capture components without
`past_key_values`, so an exported decode step holds the cache as program state (`BUFFER_MUTATION`s)."""

import copy
import tempfile
import unittest
from collections import Counter

import pytest
from parameterized import parameterized

from transformers import GenerationConfig, is_torch_available
from transformers.testing_utils import require_executorch, require_torch


if is_torch_available():
    import torch

    from transformers import (
        FalconConfig,
        FalconForCausalLM,
        Gemma4ForCausalLM,
        Gemma4TextConfig,
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        StaticCache,
    )
    from transformers.exporters import DynamoConfig, DynamoExporter
    from transformers.exporters.utils import decompose_for_generation


MAX_CACHE_LEN = 16
PROMPT = [1, 5, 9, 2]
NUM_LAYERS = 2


def _tiny_llama_config():
    return LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
    )


def _static_generation_config(use_cache_buffer):
    return GenerationConfig(
        cache_implementation="static",
        max_cache_len=MAX_CACHE_LEN,
        use_cache_buffer=use_cache_buffer,
        do_sample=False,
        pad_token_id=0,
    )


def _prompt_inputs():
    input_ids = torch.tensor([PROMPT])
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _decode_step_inputs(token, position):
    """Single-token decode inputs as `generate` passes them with a static cache."""
    return {
        "input_ids": torch.tensor([[token]]),
        "attention_mask": (torch.arange(MAX_CACHE_LEN) <= position)[None, None, None],
        "position_ids": torch.tensor([[position]]),
    }


def _greedy_decode(step):
    """Feed `PROMPT` one token per `step(token, position) -> logits` call, then continue greedily to `MAX_CACHE_LEN`."""
    tokens = list(PROMPT)
    next_token = tokens[0]
    for position in range(MAX_CACHE_LEN):
        next_token = step(next_token, position)[0, -1].argmax().item()
        if position + 1 < len(PROMPT):
            next_token = PROMPT[position + 1]
        elif position + 1 < MAX_CACHE_LEN:
            tokens.append(next_token)
    return tokens


def _cache_buffers(model):
    return {name: buffer for name, buffer in model.model.named_buffers() if name.startswith("static_cache_")}


@require_torch
class CacheBufferTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.reference = LlamaForCausalLM(_tiny_llama_config()).eval()
        with torch.no_grad():
            self.reference_tokens = self.reference.generate(
                **_prompt_inputs(),
                generation_config=_static_generation_config(False),
                max_new_tokens=MAX_CACHE_LEN - len(PROMPT),
            )[0].tolist()

    def _model(self):
        model = LlamaForCausalLM(copy.deepcopy(self.reference.config)).eval()
        model.load_state_dict(self.reference.state_dict())
        return model

    def test_generate_keeps_cache_as_non_persistent_buffers(self):
        model = self._model()
        with torch.no_grad():
            model.generate(**_prompt_inputs(), generation_config=_static_generation_config(True), max_new_tokens=2)
        buffers = _cache_buffers(model)
        self.assertEqual(len(buffers), 3 * NUM_LAYERS)
        config = model.config
        head_dim = config.hidden_size // config.num_attention_heads
        self.assertEqual(
            tuple(buffers["static_cache_keys_0"].shape), (1, config.num_key_value_heads, MAX_CACHE_LEN, head_dim)
        )
        self.assertFalse(any(key.startswith("model.static_cache_") for key in model.state_dict()))

    @parameterized.expand(
        [
            ("dynamic_cache", {"cache_implementation": "dynamic"}),
            ("flag_off", {"use_cache_buffer": False}),
        ]
    )
    def test_no_buffers_without_static_cache_and_flag(self, _, overrides):
        generation_config = _static_generation_config(True)
        generation_config.update(**overrides)
        model = self._model()
        with torch.no_grad():
            model.generate(**_prompt_inputs(), generation_config=generation_config, max_new_tokens=2)
        self.assertEqual(_cache_buffers(model), {})

    @parameterized.expand(
        [
            ("greedy", {}),
            ("batched_left_padding", {"batched": True}),
            ("beam_search", {"num_beams": 2}),
            ("beams_with_return_sequences", {"num_beams": 2, "num_return_sequences": 2}),
            ("sampling", {"do_sample": True}),
            ("longer_than_max_cache_len", {"max_new_tokens": 2 * MAX_CACHE_LEN}),
        ]
    )
    def test_generate_matches_static_cache(self, _, overrides):
        overrides = dict(overrides)
        if overrides.pop("batched", False):
            inputs = {
                "input_ids": torch.tensor([[0, 0, 7, 3], PROMPT]),
                "attention_mask": torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
            }
        else:
            inputs = _prompt_inputs()
        kwargs = {**inputs, "max_new_tokens": 8, **overrides}

        def generate(model, use_cache_buffer):
            torch.manual_seed(0)
            with torch.no_grad():
                return model.generate(**kwargs, generation_config=_static_generation_config(use_cache_buffer))

        self.assertEqual(generate(self._model(), True).tolist(), generate(self.reference, False).tolist())

    def test_forward_with_or_without_past_key_values(self):
        """After `generate` kept its cache as buffers, a passed cache is used and returned as usual, while a forward
        without one uses the buffers and does not return them."""
        model = self._model()
        with torch.no_grad():
            model.generate(**_prompt_inputs(), generation_config=_static_generation_config(True), max_new_tokens=2)
            cache = StaticCache(config=model.config, max_cache_len=MAX_CACHE_LEN)
            passed = model(input_ids=torch.tensor([PROMPT]), past_key_values=cache)
            expected = self.reference(
                input_ids=torch.tensor([PROMPT]),
                past_key_values=StaticCache(config=model.config, max_cache_len=MAX_CACHE_LEN),
            )
            owned = model(**_decode_step_inputs(PROMPT[0], 0))
        self.assertIs(passed.past_key_values, cache)
        torch.testing.assert_close(passed.logits, expected.logits)
        self.assertEqual(list(owned.keys()), ["logits"])

    @parameterized.expand([("single_token", False), ("multi_token", True)])
    def test_decompose_for_generation_drops_cache_inputs(self, _, multi_token_decode):
        model = self._model()
        components = decompose_for_generation(
            model,
            _prompt_inputs(),
            generation_config=_static_generation_config(True),
            multi_token_decode=multi_token_decode,
        )
        for name, (_, inputs) in components.items():
            self.assertNotIn("past_key_values", inputs, name)
        self.assertEqual(model.model.static_cache_cumulative_length_0.item(), 0)

    @parameterized.expand([("non_strict", False), ("strict", True)])
    @pytest.mark.torch_export_test
    def test_export_for_generation_holds_cache_as_program_state(self, _, strict):
        model = self._model()
        with torch.no_grad():
            decode = DynamoExporter().export_for_generation(
                model, _prompt_inputs(), DynamoConfig(strict=strict), generation_config=_static_generation_config(True)
            )["decode"]
        signature = decode.run_decompositions({}).graph_signature
        self.assertEqual(
            [spec.arg.name for spec in signature.input_specs if spec.kind.name == "USER_INPUT"],
            ["input_ids", "attention_mask", "position_ids"],
        )
        self.assertFalse(any(spec.kind.name == "CONSTANT_TENSOR" for spec in signature.input_specs))
        self.assertEqual(
            dict(Counter(spec.kind.name for spec in signature.output_specs)),
            {"BUFFER_MUTATION": 3 * NUM_LAYERS, "USER_OUTPUT": 1},
        )

        module = decode.module()
        with torch.no_grad():
            tokens = _greedy_decode(lambda token, position: module(**_decode_step_inputs(token, position)).logits)
        self.assertEqual(tokens, self.reference_tokens)

    @require_executorch
    @pytest.mark.executorch_export_test
    def test_executorch_lowering_and_runtime(self):
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.exir import to_edge_transform_and_lower
        from executorch.runtime import Runtime

        model = self._model()
        with torch.no_grad():
            decode = DynamoExporter().export_for_generation(
                model, _prompt_inputs(), DynamoConfig(), generation_config=_static_generation_config(True)
            )["decode"]
        program = to_edge_transform_and_lower(decode, partitioner=[XnnpackPartitioner()]).to_executorch()
        method = Runtime.get().load_program(program.buffer).load_method("forward")
        self.assertEqual((method.metadata.num_inputs(), method.metadata.num_outputs()), (3, 1))
        tokens = _greedy_decode(
            lambda token, position: method.execute(list(_decode_step_inputs(token, position).values()))[0]
        )
        self.assertEqual(tokens, self.reference_tokens)

    def test_shared_kv_layers(self):
        """Layers reusing another layer's KV (`DummyLayer`) hold no state, so they get no buffers; generation and the
        exported decode step still match a regular static cache."""
        config = Gemma4TextConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            layer_types=["full_attention"] * 4,
            num_kv_shared_layers=2,
        )

        def build():
            torch.manual_seed(0)
            return Gemma4ForCausalLM(config).eval()

        with torch.no_grad():
            expected = build().generate(
                **_prompt_inputs(), generation_config=_static_generation_config(False), max_new_tokens=8
            )
            model = build()
            generated = model.generate(
                **_prompt_inputs(), generation_config=_static_generation_config(True), max_new_tokens=8
            )
            decode = DynamoExporter().export_for_generation(
                build(), _prompt_inputs(), DynamoConfig(), generation_config=_static_generation_config(True)
            )["decode"]
        self.assertEqual(generated.tolist(), expected.tolist())
        self.assertEqual(len(_cache_buffers(model)), 3 * 2)  # only the 2 layers that own their KV
        output_kinds = Counter(spec.kind.name for spec in decode.run_decompositions({}).graph_signature.output_specs)
        self.assertEqual(dict(output_kinds), {"BUFFER_MUTATION": 3 * 2, "USER_OUTPUT": 1})

    @parameterized.expand([("sliding_window",), ("no_merge_with_config_defaults",)])
    def test_unsupported_models_raise(self, case):
        if case == "sliding_window":
            model = MistralForCausalLM(MistralConfig(**{**_tiny_llama_config().to_dict(), "sliding_window": 8}))
        else:
            config = FalconConfig(hidden_size=32, num_hidden_layers=NUM_LAYERS, num_attention_heads=4, vocab_size=64)
            model = FalconForCausalLM(config)
        with self.assertRaisesRegex(ValueError, "`use_cache_buffer=True` only supports static caches"):
            model.eval().generate(
                **_prompt_inputs(), generation_config=_static_generation_config(True), max_new_tokens=2
            )

    def test_generation_config_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _static_generation_config(True).save_pretrained(tmp_dir)
            self.assertTrue(GenerationConfig.from_pretrained(tmp_dir).use_cache_buffer)
        self.assertFalse(GenerationConfig().use_cache_buffer)

    def test_bind_buffers_follows_module_device_and_dtype(self):
        model = self._model()
        with torch.no_grad():
            model.generate(**_prompt_inputs(), generation_config=_static_generation_config(True), max_new_tokens=2)
        model.to(device="meta", dtype=torch.float16)
        cache = model.model._cache_buffer
        cache.bind_buffers(model.model)
        self.assertEqual((cache.layers[0].device.type, cache.layers[0].dtype), ("meta", torch.float16))
