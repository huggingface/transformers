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

"""Extension tests using projected attention and a pure state-space decoder, without adding built-in adapters."""

import copy
import inspect
import pickle
import re
import tempfile
import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    import torch.multiprocessing as mp

    from transformers import (
        DecoderLayerExecutionAdapter,
        DynamicCache,
        LayerExecutionCache,
        MambaConfig,
        MambaForCausalLM,
        OPTConfig,
        OPTForCausalLM,
        register_layer_execution_adapter,
    )
    from transformers.layer_execution.adapters import _ADAPTERS
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.mamba.modeling_mamba import MambaOutput
    from transformers.models.opt.modeling_opt import OPTDecoder

    from . import test_layer_execution as execution_tests

    class _OPTAdapter(DecoderLayerExecutionAdapter):
        # This deliberately cannot use the default attribute lookup.
        layer_container_name = "unused"

        def get_layers(self, decoder):
            path = "block_stack.blocks" if hasattr(decoder, "block_stack") else "layers"
            return decoder.get_submodule(path)

        def validate(self, decoder):
            super().validate(decoder)
            if decoder.layerdrop != 0:
                raise ValueError("This example adapter requires layerdrop=0.")

        def prepare(self, decoder, inputs_embeds, attention_mask, position_ids, cache, **kwargs):
            offset = cache.get_seq_length() if cache is not None else 0
            position_mask = attention_mask
            if position_mask is None:
                position_mask = inputs_embeds.new_ones(inputs_embeds.shape[0], offset + inputs_embeds.shape[1])
            if position_ids is None:
                position_ids = (position_mask.cumsum(dim=1) * position_mask - 1).long()[:, offset:]
            causal_mask = create_causal_mask(decoder.config, inputs_embeds, attention_mask, cache)
            positions = decoder.embed_positions(position_mask, offset, position_ids=position_ids)
            hidden_states = decoder.project_in(inputs_embeds) if decoder.project_in is not None else inputs_embeds
            return hidden_states + positions.to(hidden_states.device), {
                "attention_mask": causal_mask,
                "position_ids": position_ids,
            }

        def finalize(self, decoder, hidden_states, cache, context):
            if decoder.final_layer_norm is not None:
                hidden_states = decoder.final_layer_norm(hidden_states)
            if decoder.project_out is not None:
                hidden_states = decoder.project_out(hidden_states)
            return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=cache)

    class _MambaAdapter(DecoderLayerExecutionAdapter):
        cache_name = "cache_params"

        def prepare(self, decoder, inputs_embeds, attention_mask, position_ids, cache, **kwargs):
            # Native Mamba's full-sequence scan does not accept the previous recurrent state for cached chunks.
            if cache is not None and cache.get_seq_length() > 0 and inputs_embeds.shape[1] != 1:
                raise ValueError("This example adapter requires single-token cached continuation.")
            record = kwargs.get("output_hidden_states", decoder.config.output_hidden_states)
            return inputs_embeds, {"attention_mask": attention_mask, "history": [] if record else None}

        def layer_kwargs(self, decoder, source_index, context):
            return context

        def execute_layer(self, layer, hidden_states, layer_kwargs, cache, use_cache):
            output = layer(hidden_states, cache_params=cache, attention_mask=layer_kwargs["attention_mask"])
            if layer_kwargs["history"] is not None:
                layer_kwargs["history"].append(output)
            return output

        def finalize(self, decoder, hidden_states, cache, context):
            hidden_states = decoder.norm_f(hidden_states)
            history = tuple(context["history"]) + (hidden_states,) if context["history"] is not None else None
            return MambaOutput(last_hidden_state=hidden_states, cache_params=cache, hidden_states=history)

    class _NestedOPTDecoder(OPTDecoder):
        def __init__(self, config):
            super().__init__(config)
            self.block_stack = torch.nn.ModuleDict({"blocks": self.layers})
            del self.layers

        @property
        def layers(self):
            if "block_stack" in self._modules:
                return self.block_stack["blocks"]
            return self._modules["layers"]


def _spawned_adapter_worker(rank, payloads):
    torch.set_num_threads(1)
    for payload, expected in payloads:
        model = pickle.loads(payload).eval()
        if model.config.model_type in _ADAPTERS:
            raise AssertionError("This test requires a fresh process without the custom adapter registration.")
        inputs = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            actual = model.generate(inputs, max_new_tokens=3, do_sample=False)
        torch.testing.assert_close(actual, torch.tensor(expected))


@require_torch
class LayerExecutionAdapterTest(unittest.TestCase):
    def make_model(self, family):
        torch.manual_seed(17)
        common = {
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 3,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": None,
        }
        if family == "opt":
            model = OPTForCausalLM(
                OPTConfig(
                    **common,
                    word_embed_proj_dim=8,
                    ffn_dim=32,
                    num_attention_heads=2,
                    dropout=0.0,
                    attention_dropout=0.0,
                )
            )
            adapter = _OPTAdapter
        else:
            model = MambaForCausalLM(
                MambaConfig(**common, state_size=4, conv_kernel=2, time_step_rank=2, use_associative_scan=False)
            )
            adapter = _MambaAdapter
        register_layer_execution_adapter(model.config.model_type, adapter)
        self.addCleanup(_ADAPTERS.pop, model.config.model_type)
        return model.cpu().eval()

    def reference(self, model, order):
        return execution_tests.LayerExecutionModelTest().expanded_reference(model, order)

    @parameterized.expand([("opt",), ("mamba",)])
    def test_extension_forward_cache_outputs_and_generation(self, family):
        model = self.make_model(family)
        order = (0, 1, 0, 1, 2)
        reference = self.reference(model, order)
        inputs = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        mask = inputs.ne(0).long()
        decoder = model.get_decoder()
        signature = inspect.signature(decoder.forward)
        identities = {name: id(parameter) for name, parameter in model.named_parameters()}
        keys = set(model.state_dict())
        with torch.no_grad():
            original = model(inputs, attention_mask=mask, use_cache=False).logits
        model.set_layer_execution_plan(order)
        self.assertEqual(inspect.signature(decoder.forward), signature)
        cache_name = _ADAPTERS[model.config.model_type].cache_name
        with self.assertRaisesRegex(ValueError, "LayerExecutionCache"):
            model(inputs, **{cache_name: DynamicCache(config=model.config)})
        with torch.no_grad():
            actual = model(inputs, attention_mask=mask, use_cache=True, output_hidden_states=True)
            expected = reference(inputs, attention_mask=mask, use_cache=False, output_hidden_states=True)
            torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-4)
            self.assertEqual(len(actual.hidden_states), len(order) + 1)
            for actual_hidden, expected_hidden in zip(actual.hidden_states, expected.hidden_states):
                torch.testing.assert_close(actual_hidden, expected_hidden, atol=2e-5, rtol=2e-4)
            cache = getattr(actual, cache_name)
            self.assertIsInstance(cache, LayerExecutionCache)
            if family == "mamba":
                self.assertNotEqual(
                    cache.layers[0].conv_states[0].data_ptr(), cache.layers[2].conv_states[0].data_ptr()
                )
                self.assertNotEqual(
                    cache.layers[0].recurrent_states[0].data_ptr(), cache.layers[2].recurrent_states[0].data_ptr()
                )
            else:
                self.assertNotEqual(cache.layers[0].keys.data_ptr(), cache.layers[2].keys.data_ptr())
                self.assertEqual(actual.hidden_states[0].shape[-1], 16)
                self.assertEqual(actual.hidden_states[-1].shape[-1], 8)
            next_tokens = torch.tensor([[5, 6], [6, 7]])
            if family == "mamba":
                with self.assertRaisesRegex(ValueError, "single-token"):
                    model(next_tokens, cache_params=cache)
                next_tokens = next_tokens[:, :1]
            full_mask = torch.cat([mask, torch.ones_like(next_tokens)], dim=1)
            continuation_mask = full_mask if family == "opt" else None
            saved_cache = copy.deepcopy(cache)
            continuation = model(
                next_tokens,
                attention_mask=continuation_mask,
                **{cache_name: cache},
                use_cache=True,
                output_hidden_states=True,
            )
            expected = reference(torch.cat([inputs, next_tokens], dim=1), attention_mask=full_mask, use_cache=False)
            torch.testing.assert_close(
                continuation.logits, expected.logits[:, -next_tokens.shape[1] :], atol=2e-5, rtol=2e-4
            )
            # The models disagree on positional parameter order. Both retain their own native API.
            if family == "opt":
                positional = decoder(next_tokens, full_mask, saved_cache, None, True)
            else:
                positional = decoder(next_tokens, None, saved_cache, True)
            torch.testing.assert_close(
                positional.last_hidden_state, continuation.hidden_states[-1], atol=2e-5, rtol=2e-4
            )
            for beams in (1, 2):
                cached = model.generate(
                    inputs, attention_mask=mask, num_beams=beams, max_new_tokens=3, do_sample=False
                )
                uncached = reference.generate(
                    inputs, attention_mask=mask, num_beams=beams, use_cache=False, max_new_tokens=3, do_sample=False
                )
                torch.testing.assert_close(cached, uncached)
            as_tuple = decoder(input_ids=inputs, attention_mask=mask, use_cache=False, return_dict=False)
            as_dict = decoder(input_ids=inputs, attention_mask=mask, use_cache=False, return_dict=True)
            torch.testing.assert_close(as_tuple[0], as_dict.last_hidden_state)
            model.set_layer_execution_plan(None)
            torch.testing.assert_close(
                model(inputs, attention_mask=mask, use_cache=False).logits, original, atol=0, rtol=0
            )
            with self.assertRaisesRegex(ValueError, "clear the cache"):
                decoder(input_ids=next_tokens, **{cache_name: cache})
        self.assertEqual(keys, set(model.state_dict()))
        self.assertEqual(identities, {name: id(parameter) for name, parameter in model.named_parameters()})

    @parameterized.expand([(family, checkpoint) for family in ("opt", "mamba") for checkpoint in (False, True)])
    def test_extension_shared_gradients_and_checkpointing(self, family, checkpoint):
        model = self.make_model(family)
        order = (0, 1, 0, 1, 2)
        reference = self.reference(model, order)
        model.set_layer_execution_plan(order)
        model.train()
        reference.train()
        if checkpoint:
            model.gradient_checkpointing_enable()
        inputs = torch.tensor([[1, 2, 3, 4]])
        output = model(inputs, labels=inputs, use_cache=True)
        cache_name = _ADAPTERS[model.config.model_type].cache_name
        self.assertIsNone(getattr(output, cache_name))
        expected = reference(inputs, labels=inputs, use_cache=False)
        torch.testing.assert_close(output.loss, expected.loss, atol=1e-6, rtol=1e-5)
        output.loss.backward()
        expected.loss.backward()
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            match = re.search(r"\.layers\.(\d+)\.", name)
            if match:
                source_index = int(match[1])
                gradients = [
                    reference_parameters[name.replace(f"layers.{source_index}.", f"layers.{index}.")].grad
                    for index, source in enumerate(order)
                    if source == source_index
                ]
                gradient = sum(gradients)
            else:
                gradient = reference_parameters[name].grad
            torch.testing.assert_close(parameter.grad, gradient, atol=3e-5, rtol=3e-4)
        with self.assertRaisesRegex(ValueError, "Training"):
            model(inputs, **{cache_name: LayerExecutionCache(model.config)})

    @parameterized.expand([("opt",), ("mamba",)])
    def test_extension_checkpoint_and_pickle(self, family):
        model = self.make_model(family)
        model.set_layer_execution_plan([0, 1, 1, 2])
        inputs = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            expected = model(inputs, use_cache=False).logits
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = type(model).from_pretrained(directory).eval()
            self.assertEqual(restored.get_layer_execution_plan(), model.get_layer_execution_plan())
            self.assertEqual(set(restored.state_dict()), set(model.state_dict()))
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs, use_cache=False).logits, expected, atol=0, rtol=0)
        restored = pickle.loads(pickle.dumps(model))
        with torch.no_grad():
            torch.testing.assert_close(restored(inputs, use_cache=False).logits, expected, atol=0, rtol=0)
        restored.set_layer_execution_plan(None)
        self.assertIsNotNone(model.get_layer_execution_plan())

    def test_nested_container_calls_original_modules(self):
        model = self.make_model("opt")
        order = (0, 1, 1, 2)
        reference = self.reference(model, order)
        source_decoder = model.get_decoder()
        decoder = _NestedOPTDecoder(model.config)
        decoder.load_state_dict(
            {
                name.replace("layers.", "block_stack.blocks."): value
                for name, value in source_decoder.state_dict().items()
            }
        )
        model.set_decoder(decoder)
        model.eval()
        inputs = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            original = model(inputs, use_cache=False).logits
        identities = {name: id(parameter) for name, parameter in model.named_parameters()}
        model.set_layer_execution_plan(order)
        calls = []
        with decoder.block_stack["blocks"][1].register_forward_hook(lambda module, args, output: calls.append(module)):
            with torch.no_grad():
                torch.testing.assert_close(
                    model(inputs, use_cache=False).logits, reference(inputs, use_cache=False).logits
                )
        self.assertEqual(calls, [decoder.layers[1], decoder.layers[1]])
        self.assertEqual(identities, {name: id(parameter) for name, parameter in model.named_parameters()})
        model.set_layer_execution_plan(None)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs, use_cache=False).logits, original, atol=0, rtol=0)

    def test_adapter_rejects_unsupported_stack_behavior(self):
        model = self.make_model("opt")
        model.get_decoder().layerdrop = 0.1
        with self.assertRaisesRegex(ValueError, "layerdrop"):
            model.set_layer_execution_plan([0, 1, 1, 2])
        self.assertIsNone(model.get_layer_execution_plan())
        self.assertFalse(hasattr(model.get_decoder(), "_layer_execution_original_forward"))

    def test_pickled_custom_adapters_generate_in_a_fresh_process(self):
        payloads = []
        for family in ("opt", "mamba"):
            model = self.make_model(family)
            model.set_layer_execution_plan([0, 1, 1, 2])
            with torch.no_grad():
                expected = model.generate(
                    torch.tensor([[1, 2, 3]]), use_cache=False, max_new_tokens=3, do_sample=False
                )
            payloads.append((pickle.dumps(model), expected.tolist()))
        mp.spawn(_spawned_adapter_worker, args=(payloads,), nprocs=1, join=True)
