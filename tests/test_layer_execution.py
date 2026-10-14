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
import pickle
import re
import tempfile
import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        BertConfig,
        BertModel,
        DecoderLayerExecutionAdapter,
        DynamicCache,
        GPT2Config,
        GPT2LMHeadModel,
        LayerExecutionCache,
        LayerExecutionPlan,
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedConfig,
        PreTrainedModel,
        Qwen3_5Config,
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5TextConfig,
        RepeatRange,
        T5Config,
        T5ForConditionalGeneration,
        get_layer_execution_plan,
        register_layer_execution_adapter,
        set_layer_execution_plan,
    )

    class _NestedCausalLM(PreTrainedModel):
        config_class = LlamaConfig
        base_model_prefix = "language_model"
        _supports_attention_backend = True
        _supports_sdpa = True

        def __init__(self, config):
            super().__init__(config)
            self.language_model = LlamaForCausalLM(config)
            self.post_init()

        def forward(self, *args, **kwargs):
            return self.language_model(*args, **kwargs)


@require_torch
class LayerExecutionPlanTest(unittest.TestCase):
    def test_range_semantics(self):
        self.assertEqual(LayerExecutionPlan.from_repeats(4, [RepeatRange(1, 2)]).layer_order, (0, 1, 1, 2, 3))
        self.assertEqual(LayerExecutionPlan.from_repeats(4, [RepeatRange(1, 3)]).layer_order, (0, 1, 2, 1, 2, 3))
        self.assertEqual(LayerExecutionPlan.from_repeats(4, [RepeatRange(0, 4)]).layer_order, tuple(range(4)) * 2)
        self.assertEqual(
            LayerExecutionPlan.from_repeats(4, [RepeatRange(3, 4), RepeatRange(0, 1)]).layer_order,
            (0, 0, 1, 2, 3, 3),
        )

    def test_invalid_plans(self):
        for order in ([], [-1], [True], [1.0]):
            with self.subTest(order=order), self.assertRaises(ValueError):
                LayerExecutionPlan(order)
        for args in ((0, 0, 2), (-1, 2, 2), (0, 1, 0), (True, 2, 2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                RepeatRange(*args)
        with self.assertRaises(ValueError):
            LayerExecutionPlan.from_repeats(3, [RepeatRange(0, 2), RepeatRange(1, 3)])
        with self.assertRaises(ValueError):
            LayerExecutionPlan.from_repeats(3, [RepeatRange(2, 4)])
        with self.assertRaises(ValueError):
            LayerExecutionPlan((3,)).validate(3)

    def test_plan_does_not_alias_input(self):
        order = [0, 1, 1, 2]
        plan = LayerExecutionPlan(order)
        order[0] = 2
        self.assertEqual(plan.layer_order, (0, 1, 1, 2))

    def test_inactive_plan_does_not_resolve_ambiguous_configs(self):
        config = PreTrainedConfig()
        config.decoder = LlamaConfig()
        config.text_config = LlamaConfig()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            config.get_text_config(decoder=True)
        self.assertIsNone(config._get_layer_execution_config())
        config.text_config.layer_execution_plan = [0, 0]
        self.assertIs(config._get_layer_execution_config(), config.text_config)
        config.layer_execution_plan = [0]
        self.assertIs(config._get_layer_execution_config(), config)


@require_torch
class LayerExecutionModelTest(unittest.TestCase):
    def make_model(self, family):
        torch.manual_seed(17)
        common = {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "pad_token_id": 0,
            "eos_token_id": None,
            "attention_dropout": 0.0,
        }
        if family == "llama":
            return LlamaForCausalLM(LlamaConfig(**common)).cpu().eval()
        config = Qwen3_5TextConfig(
            **common,
            head_dim=8,
            layer_types=["linear_attention", "full_attention", "linear_attention"],
            linear_conv_kernel_dim=2,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [1, 0, 0],
                "mrope_interleaved": True,
            },
        )
        return Qwen3_5ForCausalLM(config).cpu().eval()

    def expanded_reference(self, model, order):
        # Independently instantiated logical layers give a reference without cache remapping or shared modules.
        config = copy.deepcopy(model.config)
        config.layer_execution_plan = None
        text_config = config.get_text_config()
        source_config = model.config.get_text_config()
        text_config.layer_execution_plan = None
        text_config.num_hidden_layers = len(order)
        if getattr(text_config, "layer_types", None) is not None:
            layer_types = [source_config.layer_types[index] for index in order]
            if text_config.layer_types != layer_types:
                text_config.layer_types = layer_types
        reference = type(model)(config).cpu().eval()
        source_state = model.state_dict()
        reference_state = {}
        for name in reference.state_dict():
            source_name = re.sub(r"\.(layers|h)\.(\d+)\.", lambda match: f".{match[1]}.{order[int(match[2])]}.", name)
            reference_state[name] = source_state[source_name]
        reference.load_state_dict(reference_state)
        return reference

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_model_api_and_plan_snapshots(self, family):
        model = self.make_model(family)
        self.assertIsNone(model.get_layer_execution_plan())
        self.assertIs(model.set_layer_execution_plan(repeats=[RepeatRange(1, 2)]), model)
        snapshot = get_layer_execution_plan(model)
        self.assertEqual(snapshot.layer_order, (0, 1, 1, 2))
        self.assertIs(set_layer_execution_plan(model, [0, 1, 0, 1, 2]), model)
        self.assertEqual(model.get_layer_execution_plan().layer_order, (0, 1, 0, 1, 2))
        self.assertEqual(snapshot.layer_order, (0, 1, 1, 2))
        with self.assertRaisesRegex(ValueError, "either plan or repeats"):
            model.set_layer_execution_plan([0, 1, 2], repeats=[RepeatRange(1, 2)])
        with self.assertRaises(ValueError):
            model.set_layer_execution_plan([3])
        self.assertEqual(model.get_layer_execution_plan().layer_order, (0, 1, 0, 1, 2))
        model.set_layer_execution_plan()
        self.assertIsNone(model.get_layer_execution_plan())

    def test_enable_uses_configuration_width_with_sharded_embeddings(self):
        model = self.make_model("llama")
        embedding = model.get_input_embeddings()
        weight = embedding.weight
        try:
            embedding.weight = torch.nn.Parameter(weight.detach().flatten())
            model.set_layer_execution_plan(repeats=[RepeatRange(1, 2)])
        finally:
            embedding.weight = weight
        with torch.no_grad():
            model(torch.tensor([[1, 2, 3]]), use_cache=False)

    def test_data_parallel_replica_rebinds_decoder_methods(self):
        model = self.make_model("llama")
        model.set_layer_execution_plan(repeats=[RepeatRange(1, 2)])
        decoder = model.get_decoder()
        replica = decoder._replicate_for_data_parallel()
        replica._modules = copy.deepcopy(decoder._modules)
        replica.config = copy.deepcopy(decoder.config)
        with torch.no_grad():
            replica.norm.weight.mul_(2)
        self.assertIs(replica.forward.args[0], replica)
        self.assertIs(replica._layer_execution_original_forward.__self__, replica)
        inputs = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            torch.testing.assert_close(
                replica(inputs, use_cache=False).last_hidden_state,
                2 * decoder(inputs, use_cache=False).last_hidden_state,
            )
        replica.set_layer_execution_plan(None)
        self.assertIs(replica.forward.__self__, replica)
        self.assertIsNotNone(decoder.get_layer_execution_plan())

    def test_nested_causal_lm_wrapper_resolves_the_decoder_stack(self):
        model = _NestedCausalLM(self.make_model("llama").config).eval()
        order = (0, 1, 1, 2)
        reference = self.expanded_reference(model, order)
        identities = {name: id(parameter) for name, parameter in model.named_parameters()}
        inputs = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            original = model(inputs, use_cache=False).logits
        model.set_layer_execution_plan(order)
        self.assertEqual(model.get_layer_execution_plan().layer_order, order)
        self.assertFalse(hasattr(model.language_model, "_layer_execution_adapter"))
        self.assertTrue(hasattr(model.language_model.model, "_layer_execution_adapter"))
        with torch.no_grad():
            torch.testing.assert_close(
                model(inputs, use_cache=False).logits, reference(inputs, use_cache=False).logits
            )
        self.assertEqual(identities, {name: id(parameter) for name, parameter in model.named_parameters()})
        model.set_layer_execution_plan(None)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs, use_cache=False).logits, original, atol=0, rtol=0)

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_pickle_preserves_execution_and_original_forward(self, family):
        model = self.make_model(family)
        inputs = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            original = model(inputs, use_cache=False).logits
        model.set_layer_execution_plan([0, 1, 1, 2])
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(restored.get_layer_execution_plan(), model.get_layer_execution_plan())
        with torch.no_grad():
            torch.testing.assert_close(restored(inputs, use_cache=False).logits, model(inputs, use_cache=False).logits)
            restored.set_layer_execution_plan(None)
            torch.testing.assert_close(restored(inputs, use_cache=False).logits, original, atol=0, rtol=0)
        self.assertIsNotNone(model.get_layer_execution_plan())

    def test_trainer_accounts_for_omitted_layers_with_checkpointing(self):
        from transformers import Trainer, TrainingArguments

        model = self.make_model("llama")
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory, use_cpu=True, gradient_checkpointing=True, report_to="none"
                ),
            )
            for order, expected in ((None, False), ([0, 1, 0, 1, 2], False), ([0, 0, 2], True)):
                model.set_layer_execution_plan(order)
                self.assertEqual(
                    trainer._build_accelerator_args()["kwargs_handlers"][0].find_unused_parameters, expected
                )
            trainer.args.ddp_find_unused_parameters = False
            self.assertFalse(trainer._build_accelerator_args()["kwargs_handlers"][0].find_unused_parameters)

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_no_grad_rollout_cache_preserves_training_mode(self, family):
        model = self.make_model(family)
        model.set_layer_execution_plan([0, 1, 1, 2])
        model.train()
        inputs = torch.tensor([[1, 2, 3]])
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                cache = model(inputs, use_cache=True).past_key_values
                self.assertIsInstance(cache, LayerExecutionCache)
                output = model(torch.tensor([[4]]), past_key_values=cache).logits
                reference = model(torch.tensor([[1, 2, 3, 4]]), use_cache=False).logits[:, -1:]
                torch.testing.assert_close(output, reference, atol=3e-5, rtol=3e-4)
                torch.testing.assert_close(
                    model.generate(inputs, max_new_tokens=3),
                    model.generate(inputs, max_new_tokens=3, use_cache=False),
                )
            self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, "requires past_key_values=None"):
            model(inputs, past_key_values=cache)
        output = model(inputs, labels=inputs, use_cache=True)
        self.assertIsNone(output.past_key_values)
        output.loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_cache_views_expose_source_metadata(self, family):
        from transformers.layer_execution.cache import _ExecutionCacheView

        model = self.make_model(family)
        model.set_layer_execution_plan([2, 1, 2, 0])
        with torch.no_grad():
            cache = model(torch.tensor([[1, 2, 3]]), use_cache=True).past_key_values
        view = _ExecutionCacheView(cache, source_index=2, execution_index=0)
        self.assertEqual(view.get_seq_length(), 3)
        self.assertEqual(view.get_max_length(), -1)
        self.assertEqual(view.batch_size, cache.batch_size)
        self.assertFalse(view.is_compileable)
        self.assertEqual(view.is_initialized, family == "llama")
        self.assertEqual(view.is_sliding[2], cache.is_sliding[0])
        self.assertEqual(view.is_linear[2], cache.is_linear[0])
        self.assertIs(view.layers[2], cache.layers[0])
        with self.assertRaisesRegex(ValueError, "another source layer"):
            view.get_max_length(0)

    @parameterized.expand([(family, kind) for family in ("llama", "qwen3_5") for kind in ("single", "range", "full")])
    def test_forward_matches_expanded_reference_and_preserves_parameters(self, family, kind):
        model = self.make_model(family)
        repeat = {"single": RepeatRange(1, 2), "range": RepeatRange(0, 2), "full": RepeatRange(0, 3)}[kind]
        plan = LayerExecutionPlan.from_repeats(3, [repeat])
        reference = self.expanded_reference(model, plan.layer_order)
        identities = {name: id(parameter) for name, parameter in model.named_parameters()}
        keys = set(model.state_dict())
        set_layer_execution_plan(model, plan)
        inputs = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            actual = model(inputs, use_cache=False, output_hidden_states=True)
            expected = reference(inputs, use_cache=False)
        torch.testing.assert_close(actual.logits, expected.logits, atol=1e-6, rtol=1e-5)
        self.assertEqual(identities, {name: id(parameter) for name, parameter in model.named_parameters()})
        self.assertEqual(keys, set(model.state_dict()))
        self.assertEqual(model.config.num_hidden_layers, 3)
        self.assertEqual(len(model.model.layers), 3)
        self.assertEqual(len(actual.hidden_states), len(plan.layer_order) + 1)

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_identity_and_disable_restore_original(self, family):
        model = self.make_model(family)
        inputs = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            original = model(inputs, use_cache=False).logits
            set_layer_execution_plan(model, LayerExecutionPlan((0, 1, 2)))
            torch.testing.assert_close(model(inputs, use_cache=False).logits, original, atol=0, rtol=0)
            set_layer_execution_plan(model, LayerExecutionPlan((0, 1, 1, 2)))
            looped = model(inputs, use_cache=False).logits
            set_layer_execution_plan(model, None)
            torch.testing.assert_close(model(inputs, use_cache=False).logits, original, atol=0, rtol=0)
            restored = model(inputs, use_cache=True)
            self.assertNotIsInstance(restored.past_key_values, LayerExecutionCache)
            torch.testing.assert_close(restored.logits, original, atol=0, rtol=0)
            set_layer_execution_plan(model, LayerExecutionPlan((0, 1, 1, 2)))
            torch.testing.assert_close(model(inputs, use_cache=True).logits, looped, atol=0, rtol=0)

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_independent_cache_and_prefill_decode_parity(self, family):
        model = self.make_model(family)
        plan = LayerExecutionPlan.from_repeats(3, [RepeatRange(0, 3)])
        reference = self.expanded_reference(model, plan.layer_order)
        set_layer_execution_plan(model, plan)
        prefix = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            actual = model(prefix, use_cache=True)
            expected = reference(prefix, use_cache=True)
            cache = actual.past_key_values
            self.assertIsInstance(cache, LayerExecutionCache)
            self.assertEqual(len(cache.layers), 6)
            self.assertEqual(len({id(layer) for layer in cache.layers}), 6)
            self.assertEqual(cache.get_seq_length(), prefix.shape[1])
            for position in range(3):
                left, right = cache.layers[position], cache.layers[position + 3]
                if hasattr(left, "keys"):
                    self.assertNotEqual(left.keys.data_ptr(), right.keys.data_ptr())
                    self.assertNotEqual(left.values.data_ptr(), right.values.data_ptr())
                else:
                    self.assertNotEqual(left.conv_states[0].data_ptr(), right.conv_states[0].data_ptr())
                    self.assertNotEqual(left.recurrent_states[0].data_ptr(), right.recurrent_states[0].data_ptr())
            for continuation in ([4], [5, 6]):
                tokens = torch.tensor([continuation])
                prefix = torch.cat((prefix, tokens), dim=1)
                actual = model(tokens, past_key_values=cache, use_cache=True)
                expected = reference(tokens, past_key_values=expected.past_key_values, use_cache=True)
                uncached = model(prefix, use_cache=False)
                torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-4)
                torch.testing.assert_close(actual.logits, uncached.logits[:, -tokens.shape[1] :], atol=2e-5, rtol=2e-4)
                self.assertEqual(cache.get_seq_length(), prefix.shape[1])
            cache.reset()
            self.assertEqual(cache.get_seq_length(), 0)

    @parameterized.expand([(family, beams) for family in ("llama", "qwen3_5") for beams in (1, 2)])
    def test_generate_cached_and_uncached(self, family, beams):
        model = self.make_model(family)
        set_layer_execution_plan(model, LayerExecutionPlan((0, 1, 0, 1, 2)))
        inputs = torch.tensor([[1, 2, 3], [2, 3, 4]])
        actual = model.generate(inputs, use_cache=True, num_beams=beams, max_new_tokens=3, do_sample=False)
        expected = model.generate(inputs, use_cache=False, num_beams=beams, max_new_tokens=3, do_sample=False)
        torch.testing.assert_close(actual, expected)

    @parameterized.expand([(family, checkpoint) for family in ("llama", "qwen3_5") for checkpoint in (False, True)])
    def test_shared_gradients_match_sum_of_expanded_gradients(self, family, checkpoint):
        model = self.make_model(family)
        order = (0, 1, 0, 1, 2)
        reference = self.expanded_reference(model, order)
        set_layer_execution_plan(model, LayerExecutionPlan(order))
        model.train()
        reference.train()
        if checkpoint:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs = torch.tensor([[1, 2, 3, 4]])
        output = model(inputs, labels=inputs)
        self.assertIsNone(output.past_key_values)
        expected = reference(inputs, labels=inputs, use_cache=False)
        torch.testing.assert_close(output.loss, expected.loss, atol=1e-6, rtol=1e-5)
        output.loss.backward()
        expected.loss.backward()
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            match = re.search(r"model\.layers\.(\d+)\.", name)
            if match:
                source_index = int(match[1])
                gradients = [
                    reference_parameters[name.replace(f"layers.{source_index}.", f"layers.{position}.")].grad
                    for position, source in enumerate(order)
                    if source == source_index
                ]
                expected_gradient = sum(gradients)
            else:
                expected_gradient = reference_parameters[name].grad
            torch.testing.assert_close(parameter.grad, expected_gradient, atol=3e-5, rtol=3e-4)

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_checkpoint_restores_plan_and_original_weight_keys(self, family):
        model = self.make_model(family)
        keys = set(model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = AutoModelForCausalLM.from_pretrained(directory).eval()
            set_layer_execution_plan(loaded, LayerExecutionPlan((0, 1, 1, 2)))
            inputs = torch.tensor([[1, 2, 3]])
            with torch.no_grad():
                expected = loaded(inputs, use_cache=False).logits
            loaded.save_pretrained(directory)
            restored = AutoModelForCausalLM.from_pretrained(directory).eval()
            self.assertEqual(restored.config.layer_execution_plan, [0, 1, 1, 2])
            self.assertEqual(set(restored.state_dict()), keys)
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs, use_cache=False).logits, expected, atol=0, rtol=0)

    def test_reject_incompatible_cache_and_changed_plan(self):
        model = self.make_model("llama")
        set_layer_execution_plan(model, LayerExecutionPlan((0, 1, 1, 2)))
        inputs = torch.tensor([[1, 2]])
        with self.assertRaisesRegex(ValueError, "LayerExecutionCache"):
            model(inputs, past_key_values=DynamicCache(config=model.config))
        cache = model(inputs, use_cache=True).past_key_values
        set_layer_execution_plan(model, LayerExecutionPlan((0, 1, 2, 2)))
        with self.assertRaisesRegex(ValueError, "clear the cache"):
            model(inputs, past_key_values=cache)
        with self.assertRaisesRegex(ValueError, "Unsupported layer execution cache"):
            LayerExecutionCache(model.config, cache_implementation="unknown")
        set_layer_execution_plan(model, None)
        with self.assertRaisesRegex(ValueError, "clear the cache"):
            model(inputs, past_key_values=cache)
        with self.assertRaisesRegex(ValueError, "clear the cache"):
            model.model(inputs, None, None, cache)

    @parameterized.expand([("linear", (0, 0, 2)), ("full", (1, 1))])
    def test_cache_with_only_one_attention_type(self, kind, order):
        model = self.make_model("qwen3_5")
        set_layer_execution_plan(model, LayerExecutionPlan(order))
        prefix = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            cache = model(prefix, use_cache=True).past_key_values
            output = model(torch.tensor([[4, 5]]), past_key_values=cache, use_cache=True)
            expected = model(torch.tensor([[1, 2, 3, 4, 5]]), use_cache=False)
        torch.testing.assert_close(output.logits, expected.logits[:, -2:], atol=2e-5, rtol=2e-4)
        self.assertEqual(cache.get_seq_length(), 5)
        cached = model.generate(prefix, max_new_tokens=3, do_sample=False)
        uncached = model.generate(prefix, use_cache=False, max_new_tokens=3, do_sample=False)
        torch.testing.assert_close(cached, uncached)

    def test_crop_preserves_depth_independent_token_count(self):
        model = self.make_model("llama")
        set_layer_execution_plan(model, LayerExecutionPlan((0, 1, 0, 1, 2)))
        prefix = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            cache = model(prefix, use_cache=True).past_key_values
            cache.crop(0)
            self.assertEqual(cache.get_seq_length(), 3)
            cache.crop(-1)
            self.assertEqual(cache.get_seq_length(), 2)
            actual = model(torch.tensor([[4]]), past_key_values=cache, use_cache=True)
            expected = model(torch.tensor([[1, 2, 4]]), use_cache=False)
        torch.testing.assert_close(actual.logits, expected.logits[:, -1:], atol=1e-6, rtol=1e-5)

    @parameterized.expand([("llama",), ("qwen3_5",)])
    def test_padded_cache_continuation_and_inputs_embeds(self, family):
        model = self.make_model(family)
        set_layer_execution_plan(model, LayerExecutionPlan((0, 1, 0, 1, 2)))
        input_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        mask = input_ids.ne(0).long()
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            cached = model(input_ids[:, :2], attention_mask=mask[:, :2], position_ids=positions[:, :2])
            output = model(
                inputs_embeds=model.get_input_embeddings()(input_ids[:, 2:]),
                attention_mask=mask,
                position_ids=positions[:, 2:],
                past_key_values=cached.past_key_values,
            )
            expected = model(input_ids, attention_mask=mask, position_ids=positions, use_cache=False)
        torch.testing.assert_close(output.logits, expected.logits[:, -2:], atol=2e-5, rtol=2e-4)

    def test_qwen_multimodal_wrapper_and_checkpoint(self):
        text_config = self.make_model("qwen3_5").config
        config = Qwen3_5Config(
            text_config=text_config,
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "patch_size": 2,
                "spatial_merge_size": 1,
                "temporal_patch_size": 1,
                "out_hidden_size": 16,
                "num_position_embeddings": 16,
            },
            image_token_id=28,
            video_token_id=29,
            vision_start_token_id=30,
            vision_end_token_id=31,
            layer_execution_plan=[0, 1, 0, 1, 2],
        )
        model = Qwen3_5ForConditionalGeneration(config).eval()
        order = (0, 1, 0, 1, 2)
        self.assertIsNone(model.config.layer_execution_plan)
        self.assertEqual(model.config.text_config.layer_execution_plan, list(order))
        reference = self.expanded_reference(model, order)
        keys = set(model.state_dict())
        set_layer_execution_plan(model, LayerExecutionPlan(order))
        inputs = {
            "input_ids": torch.tensor([[1, 30, 28, 28, 28, 28, 31, 2]]),
            "pixel_values": torch.randn(4, 12),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]]),
        }
        with torch.no_grad():
            actual = model(**inputs)
            expected = reference(**inputs, use_cache=False)
            torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-4)
            self.assertIsInstance(actual.past_key_values, LayerExecutionCache)
            cached = model.generate(**inputs, max_new_tokens=2, do_sample=False)
            uncached = model.generate(**inputs, use_cache=False, max_new_tokens=2, do_sample=False)
            torch.testing.assert_close(cached, uncached)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = Qwen3_5ForConditionalGeneration.from_pretrained(directory).eval()
            self.assertEqual(restored.config.text_config.layer_execution_plan, list(order))
            self.assertEqual(set(restored.state_dict()), keys)
            self.assertIsNone(restored.config.vision_config.layer_execution_plan)
            with torch.no_grad():
                torch.testing.assert_close(restored(**inputs, use_cache=False).logits, expected.logits, atol=0, rtol=0)
            text_model = AutoModelForCausalLM.from_pretrained(directory).eval()
            self.assertEqual(text_model.get_layer_execution_plan().layer_order, order)
            text_ids = torch.tensor([[1, 2, 3]])
            with torch.no_grad():
                torch.testing.assert_close(
                    text_model(text_ids, use_cache=False).logits,
                    model(text_ids, use_cache=False).logits,
                    atol=0,
                    rtol=0,
                )

    def test_adapter_registration(self):
        from transformers.layer_execution.adapters import _ADAPTERS
        from transformers.masking_utils import create_causal_mask
        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

        class CustomAdapter(DecoderLayerExecutionAdapter):
            layer_container_name = "h"

            def prepare(self, decoder, inputs_embeds, attention_mask, position_ids, cache, **kwargs):
                if position_ids is None:
                    offset = cache.get_seq_length() if cache is not None else 0
                    position_ids = (torch.arange(inputs_embeds.shape[1]) + offset).unsqueeze(0)
                hidden_states = inputs_embeds + decoder.wpe(position_ids)
                if kwargs.get("token_type_ids") is not None:
                    hidden_states = hidden_states + decoder.wte(kwargs["token_type_ids"])
                return decoder.drop(hidden_states), {
                    "position_ids": position_ids,
                    "attention_mask": create_causal_mask(decoder.config, inputs_embeds, attention_mask, cache),
                }

            def extract_hidden_states(self, layer_output):
                return layer_output if isinstance(layer_output, torch.Tensor) else layer_output[0]

            def finalize(self, decoder, hidden_states, cache, context):
                return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=decoder.ln_f(hidden_states), past_key_values=cache
                )

        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=32,
                n_embd=16,
                n_layer=3,
                n_head=2,
                n_inner=32,
                n_positions=32,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=None,
            )
        ).eval()
        self.addCleanup(_ADAPTERS.pop, model.config.model_type)
        register_layer_execution_adapter(model.config.model_type, CustomAdapter)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_layer_execution_adapter(model.config.model_type, CustomAdapter)
        order = (0, 1, 1, 2)
        reference = self.expanded_reference(model, order)
        set_layer_execution_plan(model, LayerExecutionPlan(order))
        inputs = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            token_types = torch.ones_like(inputs)
            expected = reference(inputs, token_type_ids=token_types, use_cache=False)
            actual = model(inputs, token_type_ids=token_types, use_cache=True)
            positional = model.transformer(inputs, None, None, token_types, use_cache=False)
            torch.testing.assert_close(
                positional.last_hidden_state,
                model.transformer(inputs, token_type_ids=token_types, use_cache=False).last_hidden_state,
            )
            torch.testing.assert_close(actual.logits, expected.logits)
            with model.transformer.h[1].register_forward_hook(lambda module, args, output: (output, None)):
                structured = model(inputs, token_type_ids=token_types, use_cache=False)
                torch.testing.assert_close(structured.logits, expected.logits)
            continuation = model(
                torch.tensor([[4]]),
                token_type_ids=token_types[:, :1],
                past_key_values=actual.past_key_values,
            )
            expected = reference(
                torch.tensor([[1, 2, 3, 4]]), token_type_ids=torch.ones(1, 4, dtype=torch.long), use_cache=False
            )
            torch.testing.assert_close(continuation.logits, expected.logits[:, -1:])
            cached = model.generate(inputs, max_new_tokens=3, do_sample=False)
            uncached = model.generate(inputs, use_cache=False, max_new_tokens=3, do_sample=False)
            torch.testing.assert_close(cached, uncached)
            set_layer_execution_plan(model, None)
            with self.assertRaisesRegex(ValueError, "clear the cache"):
                model.transformer(inputs, actual.past_key_values)

    def test_reject_width_changes_and_pipeline_parallelism(self):
        from transformers.distributed.pipeline_parallel import apply_pipeline_parallelism

        model = self.make_model("llama")
        model.config.per_layer_config = {1: {"hidden_size": 32}}
        with self.assertRaisesRegex(ValueError, "same hidden size"):
            set_layer_execution_plan(model, LayerExecutionPlan((0, 1, 1, 2)))
        model.config.per_layer_config = None
        set_layer_execution_plan(model, LayerExecutionPlan((0, 1, 1, 2)))
        with self.assertRaisesRegex(ValueError, "[Pp]ipeline parallelism"):
            apply_pipeline_parallelism(model, None)
        set_layer_execution_plan(model, None)
        model._pp_stage = object()
        with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
            set_layer_execution_plan(model, LayerExecutionPlan((0, 1, 1, 2)))
        del model._pp_stage
        model.model._pp_stage = object()
        with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
            model.set_layer_execution_plan(repeats=[RepeatRange(1, 2)])

    def test_reject_encoder_and_encoder_decoder(self):
        encoder = BertModel(
            BertConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2)
        )
        with self.assertRaisesRegex(ValueError, "adapter"):
            set_layer_execution_plan(encoder, LayerExecutionPlan((0, 0)))
        seq2seq = T5ForConditionalGeneration(T5Config(d_model=16, d_ff=32, num_layers=1, num_heads=2))
        with self.assertRaisesRegex(ValueError, "decoder-only"):
            set_layer_execution_plan(seq2seq, LayerExecutionPlan((0, 0)))
