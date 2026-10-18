# Copyright 2024 The HuggingFace Team. All rights reserved.
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


import unittest

from transformers import AutoTokenizer, Mamba2Config, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_kernels,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import DynamicCache, Mamba2ForCausalLM, Mamba2Model
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer, MambaRMSNormGated


class Mamba2ConfigTester(ConfigTester):
    def _create_config(self, hidden_size: int, num_heads: int, expand: int, head_dim: int):
        _input_dict = self.inputs_dict.copy()
        _input_dict["hidden_size"] = hidden_size
        _input_dict["num_heads"] = num_heads
        _input_dict["expand"] = expand
        _input_dict["head_dim"] = head_dim
        return self.config_class(**_input_dict)

    def test_hidden_size_compatibility(self):
        self._create_config(hidden_size=2, num_heads=2, expand=2, head_dim=2)
        self._create_config(hidden_size=4, num_heads=4, expand=2, head_dim=2)
        self._create_config(hidden_size=2, num_heads=4, expand=4, head_dim=2)
        with self.parent.assertRaises(ValueError):
            self._create_config(hidden_size=2, num_heads=4, expand=2, head_dim=4)
        with self.parent.assertRaises(ValueError):
            self._create_config(hidden_size=4, num_heads=2, expand=4, head_dim=2)

    def run_common_tests(self):
        self.test_hidden_size_compatibility()
        return super().run_common_tests()


class Mamba2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        num_heads=8,
        n_groups=8,
        state_size=2,
        head_dim=8,
        conv_kernel=4,
        chunk_size=8,
        seq_length=7,
        is_training=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        tie_word_embeddings=False,
    ):
        self.parent = parent
        self.num_heads = num_heads
        self.n_groups = n_groups
        self.head_dim = head_dim
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1
        self.tie_word_embeddings = tie_word_embeddings

    def prepare_config_and_inputs(
        self, gradient_checkpointing=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False
    ):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        # Only left padding is valid
        attention_mask = torch.ones(size=(self.batch_size, self.seq_length), device=input_ids.device, dtype=torch.long)
        attention_mask[0, :1] = 0

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config(
            gradient_checkpointing=gradient_checkpointing,
        )

        return (
            config,
            input_ids,
            attention_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self, gradient_checkpointing=False):
        return Mamba2Config(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            n_groups=self.n_groups,
            state_size=self.state_size,
            conv_kernel=self.conv_kernel,
            chunk_size=self.chunk_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            activation_function=self.hidden_act,
            n_positions=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def prepare_config_and_inputs_for_common(self):
        (
            config,
            input_ids,
            _,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict

    def create_and_check_mamba2_caching(self, config, input_ids, attention_mask, *args):
        model = Mamba2Model(config=config)
        model.to(torch_device)
        model.eval()

        output_whole = model(input_ids, attention_mask=attention_mask).last_hidden_state

        outputs = model(
            input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            use_cache=True,
        )
        output_one = outputs.last_hidden_state

        # Using the state computed on the first inputs, we will get the same output
        outputs = model(
            input_ids[:, -1:],
            attention_mask=attention_mask[:, -1:],
            use_cache=True,
            cache_params=outputs.cache_params,
        )
        output_two = outputs.last_hidden_state

        self.parent.assertTrue(
            torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-3, rtol=1e-3)
        )

    def create_and_check_mamba2_chunked_prefill(self, config, input_ids, *args, device="cpu"):
        """
        Adapted from `test_linear_attention_multi_token_cached_forward_matches_single_token`
        to check whether multi-token cached input is properly handled.

        Can either be run on GPU (fast path) or CPU (slow path), see `test_mamba2_chunked_prefill_*`
        """
        model = Mamba2Model(config=config)
        model.to(device)
        model.eval()

        input_ids = input_ids[:1].to(device)
        prefill_len = input_ids.shape[1] // 2 + 1
        prompt = input_ids[:, :prefill_len]
        next_token = input_ids[:, prefill_len : prefill_len + 1]
        distractors = input_ids[:, prefill_len + 1 :]
        multi_input = torch.cat([next_token, distractors], dim=1)

        cache_single = DynamicCache(config=config)
        with torch.no_grad():
            model(input_ids=prompt, cache_params=cache_single, use_cache=True)
            single_out = model(input_ids=next_token, cache_params=cache_single, use_cache=True)
        ref_first = single_out.last_hidden_state[:, 0, :]

        cache_multi = DynamicCache(config=config)
        with torch.no_grad():
            model(input_ids=prompt, cache_params=cache_multi, use_cache=True)
            multi_out = model(input_ids=multi_input, cache_params=cache_multi, use_cache=True)
        under_test_first = multi_out.last_hidden_state[:, 0, :]

        self.parent.assertTrue(
            torch.allclose(ref_first, under_test_first, atol=1e-4, rtol=1e-4),
            msg=f"Max diff: {(ref_first - under_test_first).abs().max().item():.6f}",
        )

    def create_and_check_mamba2_slow_vs_fast_forward(self, config, input_ids, *args, gradient_checkpointing=False):
        """Slow vs fast path check guarded by require kernels to enable fast path"""
        model = Mamba2Model(config)
        model.eval()
        model.to(torch_device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        token_emb = model.embeddings(input_ids)
        outputs_fast = model.layers[0].mixer.cuda_kernels_forward(token_emb)
        outputs_slow = model.layers[0].mixer.torch_forward(token_emb)

        self.parent.assertTrue(torch.allclose(outputs_fast, outputs_slow, atol=1e-3, rtol=1e-3))

    def create_and_check_kwargs_reach_mamba2_mixer(self, config, input_ids, *args):
        """
        Kernel kwargs given to the model must reach the Mamba2 mixer, which splats them into
        the fused conv1d+scan, the conv and the chunk scan. This is how `seq_idx` reaches the
        kernels for packed / variable-length batches.
        """
        model = Mamba2Model(config)
        model.to(torch_device)
        model.eval()

        mixer = model.layers[0].mixer
        original_forward = mixer.forward
        seen = []

        def recording_forward(*fwd_args, **fwd_kwargs):
            seen.append(set(fwd_kwargs))
            return original_forward(*fwd_args, **fwd_kwargs)

        mixer.forward = recording_forward

        input_ids = input_ids.to(torch_device)
        seq_idx = torch.zeros(input_ids.shape, dtype=torch.int32, device=torch_device)
        with torch.no_grad():
            model(input_ids, seq_idx=seq_idx)

        self.parent.assertTrue(seen, "the Mamba2 mixer was never called")
        self.parent.assertIn("seq_idx", seen[0])


@require_torch
class Mamba2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Mamba2Model, Mamba2ForCausalLM) if is_torch_available() else ()
    has_attentions = False  # Mamba does not support attentions
    test_missing_keys = False

    pipeline_model_mapping = (
        {"feature-extraction": Mamba2Model, "text-generation": Mamba2ForCausalLM} if is_torch_available() else {}
    )

    def setUp(self):
        self.model_tester = Mamba2ModelTester(self)
        self.config_tester = Mamba2ConfigTester(
            self, config_class=Mamba2Config, n_embd=37, common_properties=["hidden_size", "num_hidden_layers"]
        )

    def _get_conv_state_shape(self, batch_size: int, config):
        intermediate_size = config.expand * config.hidden_size
        conv_shape = (
            batch_size,
            intermediate_size + 2 * config.n_groups * config.state_size,
            config.conv_kernel,
        )
        return conv_shape

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.num_heads, config.head_dim, config.state_size)

    def test_mamba2_caching(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_caching(*config_and_inputs)

    def test_kwargs_reach_mamba2_mixer(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_kwargs_reach_mamba2_mixer(*config_and_inputs)

    def test_mamba2_chunked_prefill_cpu(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_chunked_prefill(*config_and_inputs, device="cpu")

    @require_torch_accelerator
    @require_kernels
    def test_mamba2_chunked_prefill_torch_device(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_chunked_prefill(*config_and_inputs, device=torch_device)

    @require_torch_accelerator
    @require_kernels
    def test_mamba2_slow_vs_fast_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_slow_vs_fast_forward(*config_and_inputs)

    # This test adjusts n_groups to half the original setting and effectively
    # creates a grouped SSD configuration in the mamba2 layers
    # See https://github.com/huggingface/transformers/pull/37533/
    @require_torch_accelerator
    @require_kernels
    def test_mamba2_slow_vs_fast_forward_grouped(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config_and_inputs[0].n_groups //= 2
        self.model_tester.create_and_check_mamba2_slow_vs_fast_forward(*config_and_inputs)

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, DynamicCache):  # MODIFIED PART START
                        for idx in range(len(tuple_object)):
                            recursive_check(
                                tuple_object.layers[idx].conv_states[0], dict_object.layers[idx].conv_states[0]
                            )
                            recursive_check(
                                tuple_object.layers[idx].recurrent_states[0],
                                dict_object.layers[idx].recurrent_states[0],
                            )
                    elif isinstance(tuple_object, (list, tuple)):  # MODIFIED PART END
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(tuple_object, dict_object, atol=1e-5),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

    def test_tied_weight_embeddings(self):
        """Regression test for https://github.com/huggingface/transformers/issues/43206."""
        config = self.model_tester.get_config()

        config.tie_word_embeddings = True
        model = Mamba2ForCausalLM(config)

        self.assertEqual(
            model.lm_head.weight.data_ptr(),
            model.backbone.embeddings.weight.data_ptr(),
        )

        config.tie_word_embeddings = False
        model = Mamba2ForCausalLM(config)

        self.assertNotEqual(
            model.lm_head.weight.data_ptr(),
            model.backbone.embeddings.weight.data_ptr(),
        )


@require_torch
@slow
class Mamba2IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "mistralai/Mamba-Codestral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, from_slow=True, legacy=False)
        self.prompt = ("[INST]Write a hello world program in C++.",)

    @slow
    @require_torch
    def test_simple_generate(self):
        """
        Simple generate test to avoid regressions.
        Note: state-spaces (cuda) implementation and pure torch implementation
        have irreconciliable differences as of now, which will cause this test to fail
        in an environment with state-spaces installed.
        """
        tokenizer = self.tokenizer
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = Mamba2ForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16)
        model.to(torch_device)
        inputs = tokenizer("[INST]Write a hello world program in C++.[/INST]", return_tensors="pt").to(torch_device)

        out = model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=30)
        output_sentence = tokenizer.decode(out[0])
        ground_truth_sentences = Expectations(
            {
                ("xpu", 3): """<s>[INST]Write a hello world program in C++.[/INST] Sure, here is a simple "Hello, World!" program written in C++:\n\n```cpp\n#include <iostream>\n""",
                ("cuda", 7): """<s>[INST]Write a hello world program in C++.[/INST] Sure, here is a simple "Hello, World!" program in C++:\n\n```cpp\n#include <iostream>\n\n""",
            }
        )  # fmt: skip
        ground_truth_sentence = ground_truth_sentences.get_expectation()
        self.assertEqual(output_sentence, ground_truth_sentence)

    @slow
    @require_torch_accelerator
    def test_batched_equivalence_with_cache(self):
        """
        Verifies that batched generation matches individual generation.
        Important because of the specific caching mechanism + statefulness of mamba model.
        Depending on precision and devices, differences can be observed from generation to generation.
        """
        tokenizer = self.tokenizer
        prompt = [
            "[INST]Write C#.[/INST]",
            "[INST]Write a hello world in C++.[/INST]",
            "[INST] Write a simple Fibonacci number computation function in Rust that does memoization, with comments, in safe Rust.[/INST]",
        ]

        model = Mamba2ForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16).to(torch_device)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # batched generation
        tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding="longest").to(torch_device)
        batched_gen = model.generate(**tokenized_prompts, max_new_tokens=30, use_cache=True)
        batched_output = tokenizer.batch_decode(batched_gen, skip_special_tokens=True)

        # individual generation

        for index_gen, individual_prompt in enumerate(prompt):
            inputs = tokenizer(individual_prompt, return_tensors="pt", padding="longest").to(torch_device)
            individual_gen = model.generate(**inputs, max_new_tokens=30, use_cache=True)
            individual_output = tokenizer.batch_decode(individual_gen, skip_special_tokens=True)[0]
            self.assertEqual(individual_output[:100], batched_output[index_gen][:100])

    @slow
    @require_torch_accelerator
    def test_batched_equivalence_without_cache(self):
        """
        Verifies that batched generation matches individual generation without cache.
        Important because of the specific caching mechanism + statefulness of mamba model.
        Depending on precision and devices, differences can be observed from generation to generation.
        """
        tokenizer = self.tokenizer
        prompt = [
            "[INST]Write C#.[/INST]",
            "[INST]Write a hello world in C++.[/INST]",
            "[INST] Write a simple Fibonacci number computation function in Rust that does memoization, with comments, in safe Rust.[/INST]",
        ]

        model = Mamba2ForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16).to(torch_device)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # batched generation
        tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding="longest").to(torch_device)
        batched_gen = model.generate(**tokenized_prompts, max_new_tokens=30, use_cache=True)
        batched_output = tokenizer.batch_decode(batched_gen, skip_special_tokens=True)

        # individual generation

        for index_gen, individual_prompt in enumerate(prompt):
            inputs = tokenizer(individual_prompt, return_tensors="pt", padding="longest").to(torch_device)
            individual_gen = model.generate(**inputs, max_new_tokens=30, use_cache=True)
            individual_output = tokenizer.batch_decode(individual_gen, skip_special_tokens=True)[0]
            self.assertEqual(individual_output[:100], batched_output[index_gen][:100])

    @slow
    @require_torch_accelerator
    def test_mamba2_mixer_train_vs_eval_equivalence(self):
        # Based on https://github.com/sustcsonglin/flash-linear-attention/issues/63
        # Credit to zhixuan-lin

        B, T, D = 4, 512, 768
        dtype = torch.bfloat16
        config = Mamba2Config(num_heads=24, head_dim=64, hidden_size=768, expand=2, n_groups=1)

        torch.manual_seed(42)
        with torch.autocast(device_type=torch_device, dtype=dtype):
            with torch.no_grad():
                mixer = Mamba2Mixer(config, layer_idx=0).to(torch_device)
                hidden_states = torch.rand(size=(B, T, D), dtype=dtype, device=torch_device)

                mixer.train()
                out_train = mixer(hidden_states)

                mixer.eval()
                out_eval = mixer(hidden_states)

                torch.testing.assert_close(out_train, out_eval, rtol=1e-3, atol=1e-3)


@require_torch
class Mamba2RMSNormGatedTest(unittest.TestCase):
    """
    Regression tests for Mamba2 grouped RMSNorm (Issue #49536).
    Verifies that MambaRMSNormGated normalizes across group_size slices rather than
    the full intermediate dimension when group_size is specified, while preserving
    legacy single-group behavior when group_size is None or equal to hidden_size.
    """

    def _reference_grouped_rmsnorm(self, hidden_states, weight, eps, group_size, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * torch.nn.functional.silu(gate.to(torch.float32))

        input_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(*input_shape[:-1], -1, group_size)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        hidden_states = hidden_states.reshape(input_shape)
        return (weight.to(torch.float32) * hidden_states).to(input_dtype)

    def test_grouped_rmsnorm_matches_reference(self):
        torch.manual_seed(42)
        shapes = [
            (2, 64),          # 2D: (batch_size, hidden_size)
            (2, 8, 128),      # 3D: (batch_size, seq_len, hidden_size)
            (2, 4, 8, 256),   # 4D: (batch, heads, seq_len, hidden_size)
        ]
        group_configs = [
            (64, 4),    # hidden_size=64, n_groups=4 -> group_size=16
            (128, 8),   # hidden_size=128, n_groups=8 -> group_size=16
            (256, 4),   # hidden_size=256, n_groups=4 -> group_size=64
        ]
        dtypes = [torch.float32]
        if torch.cuda.is_available():
            dtypes.extend([torch.float16, torch.bfloat16])
        else:
            dtypes.append(torch.bfloat16)

        eps = 1e-5
        for (hidden_size, n_groups), shape in zip(group_configs, shapes):
            group_size = hidden_size // n_groups
            for dtype in dtypes:
                norm = MambaRMSNormGated(hidden_size, eps=eps, group_size=group_size).to(dtype)
                with torch.no_grad():
                    norm.weight.copy_(torch.randn_like(norm.weight))

                for with_gate in [False, True]:
                    x = torch.randn(*shape, dtype=dtype)
                    gate = torch.randn(*shape, dtype=dtype) if with_gate else None

                    out = norm(x, gate=gate)
                    expected = self._reference_grouped_rmsnorm(
                        x, norm.weight, eps=eps, group_size=group_size, gate=gate
                    )

                    tol = 1e-5 if dtype == torch.float32 else 1e-2
                    torch.testing.assert_close(
                        out,
                        expected,
                        rtol=tol,
                        atol=tol,
                        msg=f"Failed for shape={shape}, dtype={dtype}, with_gate={with_gate}, n_groups={n_groups}",
                    )

    def test_single_group_preserves_legacy_behavior(self):
        torch.manual_seed(42)
        hidden_size = 128
        eps = 1e-5
        x = torch.randn(2, 4, hidden_size, dtype=torch.float32)
        gate = torch.randn(2, 4, hidden_size, dtype=torch.float32)

        # Legacy behavior: group_size=None or group_size=hidden_size
        norm_default = MambaRMSNormGated(hidden_size, eps=eps)
        norm_explicit_full = MambaRMSNormGated(hidden_size, eps=eps, group_size=hidden_size)
        norm_explicit_none = MambaRMSNormGated(hidden_size, eps=eps, group_size=None)

        with torch.no_grad():
            w = torch.randn(hidden_size)
            norm_default.weight.copy_(w)
            norm_explicit_full.weight.copy_(w)
            norm_explicit_none.weight.copy_(w)

        # Reference un-grouped output
        h = x * torch.nn.functional.silu(gate)
        legacy_ref = w * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))

        out_default = norm_default(x, gate=gate)
        out_explicit_full = norm_explicit_full(x, gate=gate)
        out_explicit_none = norm_explicit_none(x, gate=gate)

        # Must be bitwise identical
        torch.testing.assert_close(out_default, legacy_ref, rtol=0, atol=0)
        torch.testing.assert_close(out_explicit_full, legacy_ref, rtol=0, atol=0)
        torch.testing.assert_close(out_explicit_none, legacy_ref, rtol=0, atol=0)

    def test_cross_group_gradient_isolation(self):
        """
        Verify that in grouped normalization, gradients of outputs in group A
        with respect to inputs in group B (A != B) are strictly zero.
        """
        hidden_size = 128
        n_groups = 4
        group_size = hidden_size // n_groups  # 32

        norm = MambaRMSNormGated(hidden_size, eps=1e-5, group_size=group_size)

        x = torch.randn(2, 4, hidden_size, dtype=torch.float32, requires_grad=True)
        y = norm(x)

        # Loss depends solely on group 0: indices [0:group_size]
        loss_group_0 = y[..., :group_size].sum()
        loss_group_0.backward()

        self.assertIsNotNone(x.grad)
        # Group 0 gradient must be non-zero
        self.assertTrue((x.grad[..., :group_size] != 0).any().item())
        # Gradients for all other groups must be identically zero (no coupling)
        torch.testing.assert_close(
            x.grad[..., group_size:],
            torch.zeros_like(x.grad[..., group_size:]),
            rtol=0,
            atol=0,
        )

        # Compare with un-grouped normalization where cross-group coupling DOES occur
        norm_ungrouped = MambaRMSNormGated(hidden_size, eps=1e-5, group_size=hidden_size)
        x_ungrouped = torch.randn(2, 4, hidden_size, dtype=torch.float32, requires_grad=True)
        y_ungrouped = norm_ungrouped(x_ungrouped)
        loss_ungrouped = y_ungrouped[..., :group_size].sum()
        loss_ungrouped.backward()

        # In un-grouped normalization, other groups receive non-zero gradient due to global mean
        self.assertTrue((x_ungrouped.grad[..., group_size:].abs() > 1e-7).any().item())

    def test_mamba2_mixer_non_fused_group_size_cpu(self):
        """
        Verify that Mamba2Mixer instantiates its norm with group_size = intermediate_size // n_groups
        and executes successfully through the non-fused path on CPU.
        """
        hidden_size = 64
        num_heads = 4
        head_dim = 32
        expand = 2
        n_groups = 2
        intermediate_size = expand * hidden_size  # 128
        expected_group_size = intermediate_size // n_groups  # 64

        config = Mamba2Config(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            expand=expand,
            n_groups=n_groups,
        )

        mixer = Mamba2Mixer(config, layer_idx=0)

        # 1. Check norm attributes
        self.assertEqual(mixer.norm.group_size, expected_group_size)
        self.assertEqual(mixer.norm.weight.shape, (intermediate_size,))

        # 2. Run non-fused forward pass in eval mode on CPU
        mixer.eval()
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
        with torch.no_grad():
            out = mixer(x)

        self.assertEqual(out.shape, (batch_size, seq_len, hidden_size))
        self.assertFalse(torch.isnan(out).any().item())
        self.assertFalse(torch.isinf(out).any().item())

        # 3. Verify single-group mixer config has group_size == intermediate_size
        config_single = Mamba2Config(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            expand=expand,
            n_groups=1,
        )
        mixer_single = Mamba2Mixer(config_single, layer_idx=0)
        self.assertEqual(mixer_single.norm.group_size, intermediate_size)
