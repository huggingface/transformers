# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch DeepSeekV2 model."""

import unittest

from parameterized import parameterized

from transformers import DeepseekV2Config, is_torch_available, set_seed
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        DeepseekV2ForCausalLM,
        DeepseekV2ForSequenceClassification,
        DeepseekV2Model,
    )
    from transformers.models.deepseek_v2 import DeepseekV2RotaryEmbedding


class DeepseekV2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
        moe_intermediate_size=12,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        qk_nope_head_dim=32,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=2,
        norm_topk_prob=True,
        aux_loss_alpha=0.001,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope
        self.moe_intermediate_size = moe_intermediate_size
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return DeepseekV2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            n_shared_experts=self.n_shared_experts,
            n_routed_experts=self.n_routed_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            kv_lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim,
            n_group=self.n_group,
            topk_group=self.topk_group,
            num_experts_per_tok=self.num_experts_per_tok,
            first_k_dense_replace=self.first_k_dense_replace,
            norm_topk_prob=self.norm_topk_prob,
            aux_loss_alpha=self.aux_loss_alpha,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            use_cache=True,
            pad_token_id=self.pad_token_id,
            attention_dropout=self.attention_probs_dropout_prob,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = DeepseekV2Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = DeepseekV2Model(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = DeepseekV2ForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = DeepseekV2ForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class DeepseekV2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            DeepseekV2ForCausalLM,
            DeepseekV2ForSequenceClassification,
            DeepseekV2Model,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": DeepseekV2Model,
            "text-classification": DeepseekV2ForSequenceClassification,
            "text-generation": DeepseekV2ForCausalLM,
            "zero-shot": DeepseekV2ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = DeepseekV2ForCausalLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = DeepseekV2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DeepseekV2Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("DeepseekV2 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV2 has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV2 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("DeepseekV2 has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("DeepseekV2 has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("DeepseekV2 has HybridCache and doesn't support low_memory generation")
    def test_beam_search_low_memory(self):
        pass

    @unittest.skip("DeepseekV2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("DeepseekV2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("DeepseekV2 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip(
        "DeepseekV2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip(
        "DeepseekV2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip(
        "DeepseekV2 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip("DeepseekV2's eager attn/sdpa attn outputs are expected to be different")
    def test_sdpa_equivalence(self):
        pass

    @unittest.skip("DeepseekV2 uses MLA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("DeepseekV2 uses MLA so it is not compatible with the standard cache format")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("DeepseekV2 uses MLA so it is not compatible with the standard cache format")
    def test_generate_compile_model_forward(self):
        pass

    @unittest.skip("DeepseekV2 uses MLA so it is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="DeepseekV2 uses MLA on all models so the KV cache is a non standard format")
    def test_past_key_values_format(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="DeepseekV2 does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_deepseek_v2_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = DeepseekV2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_deepseek_v2_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = DeepseekV2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_deepseek_v2_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = DeepseekV2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    @unittest.skip(reason="DeepseekV2 buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @parameterized.expand([("linear",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = DeepseekV2Model(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = DeepseekV2Model(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            torch.testing.assert_close(original_short_output, scaled_short_output, rtol=1e-5, atol=1e-5)
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    def test_model_rope_scaling(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(1, dtype=torch.float32, device=torch_device)  # used exlusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        original_rope = DeepseekV2RotaryEmbedding(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        linear_scaling_rope = DeepseekV2RotaryEmbedding(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short)
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
        ntk_scaling_rope = DeepseekV2RotaryEmbedding(config=config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short)
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        config.rope_scaling = {"type": "yarn", "factor": scaling_factor}
        yarn_scaling_rope = DeepseekV2RotaryEmbedding(config=config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short)
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(yarn_cos_short, yarn_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(yarn_sin_short, yarn_sin_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_short, original_cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_long, original_sin_long)

    def test_model_loading_old_rope_configs(self):
        def _reinitialize_config(base_config, new_kwargs):
            # Reinitialize the config with the new kwargs, forcing the config to go through its __init__ validation
            # steps.
            base_config_dict = base_config.to_dict()
            new_config = DeepseekV2Config.from_dict(config_dict={**base_config_dict, **new_kwargs})
            return new_config

        # from untouched config -> ✅
        base_config, model_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        original_model = DeepseekV2ForCausalLM(base_config).to(torch_device)
        original_model(**model_inputs)

        # from a config with the expected rope configuration -> ✅
        config = _reinitialize_config(base_config, {"rope_scaling": {"rope_type": "linear", "factor": 10.0}})
        original_model = DeepseekV2ForCausalLM(config).to(torch_device)
        original_model(**model_inputs)

        # from a config with the old rope configuration ('type' instead of 'rope_type')  -> ✅ we gracefully handle BC
        config = _reinitialize_config(base_config, {"rope_scaling": {"type": "linear", "factor": 10.0}})
        original_model = DeepseekV2ForCausalLM(config).to(torch_device)
        original_model(**model_inputs)

        # from a config with both 'type' and 'rope_type'  -> ✅ they can coexist (and both are present in the config)
        config = _reinitialize_config(
            base_config, {"rope_scaling": {"type": "linear", "rope_type": "linear", "factor": 10.0}}
        )
        self.assertTrue(config.rope_scaling["type"] == "linear")
        self.assertTrue(config.rope_scaling["rope_type"] == "linear")
        original_model = DeepseekV2ForCausalLM(config).to(torch_device)
        original_model(**model_inputs)

        # from a config with parameters in a bad range ('factor' should be >= 1.0) -> ⚠️ throws a warning
        with self.assertLogs("transformers.modeling_rope_utils", level="WARNING") as logs:
            config = _reinitialize_config(base_config, {"rope_scaling": {"rope_type": "linear", "factor": -999.0}})
            original_model = DeepseekV2ForCausalLM(config).to(torch_device)
            original_model(**model_inputs)
            self.assertEqual(len(logs.output), 1)
            self.assertIn("factor field", logs.output[0])

        # from a config with unknown parameters ('foo' isn't a rope option) -> ⚠️ throws a warning
        with self.assertLogs("transformers.modeling_rope_utils", level="WARNING") as logs:
            config = _reinitialize_config(
                base_config, {"rope_scaling": {"rope_type": "linear", "factor": 10.0, "foo": "bar"}}
            )
            original_model = DeepseekV2ForCausalLM(config).to(torch_device)
            original_model(**model_inputs)
            self.assertEqual(len(logs.output), 1)
            self.assertIn("Unrecognized keys", logs.output[0])

        # from a config with specific rope type but missing one of its mandatory parameters -> ❌ throws exception
        with self.assertRaises(KeyError):
            config = _reinitialize_config(base_config, {"rope_scaling": {"rope_type": "linear"}})  # missing "factor"


@require_torch_accelerator
class DeepseekV2IntegrationTest(unittest.TestCase):
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    @slow
    @require_read_token
    def test_deepseek_v2_lite_hard(self):
        """
        An integration test for DeepseekV2
        """

        EXPECTED_TEXT = """An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

## Attention Function

The attention function is a function that takes a query and a set of key-value pairs as input, and outputs a weighted sum of the values, where the weights are determined by the query and the keys.

The attention function is used in many applications, such as machine translation, image captioning, and question answering.

## Attention Function in Machine Translation

In machine translation, the attention function is used to determine the most relevant parts of the source sentence to be translated into the target language.

The attention function is used to determine the weights of the source sentence words, and"""

        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", device_map="auto", torch_dtype=torch.bfloat16
        )
        input_text = [
            "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors."
        ]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)

    @slow
    @require_read_token
    def test_model_lite_logits_bf16(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))
        # Expected mean on dim = -1

        # fmt: off
        EXPECTED_MEAN = {
            7: torch.tensor([[-6.1743, -5.0111, -4.0070, -2.5044, -2.1331, -2.4444, -3.7115, -3.6149]]),
            8: torch.tensor([[-6.1890, -4.9891, -3.9917, -2.4924, -2.1125, -2.4480, -3.7443, -3.5946]])
        }

        self.assertTrue(
            torch.allclose(
                EXPECTED_MEAN[self.cuda_compute_capability_major_version].to(torch_device),
                out.logits.float().mean(-1),
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICE = {
            7: torch.tensor([[-1.2031, -0.7344, -0.0762, -2.9062,  1.2656, -2.6094, -0.7227, -2.9062, -2.5312, -0.5430, -0.2949, -1.7734, -2.1562, -0.7969, -3.8594]]),
            8: torch.tensor([[-1.1875, -0.7383, -0.0601, -2.8594,  1.2578, -2.6094, -0.7383, -2.9062, -2.5469, -0.5469, -0.3125, -1.7734, -2.1719, -0.8125, -3.8438]])
        }

        # fmt: on
        self.assertTrue(
            torch.allclose(
                EXPECTED_SLICE[self.cuda_compute_capability_major_version].to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @slow
    @require_read_token
    def test_model_lite_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", device_map="auto", torch_dtype=torch.float16
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        EXPECTED_MEAN = {
            7: torch.tensor([[-6.6420, -4.1227, -4.9809, -3.2041, 0.8261, -3.0052, 1.2957, -3.3648]]),
            8: torch.tensor([[-6.1736, -5.0128, -4.0018, -2.5014, -2.1399, -2.4453, -3.7112, -3.6169]])
        }

        self.assertTrue(
            torch.allclose(
                EXPECTED_MEAN[self.cuda_compute_capability_major_version].to(torch_device),
                out.logits.float().mean(-1),
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICE = {
            7: torch.tensor([-1.0938, -0.5713, -0.2632, -2.9434,  1.2783, -2.6465, -0.6992, -2.6875, -2.3086, -0.5396, -0.2993, -1.5439, -2.2500, -0.4854, -3.7539]),
            8: torch.tensor([-1.0898, -0.5703, -0.2627, -2.9414,  1.2725, -2.6504, -0.7007, -2.6836, -2.3125, -0.5415, -0.3003, -1.5420, -2.2480, -0.4829, -3.7520])
        }
        # fmt: on

        self.assertTrue(
            torch.allclose(
                EXPECTED_SLICE[self.cuda_compute_capability_major_version].to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @slow
    @require_torch_accelerator
    @require_read_token
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        # Note on `EXPECTED_TEXT_COMPLETION`'s diff: the current value matches the original test if the original test
        # was changed to have a cache of 53 tokens (as opposed to 4096), on Ampere GPUs.

        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that  “the laws of physics are the same for all observers in uniform motion relative to one another.”\n\nThe theory of relativity is a theory of space, time, and gravity. It is",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, my hot dogs, my hamburgers, my french fries, my chicken nuggets, my pizza, my grilled cheese, my grilled",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", pad_token="</s>", padding_side="right"
        )
        model = DeepseekV2ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V2-Lite", device_map=torch_device, torch_dtype=torch.float16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        # Static Cache + compile (`generate()` internally compiles each decoding step when static cache is used)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)
