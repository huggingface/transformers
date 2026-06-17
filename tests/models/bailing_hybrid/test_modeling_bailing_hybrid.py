# Copyright 2025 InclusionAI and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch BailingHybrid model."""

import unittest

from parameterized import parameterized

from transformers import BailingHybridConfig, is_torch_available
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
        BailingHybridForCausalLM,
        BailingHybridForSequenceClassification,
        BailingHybridForTokenClassification,
        BailingHybridModel,
    )


class BailingHybridModelTester:
    if is_torch_available():
        causal_lm_class = BailingHybridForCausalLM

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
        intermediate_size=32,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_shared_experts=1,
        num_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=4,
        v_head_dim=8,
        qk_nope_head_dim=4,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        layer_group_size=4,
        group_norm_size=2,
        num_kv_heads_for_linear_attn=4,
        linear_silu=False,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        attention_probs_dropout_prob=0.0,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
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
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_shared_experts = num_shared_experts
        self.num_experts = num_experts
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
        self.layer_group_size = layer_group_size
        self.group_norm_size = group_norm_size
        self.num_kv_heads_for_linear_attn = num_kv_heads_for_linear_attn
        self.linear_silu = linear_silu
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

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
        return BailingHybridConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            moe_shared_expert_intermediate_size=self.moe_shared_expert_intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            num_shared_experts=self.num_shared_experts,
            num_experts=self.num_experts,
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
            layer_group_size=self.layer_group_size,
            group_norm_size=self.group_norm_size,
            num_kv_heads_for_linear_attn=self.num_kv_heads_for_linear_attn,
            linear_silu=self.linear_silu,
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
        model = BailingHybridModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

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
class BailingHybridModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            BailingHybridModel,
            BailingHybridForCausalLM,
            BailingHybridForSequenceClassification,
            BailingHybridForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (BailingHybridForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": BailingHybridModel,
            "text-classification": BailingHybridForSequenceClassification,
            "token-classification": BailingHybridForTokenClassification,
            "text-generation": BailingHybridForCausalLM,
            "zero-shot": BailingHybridForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    model_split_percents = [0.5, 0.7, 0.8]

    _torch_compile_train_cls = BailingHybridForCausalLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = BailingHybridModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BailingHybridConfig, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_attention_outputs(self):
        """Needs override as BailingHybrid alternates between MLA and linear attention layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        config._attn_implementation = "eager"

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            out_len = len(outputs)

            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("BailingHybrid is not compatible with assisted decoding due to hybrid cache")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("BailingHybrid is not compatible with assisted decoding due to hybrid cache")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("BailingHybrid is not compatible with assisted decoding due to hybrid cache")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("BailingHybrid uses MLA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("BailingHybrid uses MLA so it is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip("BailingHybrid uses MLA so beam search is not compatible with the standard cache format")
    def test_beam_sample_generate(self):
        pass

    @unittest.skip("BailingHybrid uses MLA so beam search is not compatible with the standard cache format")
    def test_beam_search_generate(self):
        pass

    @unittest.skip("BailingHybrid uses MLA so beam search is not compatible with the standard cache format")
    def test_beam_sample_generate_dict_output(self):
        pass

    @unittest.skip("BailingHybrid uses MLA so beam search is not compatible with the standard cache format")
    def test_beam_search_generate_dict_output(self):
        pass

    @unittest.skip("BailingHybrid uses MLA so it is not compatible with continue from past_key_values")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("BailingHybrid's linear attention has no conv1d, so conv_states are None")
    def test_past_key_values_format(self):
        pass

    @unittest.skip("BailingHybrid uses MLA so inputs_embeds generation is not compatible with cache format")
    def test_generate_from_inputs_embeds_0_greedy(self):
        pass

    @unittest.skip("BailingHybrid uses MLA so inputs_embeds generation is not compatible with cache format")
    def test_generate_from_inputs_embeds_1_beam(self):
        pass

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_bailing_hybrid_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.num_labels)
        model = BailingHybridForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))
