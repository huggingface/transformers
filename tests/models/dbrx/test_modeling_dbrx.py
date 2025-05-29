# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch DBRX model."""

import unittest

from transformers import DbrxConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import DbrxForCausalLM, DbrxModel


@require_torch
class DbrxModelTester(CausalLMModelTester):
    config_class = DbrxConfig
    if is_torch_available():
        base_model_class = DbrxModel
        causal_lm_class = DbrxForCausalLM
        sequence_classification_class = None
        token_classification_class = None

    def __init__(
        self,
        parent,
        hidden_size=32,
        ffn_hidden_size=32,
        num_attention_heads=4,
        kv_n_heads=4,
        num_hidden_layers=5,
        max_position_embeddings=512,
        type_vocab_size=16,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        use_cache=True,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        clip_qkv=8,
        rope_theta=500000,
        attn_config_model_type="",
        emb_pdrop=0.0,
        moe_jitter_eps=0,
        moe_loss_weight=0.05,
        moe_num_experts=16,
        moe_top_k=4,
        ffn_config_model_type="",
        ffn_act_fn_name="gelu",
        initializer_range=0.02,
        output_router_logits=False,
        resid_pdrop=0.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        vocab_size=99,
        is_decoder=True,
        pad_token_id=0,
    ):
        # DBRX-specific parameters
        self.clip_qkv = clip_qkv
        self.kv_n_heads = kv_n_heads
        self.rope_theta = rope_theta
        self.attn_config_model_type = attn_config_model_type
        self.ffn_hidden_size = ffn_hidden_size
        self.moe_jitter_eps = moe_jitter_eps
        self.moe_loss_weight = moe_loss_weight
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.ffn_config_model_type = ffn_config_model_type
        self.ffn_act_fn_name = ffn_act_fn_name
        self.emb_pdrop = emb_pdrop
        self.output_router_logits = output_router_logits
        self.resid_pdrop = resid_pdrop
        self.tie_word_embeddings = tie_word_embeddings
        self.torch_dtype = torch_dtype
        self.use_cache = use_cache
        
        # Call parent init
        super().__init__(
            parent=parent,
            batch_size=batch_size,
            seq_length=seq_length,
            is_training=is_training,
            use_input_mask=use_input_mask,
            use_token_type_ids=use_token_type_ids,
            use_labels=use_labels,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=kv_n_heads,
            intermediate_size=ffn_hidden_size,
            hidden_act=ffn_act_fn_name,
            hidden_dropout_prob=resid_pdrop,
            attention_probs_dropout_prob=resid_pdrop,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            type_sequence_label_size=type_sequence_label_size,
            initializer_range=initializer_range,
            num_labels=num_labels,
            num_choices=num_choices,
            pad_token_id=pad_token_id,
            is_decoder=is_decoder,
            scope=scope,
            moe_intermediate_size=ffn_hidden_size,
            num_experts_per_tok=moe_top_k,
            num_experts=moe_num_experts,
        )

        # Make the dictionaries
        self.ffn_config = {
            "ffn_hidden_size": self.ffn_hidden_size,
            "moe_jitter_eps": self.moe_jitter_eps,
            "moe_loss_weight": self.moe_loss_weight,
            "moe_num_experts": self.moe_num_experts,
            "moe_top_k": self.moe_top_k,
            "model_type": self.ffn_config_model_type,
            "ffn_act_fn": {"name": self.ffn_act_fn_name},
        }
        self.attn_config = {
            "clip_qkv": self.clip_qkv,
            "kv_n_heads": self.kv_n_heads,
            "model_type": self.attn_config_model_type,
            "rope_theta": self.rope_theta,
        }

    def get_config(self):
        # Behind the scenes, `DbrxConfig` maps the parameters `hidden_size`, `num_hidden_layers`,
        # `num_attention_heads`, `max_position_embeddings` to the parameters `d_model`, `n_layers`,
        # `n_heads`, `max_seq_len` respectively. We use the first group of parameters because
        # other tests expect every model to have these parameters with these specific names.
        config = DbrxConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,  # mapped to `d_model`
            num_hidden_layers=self.num_hidden_layers,  # mapped to `n_layers`
            num_attention_heads=self.num_attention_heads,  # mapped to `n_heads`
            max_position_embeddings=self.max_position_embeddings,  # mapped to `max_seq_len`
            attn_config=self.attn_config,
            ffn_config=self.ffn_config,
            resid_pdrop=self.resid_pdrop,
            emb_pdrop=self.emb_pdrop,
            use_cache=self.use_cache,
            initializer_range=self.initializer_range,
            output_router_logits=self.output_router_logits,
            is_decoder=self.is_decoder,
            pad_token_id=self.pad_token_id,
        )
        return config


@require_torch
class DbrxModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (DbrxModel, DbrxForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": DbrxModel,
            "text-generation": DbrxForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = DbrxModelTester
    # DBRX's rotary embedding doesn't accept config parameter, so we disable RoPE tests
    rotary_embedding_layer = None

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "eitanturok/dbrx-tiny"
        model = DbrxModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @unittest.skip(reason="Dbrx models have weight tying disabled.")
    def test_tied_weights_keys(self):
        pass

    # Offload does not work with Dbrx models because of the forward of DbrxExperts where we chunk the experts.
    # The issue is that the offloaded weights of the mlp layer are still on meta device (w1_chunked, v1_chunked, w2_chunked)
    @unittest.skip(reason="Dbrx models do not work with offload")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Dbrx models do not work with offload")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(reason="Dbrx models do not work with offload")
    def test_disk_offload_bin(self):
        pass


@require_torch
class DbrxModelIntegrationTest(unittest.TestCase):
    @slow
    def test_tiny_model_logits(self):
        model = DbrxForCausalLM.from_pretrained("Rocketknight1/dbrx-tiny-random")
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]
        vocab_size = model.vocab_size

        expected_shape = torch.Size((1, 6, vocab_size))
        self.assertEqual(output.shape, expected_shape)

        expected_slice = torch.tensor(
            [
                [
                    [-1.6300e-04, 5.0118e-04, 2.5437e-04],
                    [2.0422e-05, 2.7210e-04, -1.5125e-04],
                    [-1.5105e-04, 4.6879e-04, 3.3309e-04],
                ]
            ]
        )
        torch.testing.assert_close(output[:, :3, :3], expected_slice, rtol=1e-4, atol=1e-4)
