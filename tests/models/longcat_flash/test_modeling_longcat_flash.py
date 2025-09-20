# Copyright 2025 Meituan and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch LongcatFlash model."""

import copy
import tempfile
import unittest

from parameterized import parameterized
from pytest import mark

from transformers import LongcatFlashConfig, is_torch_available, set_seed
from transformers.testing_utils import (
    require_bitsandbytes,
    require_flash_attn,
    require_large_cpu_ram,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, LongcatFlashForCausalLM, LongcatFlashModel


class LongcatFlashModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = LongcatFlashConfig
        base_model_class = LongcatFlashModel
        causal_lm_class = LongcatFlashForCausalLM

    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=144,
        ffn_hidden_size=288,
        expert_ffn_hidden_size=48,
        num_layers=1,  # We have `self.num_hidden_layers = 2 * num_layers` in the body. See `LongcatFlashConfig`.
        num_attention_heads=8,
        num_key_value_heads=8,
        kv_lora_rank=16,
        q_lora_rank=48,
        qk_rope_head_dim=4,
        v_head_dim=8,
        qk_nope_head_dim=8,
        head_dim=4,
        n_routed_experts=4,
        zero_expert_num=2,
        moe_topk=2,
        routed_scaling_factor=1.0,
        hidden_act="silu",
        max_position_embeddings=128,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.expert_ffn_hidden_size = expert_ffn_hidden_size
        self.num_layers = num_layers
        self.num_hidden_layers = 2 * num_layers  # for compatibility
        self.expected_num_hidden_layers = 2  # embedding + 2 layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.head_dim = head_dim
        self.n_routed_experts = n_routed_experts
        self.zero_expert_num = zero_expert_num
        self.moe_topk = moe_topk
        self.routed_scaling_factor = routed_scaling_factor
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices

    def get_config(self):
        return LongcatFlashConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            expert_ffn_hidden_size=self.expert_ffn_hidden_size,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            kv_lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim,
            head_dim=self.head_dim,
            n_routed_experts=self.n_routed_experts,
            zero_expert_num=self.zero_expert_num,
            moe_topk=self.moe_topk,
            routed_scaling_factor=self.routed_scaling_factor,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LongcatFlashModel(config=config)
        model.to(torch_device)
        model.eval()
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
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        model = LongcatFlashForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones(self.batch_size, self.seq_length)).to(torch_device)

        token_type_ids = None

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels = config_and_inputs

        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class LongcatFlashModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (LongcatFlashModel, LongcatFlashForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (LongcatFlashForCausalLM,) if is_torch_available() else ()

    pipeline_model_mapping = (
        {
            "feature-extraction": LongcatFlashModel,
            "text-generation": LongcatFlashForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    model_split_percents = [0.5, 0.8]

    test_headmasking = False
    test_pruning = False

    model_tester_class = LongcatFlashModelTester

    def setUp(self):
        self.model_tester = LongcatFlashModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LongcatFlashConfig, hidden_size=37, num_attention_heads=3)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @unittest.skip("LongcatFlash buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip("LongcatFlash buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_to_base(self):
        pass

    def test_past_key_values_format(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        batch_size, seq_length = inputs["input_ids"].shape

        k_embed_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        v_embed_dim = config.v_head_dim

        self_attention_keys_shape = (batch_size, config.num_key_value_heads, seq_length, k_embed_dim)
        self_attention_values_shape = (batch_size, config.num_key_value_heads, seq_length, v_embed_dim)

        num_hidden_layers = config.num_hidden_layers
        all_cache_shapes = [[self_attention_keys_shape, self_attention_values_shape] for _ in range(num_hidden_layers)]

        super().test_past_key_values_format(custom_all_cache_shapes=all_cache_shapes)

    def _check_past_key_values_for_generate(self, batch_size, decoder_past_key_values, cache_length, config):
        from transformers.cache_utils import Cache

        self.assertIsInstance(decoder_past_key_values, (tuple, Cache))

        k_embed_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        v_embed_dim = config.v_head_dim

        expected_key_shape = (batch_size, config.num_key_value_heads, cache_length, k_embed_dim)
        expected_value_shape = (batch_size, config.num_key_value_heads, cache_length, v_embed_dim)

        if isinstance(decoder_past_key_values, Cache):
            for layer_idx in range(config.num_hidden_layers):
                self.assertEqual(decoder_past_key_values.layers[layer_idx].keys.shape, expected_key_shape)
                self.assertEqual(decoder_past_key_values.layers[layer_idx].values.shape, expected_value_shape)
        else:
            for layer_past in decoder_past_key_values:
                self.assertEqual(layer_past[0].shape, expected_key_shape)
                self.assertEqual(layer_past[1].shape, expected_value_shape)

    @unittest.skip("MoE experts may not receive gradients with small test data")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip("MoE experts may not receive gradients with small test data")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip("MoE experts may not receive gradients with small test data")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip("LongcatFlash router uses weight.type() directly in forward which prevents offloading")
    def test_cpu_offload(self):
        pass

    @unittest.skip("LongcatFlash router uses weight.type() directly in forward which prevents offloading")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip("LongcatFlash router uses weight.type() directly in forward which prevents offloading")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip("Most probably because of the MOE, the moe and router does not ignore padding tokens")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @staticmethod
    def _prepare_config_headdim(config, requested_dim):
        # there's specific head dims due to lora compressions in longcat
        config = copy.deepcopy(config)
        config.attention_dropout = 0

        if requested_dim > config.qk_rope_head_dim:
            config.qk_rope_head_dim = requested_dim
            config.qk_nope_head_dim = max(config.qk_nope_head_dim, requested_dim)
            config.v_head_dim = max(config.v_head_dim, requested_dim)
            config.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
            config.head_dim = requested_dim
            config.q_lora_rank = max(config.q_lora_rank, requested_dim * 4)
            config.kv_lora_rank = max(config.kv_lora_rank, requested_dim * 2)
            config.hidden_size = max(config.hidden_size, config.num_attention_heads * requested_dim)

        return config

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)
        original_model = self.model_tester_class.base_model_class(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = self.model_tester_class.base_model_class(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        if scaling_type == "dynamic":
            torch.testing.assert_close(original_short_output, scaled_short_output, rtol=1e-5, atol=1e-5)
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    @require_flash_attn
    @require_torch_gpu
    @require_bitsandbytes
    @mark.flash_attn_test
    @slow
    def test_flash_attn_2_fp32_ln(self):
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_generative_model_classes:  # TODO: this test should run on all classes instead
            if not model_class._supports_flash_attn:
                self.skipTest(f"{model_class.__name__} does not support Flash Attention 2")
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_input = inputs_dict[model.main_input_name]
                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
                batch_size = dummy_attention_mask.shape[0]

                is_padding_right = dummy_attention_mask[:, -1].sum().item() != batch_size

                # To avoid errors with padding_side=="right"
                if is_padding_right:
                    dummy_attention_mask = torch.ones_like(dummy_input)

                model = model_class.from_pretrained(
                    tmpdirname,
                    dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",  # small change to ensure device placement
                )

                # no upcasting at all

                if model.config.is_encoder_decoder:
                    dummy_decoder_input_ids = inputs_dict["decoder_input_ids"]
                    dummy_decoder_attention_mask = inputs_dict["decoder_attention_mask"]

                    _ = model(dummy_input, decoder_input_ids=dummy_decoder_input_ids)
                    # with attention mask
                    _ = model(
                        dummy_input,
                        attention_mask=dummy_attention_mask,
                        decoder_input_ids=dummy_decoder_input_ids,
                        decoder_attention_mask=dummy_decoder_attention_mask,
                    )
                else:
                    _ = model(dummy_input)
                    # with attention mask
                    _ = model(dummy_input, attention_mask=dummy_attention_mask)


@slow
class LongcatFlashIntegrationTest(unittest.TestCase):
    short_model_id = "hf-internal-testing/LongCat-ShortCat"
    # This is a cut-down model that matches part of the early logits of the larger one
    # Only a couple experts + layers
    # But if it fails, it means the larger model might have issues as well
    model_id = "meituan-longcat/LongCat-Flash-Chat"

    @slow
    def test_shortcat_generation(self):
        self.model = LongcatFlashForCausalLM.from_pretrained(
            self.short_model_id,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        self.model.generation_config.bos_token_id = 1
        self.model.generation_config.pad_token_id = 3
        self.model.generation_config.eos_token_id = 2
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        chat = [{"role": "user", "content": "Paris is..."}]
        inputs = self.tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=10, do_sample=False)

        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        expected_output = "[Round 0] USER:Paris is... ASSISTANT: dig年车龄juanaheast稍achaotingupebarebones"

        self.assertEqual(response, expected_output)

    @slow
    @require_large_cpu_ram
    def test_longcat_generation_cpu(self):
        # takes absolutely forever and a lot RAM, but allows to test the output in the CI
        model = LongcatFlashForCausalLM.from_pretrained(self.model_id, device_map="cpu", dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        chat = [{"role": "user", "content": "Paris is..."}]
        inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=10, do_sample=False)

        response = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        expected_output = "[Round 0] USER:Paris is... ASSISTANT:Paris is... a city of timeless charm, where"

        self.assertEqual(response, expected_output)
