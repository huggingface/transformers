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
"""Testing suite for the PyTorch BLT model."""

import unittest

import pytest
from parameterized import parameterized

from transformers import AutoTokenizer, is_torch_available, set_seed
from transformers.testing_utils import (
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_bf16,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    _test_eager_matches_sdpa_inference,
    ids_tensor,
)


if is_torch_available():
    import torch

    from transformers import BLTConfig, BLTForCausalLM, BLTModel
    from transformers.models.blt.modeling_blt import BLTRotaryEmbedding


class BLTModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = BLTConfig
        base_model_class = BLTModel
        causal_lm_class = BLTForCausalLM

    def __init__(
        self,
        parent,
        ignore_index=-100,
        seq_length=7,
        is_training=True,
    ):
        super().__init__(parent)
        self.parent = parent
        self.ignore_index = ignore_index
        self.seq_length = seq_length
        self.is_training = is_training
        self.batch_size = 3

        # Common parameters for all configs
        self.hidden_size = 16
        self.num_hidden_layers = 1
        self.num_attention_heads = 2
        self.num_key_value_heads = 2
        self.intermediate_size = 32
        self.hidden_act = "silu"
        self.max_position_embeddings = 32
        self.vocab_size = 32
        self.rope_theta = 500000.0
        self.rope_scaling = {"rope_type": "default"}
        self.norm_eps = 1e-5
        self.dropout = 0.0
        self.encoder_hash_byte_group_size = [2, 3]
        self.encoder_hash_byte_group_vocab = 64
        self.encoder_hash_byte_group_nb_functions = 1
        # Common parameters for all configs
        self.patcher_config = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "hidden_act": self.hidden_act,
            "norm_eps": self.norm_eps,
            "dropout": self.dropout,
        }

        self.encoder_config = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "hidden_act": self.hidden_act,
            "norm_eps": self.norm_eps,
            "dropout": self.dropout,
        }

        self.decoder_config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "hidden_size_global": self.hidden_size * 2,  # Must match global transformer output size
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "hidden_act": self.hidden_act,
            "norm_eps": self.norm_eps,
            "dropout": self.dropout,
        }

        self.global_config = {
            "hidden_size": self.hidden_size * 2,  # Double the hidden size for global transformer
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "hidden_act": self.hidden_act,
            "norm_eps": self.norm_eps,
            "dropout": self.dropout,
        }

    def get_config(self):
        config = BLTConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            patch_in_forward=False,  # Disable patching for tests
            patch_size=4,
            patching_mode="entropy",
            patching_threshold=1.335442066192627,
            patching_batch_size=1,
            max_patch_length=None,
            cross_attn_k=2,
            encoder_hash_byte_group_size=self.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.encoder_hash_byte_group_vocab,
            encoder_hash_byte_group_nb_functions=self.encoder_hash_byte_group_nb_functions,
            patcher_config=self.patcher_config,
            encoder_config=self.encoder_config,
            decoder_config=self.decoder_config,
            global_config=self.global_config,
            tie_word_embeddings=False,
        )

        config.num_attention_heads = config.decoder_config.num_attention_heads
        config.num_hidden_layers = config.decoder_config.num_hidden_layers
        config.hidden_size = config.decoder_config.hidden_size

        return config


@require_torch
class BLTModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            BLTModel,
            BLTForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": BLTModel,
            "text-generation": BLTForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False
    model_tester_class = BLTModelTester
    rotary_embedding_layer = BLTRotaryEmbedding  # Enables RoPE tests if set

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = BLTForCausalLM if is_torch_available() else None

    @pytest.mark.generate
    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    def test_generate_from_inputs_embeds(self, _, num_beams):
        """Skip this test for BLT as it has complex embedding computation that requires real token IDs for hash-based embeddings."""
        self.skipTest(
            "BLT requires real token IDs for its hash-based embedding computation, making inputs_embeds generation incompatible with identical outputs"
        )

    @pytest.mark.generate
    def test_inputs_embeds_matches_input_ids(self):
        """Skip this test for BLT as it has complex embedding computation that requires real token IDs for hash-based embeddings."""
        self.skipTest(
            "BLT requires real token IDs for its hash-based embedding computation, making inputs_embeds generation incompatible with identical outputs"
        )

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @require_torch_sdpa
    def test_eager_matches_sdpa_inference(
        self,
        name,
        torch_dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
    ):
        "We need to relax a bit the `atols` for fp32 here due to the altup projections"
        atols = {
            ("cpu", False, torch.float32): 2e-2,  # this was relaxed
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 2e-2,  # this was relaxed
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 2e-2,  # this was relaxed
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 2e-2,  # this was relaxed
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
        _test_eager_matches_sdpa_inference(
            self, name, torch_dtype, padding_side, use_attention_mask, output_attentions, enable_kernels, atols=atols
        )

    def test_torchscript_simple(self):
        """Skip torchscript test for BLT as it has complex patching logic that's not compatible."""
        self.skipTest("BLT has complex patching logic that's not compatible with torchscript")

    def test_torchscript_output_hidden_state(self):
        """Skip torchscript test for BLT as it has complex patching logic that's not compatible."""
        self.skipTest("BLT has complex patching logic that's not compatible with torchscript")

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        """Override rope scaling from config test to handle BLT's sub-config structure."""
        if self.rotary_embedding_layer is None:
            self.skipTest("Rotary embedding layer not set")
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = self.model_tester_class.base_model_class(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        # Propagate rope_scaling to sub-configs for BLT
        config.encoder_config.rope_scaling = config.rope_scaling
        config.decoder_config.rope_scaling = config.rope_scaling
        config.global_config.rope_scaling = config.rope_scaling
        config.patcher_config.rope_scaling = config.rope_scaling

        scaled_model = self.model_tester_class.base_model_class(config)
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

    @unittest.skip(reason="Training is not supported yet")
    def test_training_gradient_checkpointing(self):
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

    @unittest.skip(reason="Decoder cannot keep gradients")
    def test_flex_attention_with_grads():
        return


@require_torch_accelerator
class BLTIntegrationTest(unittest.TestCase):
    def tearDown(self):
        # TODO (joao): automatic compilation, i.e. compilation when `cache_implementation="static"` is used, leaves
        # some memory allocated in the cache, which means some object is not being released properly. This causes some
        # unoptimal memory usage, e.g. after certain tests a 7B model in FP16 no longer fits in a 24GB GPU.
        # Investigate the root cause.
        cleanup(torch_device, gc_collect=False)

    @slow
    @require_read_token
    def test_blt(self):
        NUM_TOKENS_TO_GENERATE = 200
        EXPECTED_TEXT = "my name is alex and i am a student at the university of michigan. i am a senior majoring in computer science and minoring in mathematics. i am also a member of the michigan math club and the michigan computer s"

        prompt = "my name is"

        model = BLTForCausalLM.from_pretrained(
            "itazap/blt-1b",
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained("itazap/blt-1b")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)

        output_text = tokenizer.decode(generated_ids[0])
        self.assertEqual(output_text, EXPECTED_TEXT)

    @slow
    @require_read_token
    def test_model_logits(self):
        EXPECTED_OUTPUT = torch.tensor(
            [
                [
                    -10.4948,
                    -10.7065,
                    -6.1813,
                    -10.5545,
                    -10.3428,
                    -9.1493,
                    -8.4937,
                    -8.6382,
                    -9.2159,
                    -9.5907,
                    -9.3679,
                    -8.4184,
                    -9.0655,
                    -3.4436,
                    2.9616,
                    -10.3157,
                    -6.3723,
                    -6.0133,
                    -9.7100,
                    -9.2128,
                    -8.8064,
                    -9.8179,
                    -9.7516,
                    -9.4681,
                    -9.7715,
                    -9.4897,
                    -9.0491,
                    -9.8098,
                    -9.4648,
                    -9.3294,
                ],
                [
                    -13.3010,
                    -13.1910,
                    -5.7230,
                    -13.2895,
                    -13.4864,
                    -8.7140,
                    -7.0275,
                    -7.0182,
                    -10.1362,
                    -10.3762,
                    -9.9086,
                    -7.8049,
                    -8.8660,
                    -5.2711,
                    -3.5778,
                    -12.5346,
                    -9.1609,
                    -6.7925,
                    -10.3717,
                    -9.2650,
                    -10.6393,
                    -11.4807,
                    -11.2128,
                    -10.9615,
                    -10.5806,
                    -10.8873,
                    -11.0651,
                    -11.3471,
                    -10.5437,
                    -9.9688,
                ],
            ]
        ).to(torch_device)

        input_ids = [1, 42, 21, 12, 43, 23, 1, 4]

        model = BLTForCausalLM.from_pretrained("itazap/blt-1b", device_map="auto")

        with torch.no_grad():
            output = model(torch.tensor([input_ids]).to(torch_device))[0]

        torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2, :30], rtol=1e-4, atol=1e-4)

    @slow
    @require_read_token
    @require_torch_bf16
    def test_model_bf16(self):
        """Test BLT model with bfloat16 precision."""
        NUM_TOKENS_TO_GENERATE = 200
        EXPECTED_TEXT = "my name is alex and i am a student at the university of michigan in the college of arts and sciences. i am a senior majoring in computer science and minoring in mathematics. i am also a member of the michigan m"

        prompt = "my name is"

        model = BLTForCausalLM.from_pretrained("itazap/blt-1b", device_map="auto", torch_dtype=torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained("itazap/blt-1b")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)

        output_text = tokenizer.decode(generated_ids[0])
        self.assertEqual(output_text, EXPECTED_TEXT)

    @slow
    @require_read_token
    @require_torch_bf16
    def test_model_logits_bf16(self):
        """Test BLT model logits with bfloat16 precision."""
        EXPECTED_OUTPUT = torch.tensor(
            [
                [
                    -10.5000,
                    -10.6875,
                    -6.1875,
                    -10.5000,
                    -10.3125,
                    -9.1250,
                    -8.5000,
                    -8.6250,
                    -9.1875,
                    -9.5625,
                    -9.3750,
                    -8.5000,
                    -9.0625,
                    -3.4062,
                    2.9688,
                    -10.3125,
                    -6.4062,
                    -5.9688,
                    -9.6875,
                    -9.1875,
                    -8.8125,
                    -9.8125,
                    -9.7500,
                    -9.4375,
                    -9.7500,
                    -9.4375,
                    -9.0000,
                    -9.7500,
                    -9.4375,
                    -9.3125,
                ],
                [
                    -13.3125,
                    -13.1875,
                    -5.6875,
                    -13.2500,
                    -13.5000,
                    -8.7500,
                    -7.0312,
                    -7.0000,
                    -10.1250,
                    -10.3750,
                    -9.8750,
                    -7.7812,
                    -8.8750,
                    -5.2500,
                    -3.5312,
                    -12.5625,
                    -9.1875,
                    -6.7812,
                    -10.3750,
                    -9.2500,
                    -10.6250,
                    -11.5000,
                    -11.1875,
                    -10.9375,
                    -10.5625,
                    -10.8750,
                    -11.0625,
                    -11.3750,
                    -10.5000,
                    -10.0000,
                ],
            ],
            dtype=torch.bfloat16,
        ).to(torch_device)

        input_ids = [1, 42, 21, 12, 43, 23, 1, 4]

        model = BLTForCausalLM.from_pretrained("itazap/blt-1b", device_map="auto", torch_dtype=torch.bfloat16)

        with torch.no_grad():
            output = model(torch.tensor([input_ids]).to(torch_device))[0]

        # print(output[0, :2, :30])

        torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2, :30], rtol=1e-3, atol=1e-3)

    @slow
    @require_read_token
    def test_model_eager(self):
        """Test BLT model with bfloat16 precision using eager attention implementation."""
        NUM_TOKENS_TO_GENERATE = 200
        EXPECTED_TEXT = "my name is alex and i am a student at the university of michigan. i am a senior majoring in computer science and minoring in mathematics. i am also a member of the michigan math club and the michigan computer s"

        prompt = "my name is"

        model = BLTForCausalLM.from_pretrained("itazap/blt-1b", device_map="auto", attn_implementation="eager")

        tokenizer = AutoTokenizer.from_pretrained("itazap/blt-1b")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)

        output_text = tokenizer.decode(generated_ids[0])
        self.assertEqual(output_text, EXPECTED_TEXT)

    @slow
    @require_read_token
    @require_torch_bf16
    def test_model_bf16_static_cache(self):
        """Test BLT model with bfloat16 precision and static cache."""
        NUM_TOKENS_TO_GENERATE = 200
        EXPECTED_TEXT = "my name is alex and i am a student at the university of michigan in the college of arts and sciences. i am a senior majoring in computer science and minoring in mathematics. i am also a member of the michigan m"

        prompt = "my name is"

        model = BLTForCausalLM.from_pretrained("itazap/blt-1b", device_map="auto", torch_dtype=torch.bfloat16)

        model.generation_config.cache_implementation = "static"

        tokenizer = AutoTokenizer.from_pretrained("itazap/blt-1b")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)

        output_text = tokenizer.decode(generated_ids[0])
        self.assertEqual(output_text, EXPECTED_TEXT)
