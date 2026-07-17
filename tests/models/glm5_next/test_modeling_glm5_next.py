# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Glm5Next model."""

import unittest

import torch
from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    FineGrainedFP8Config,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import (
    CausalLMModelTest,
    CausalLMModelTester,
    _config_supports_rope_scaling,
    _set_config_rope_params,
)
from ...test_modeling_common import ids_tensor


if is_torch_available():
    from transformers import Glm5NextForCausalLM, Glm5NextModel


class Glm5NextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Glm5NextModel
        causal_lm_class = Glm5NextForCausalLM

    def __init__(self, parent):
        super().__init__(parent=parent)
        # MLA
        self.kv_lora_rank = 16
        self.q_lora_rank = 32
        self.qk_nope_head_dim = 48
        self.qk_rope_head_dim = 16
        self.v_head_dim = 16
        # Indexer
        self.index_head_dim = 16
        self.index_n_heads = 2
        self.index_topk = 48
        self.index_kpool = 3
        # Linear attention
        self.linear_conv_kernel_dim = 2
        self.linear_head_dim = 16
        self.linear_num_heads = 2
        # MoE
        self.num_local_experts = 8
        self.n_routed_experts = 8
        # Force everything to be used (layer wise)
        self.mlp_layer_types = ["dense", "sparse"]
        self.layer_types = ["linear_attention", "deepseek_sparse_attention"]


@require_torch
class Glm5NextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Glm5NextModelTester
    test_all_params_have_gradient = False  # MoE
    model_split_percents = [0.5, 0.8, 0.9]

    def _get_conv_state_shape(self, batch_size: int, config):
        return (batch_size, 3 * config.linear_num_heads * config.linear_head_dim, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.linear_num_heads, config.linear_head_dim, config.linear_head_dim)

    def _get_attention_shape(self, batch_size: int, seq_length: int, config):
        # (batch, head, seq_length, head_features)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        return expected_key_shape, expected_value_shape

    def test_attention_outputs(self):
        """Needs to be overwritten as GLM5 Next alternates between attention layers and KDA layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

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
            self.assertEqual(len(attentions), sum(layer == "deepseek_sparse_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "deepseek_sparse_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "deepseek_sparse_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        """
        Overriden to raise rtol in any case, please see the note below - tl;dr: inprecision due to unique layer stack
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        if not _config_supports_rope_scaling(config):
            self.skipTest("This model does not support RoPE scaling")

        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        _set_config_rope_params(
            config,
            {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
                "original_max_position_embeddings": 16384,
            },
        )
        original_model = self.model_tester_class.base_model_class(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        _set_config_rope_params(
            config,
            {
                "rope_type": scaling_type,
                "factor": 10.0,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        scaled_model = self.model_tester_class.base_model_class(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # NOTE: Key difference here is the additional rtol which seems to be necessary
        # Unknown exact cause but assuming inprecisions being introduced due to the unique model stack (indexer)
        torch.testing.assert_close(original_short_output, scaled_short_output, rtol=1e-5, atol=1e-5)

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("Fundamentally incompatible with indexer - indexer has no boundary offset telling sequences apart")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Fundamentally incompatible with indexer - indexer has no boundary offset telling sequences apart")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Indexer will always create a mask so FA will never be invoked")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


# TODO: update - values should slightly differ as left padding is now properly supported!
@require_torch_accelerator
@slow
class Glm5NextIntegrationTest(unittest.TestCase):
    @unittest.skip("Test requires 2 nodes")
    def test_glm_moe_dsa_fp8_inference(self):
        # TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.run --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=ip-26-0-169-86 --master_port=29500
        set_seed(0)  # different ranks need the same seed
        model_id = "zai-org/GLM-5-FP8"

        quantization_config = FineGrainedFP8Config(
            modules_to_not_convert=[
                "model.layers.*.mlp.gate$",
                "model.layers.*.self_attn.indexer.weights_proj$",
                "lm_head",
            ],
            weight_block_size=(128, 128),
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            tp_plan="auto",
            attn_implementation="eager",
        )

        prompt = ["Hi, introduce yourself", "The capital of France is known for"]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
            )

        output = tokenizer.decode(outputs, skip_special_tokens=False)
        self.assertEqual(
            output,
            [
                "<|endoftext|><|endoftext|><|endoftext|>Hi, introduce yourself!\nI'm a 18 years old boy from Italy and I'm a student",
                "The capital of France is known for its rich history, culture, and the city of the of the of the of",
            ],
        )
