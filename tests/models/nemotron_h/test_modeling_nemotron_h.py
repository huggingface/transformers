# Copyright 2024-2025 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch NemotronH model."""

import tempfile
import unittest

import pytest
from parameterized import parameterized

from transformers import AutoTokenizer, NemotronHConfig, NemotronHForCausalLM, is_torch_available
from transformers.testing_utils import (
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        NemotronHForCausalLM,
        NemotronHModel,
    )
    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHHybridDynamicCache,
    )


class NemotronHModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        layers_block_type=["mamba", "moe", "mamba", "attention", "moe"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=40,
        moe_intermediate_size=40,
        moe_shared_expert_intermediate_size=40,
        mlp_hidden_act="relu2",
        mamba_hidden_act="silu",
        max_position_embeddings=512,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        # Mamba-specific params
        ssm_state_size=16,
        mamba_num_heads=8,
        mamba_n_groups=8,
        mamba_head_dim=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=64,
        # MoE params
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layers_block_type = layers_block_type
        # num_hidden_layers is now derived from layers_block_type length
        self.num_hidden_layers = len(layers_block_type)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.mlp_hidden_act = mlp_hidden_act
        self.mamba_hidden_act = mamba_hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices

        # Mamba params
        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.mamba_n_groups = mamba_n_groups
        self.mamba_head_dim = mamba_head_dim
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_chunk_size = mamba_chunk_size

        # MoE params
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return NemotronHConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            layers_block_type=self.layers_block_type,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            moe_shared_expert_intermediate_size=self.moe_shared_expert_intermediate_size,
            mlp_hidden_act=self.mlp_hidden_act,
            mamba_hidden_act=self.mamba_hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            is_decoder=True,
            initializer_range=self.initializer_range,
            use_mamba_kernels=False,
            ssm_state_size=self.ssm_state_size,
            mamba_num_heads=self.mamba_num_heads,
            mamba_n_groups=self.mamba_n_groups,
            mamba_head_dim=self.mamba_head_dim,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            mamba_chunk_size=self.mamba_chunk_size,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            num_experts_per_tok=self.num_experts_per_tok,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def create_and_check_model(self, config, input_ids, input_mask, _sequence_labels, _token_labels, _choice_labels):
        model = NemotronHModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        _sequence_labels,
        token_labels,
        _choice_labels,
    ):
        model = NemotronHForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids, labels=token_labels)
        result = model(input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        input_mask,
        _sequence_labels,
        _token_labels,
        _choice_labels,
    ):
        config.is_decoder = True
        config.add_cross_attention = False
        model = NemotronHForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        # Attention: NemotronH needs the cache to be initialized to return a cache!
        past_key_values = NemotronHHybridDynamicCache(config, input_ids.shape[0], model.dtype, device=model.device)
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 1), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            cache_position=torch.arange(
                input_ids.shape[1], input_ids.shape[1] + next_tokens.shape[1], device=model.device
            ),
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_mamba2_slow_vs_fast_forward(self, config, input_ids, *args):
        """
        Test that cuda_kernels_forward and torch_forward produce consistent outputs.
        This ensures that the optimized CUDA kernel path and the pure PyTorch path
        are equivalent.
        """
        model = NemotronHModel(config)
        model.eval()

        if not (is_mamba_ssm_available() and is_causal_conv1d_available()):
            self.parent.skipTest(
                "This test needs the Mamba2 fast path. Skipping as the necessary packages have not been found."
            )
        if torch_device != "cuda":
            self.parent.skipTest("This test needs the Mamba2 fast path. Skipping as we need a cuda capable device.")

        model.to(torch_device)

        # Get the first mamba layer for testing
        # Find the index of the first mamba layer
        mamba_layer_idx = None
        for idx, layer_type in enumerate(config.layers_block_type):
            if layer_type == "mamba":
                mamba_layer_idx = idx
                break

        if mamba_layer_idx is None:
            self.parent.skipTest("No mamba layer found in the model configuration.")

        # Get embeddings
        token_emb = model.embeddings(input_ids.to(torch_device))

        # Get the mamba mixer from the first mamba block
        mamba_mixer = model.layers[mamba_layer_idx].mixer

        # Test without cache
        outputs_fast = mamba_mixer.cuda_kernels_forward(token_emb)
        outputs_slow = mamba_mixer.torch_forward(token_emb)

        self.parent.assertTrue(torch.allclose(outputs_fast, outputs_slow, atol=1e-3, rtol=1e-3))

        # Test with cache
        batch_size = input_ids.shape[0]
        cache_params = NemotronHHybridDynamicCache(
            config=config, batch_size=batch_size, dtype=token_emb.dtype, device=torch_device
        )

        outputs_fast_cached = mamba_mixer.cuda_kernels_forward(token_emb, cache_params=cache_params)

        # Reset cache for fair comparison
        cache_params_slow = NemotronHHybridDynamicCache(
            config=config, batch_size=batch_size, dtype=token_emb.dtype, device=torch_device
        )
        outputs_slow_cached = mamba_mixer.torch_forward(token_emb, cache_params=cache_params_slow)

        self.parent.assertTrue(torch.allclose(outputs_fast_cached, outputs_slow_cached, atol=1e-3, rtol=1e-3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class NemotronHModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            NemotronHModel,
            NemotronHForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": NemotronHModel,
            "text-generation": NemotronHForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        self.assertIsInstance(past_key_values, NemotronHHybridDynamicCache)

        # (batch, kv heads, seq_length, head_dim)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        attention_shape = (batch_size, num_heads, seq_length, head_dim)

        # Mamba cache shapes
        intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        conv_shape = (
            batch_size,
            intermediate_size + 2 * config.n_groups * config.ssm_state_size,
            config.conv_kernel,
        )
        ssm_shape = (batch_size, config.mamba_num_heads, config.mamba_head_dim, config.ssm_state_size)

        self.assertTrue(config.num_hidden_layers, len(past_key_values))

        for idx in range(len(past_key_values)):
            if config.layers_block_type[idx] == "mamba":
                self.assertEqual(past_key_values.conv_states[idx].shape, conv_shape)
                self.assertEqual(past_key_values.ssm_states[idx].shape, ssm_shape)
            elif config.layers_block_type[idx] == "attention":
                self.assertEqual(past_key_values.key_cache[idx].shape, attention_shape)
                self.assertEqual(past_key_values.value_cache[idx].shape, attention_shape)

    def _check_caches_are_equal(self, cache1: NemotronHHybridDynamicCache, cache2: NemotronHHybridDynamicCache):
        if not isinstance(cache1, NemotronHHybridDynamicCache) or not isinstance(cache2, NemotronHHybridDynamicCache):
            raise ValueError("The wrong cache is being used!")

        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            torch.testing.assert_close(cache1.key_cache[idx], cache2.key_cache[idx])
            torch.testing.assert_close(cache1.value_cache[idx], cache2.value_cache[idx])
            torch.testing.assert_close(cache1.conv_states[idx], cache2.conv_states[idx])
            torch.testing.assert_close(cache1.ssm_states[idx], cache2.ssm_states[idx])

    def setUp(self):
        self.model_tester = NemotronHModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=NemotronHConfig, common_properties=["hidden_size", "num_attention_heads"]
        )
        # Save original settings
        self._original_deterministic = torch.are_deterministic_algorithms_enabled()
        self._original_cudnn_deterministic = torch.backends.cudnn.deterministic
        self._original_cudnn_benchmark = torch.backends.cudnn.benchmark
        # Apply deterministic settings for NemotronH tests
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def tearDown(self):
        # Restore original settings
        torch.use_deterministic_algorithms(self._original_deterministic)
        torch.backends.cudnn.deterministic = self._original_cudnn_deterministic
        torch.backends.cudnn.benchmark = self._original_cudnn_benchmark

    @unittest.skip(reason="NemotronH needs at least 3 layers to test (mamba, moe, attention)")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("position_ids cannot be used to pad due to Mamba2 layers")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(reason="NemotronH has hybrid cache.")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip(reason="A large nemotron3 would be necessary (and costly) for that")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_reverse_loading_mapping(self):
        original_all_model_classes = self.all_model_classes
        self.all_model_classes = (NemotronHForCausalLM,) if is_torch_available() else ()
        super().test_reverse_loading_mapping()
        self.all_model_classes = original_all_model_classes

    # TODO(liding):
    # in test_configuration_common.py, three tests failed
    # create_and_test_config_to_json_file
    # create_and_test_config_from_and_save_pretrained
    # create_and_test_config_from_and_save_pretrained_composite
    # def test_config(self):
    #     self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_mamba2_slow_vs_fast_forward(self):
        """
        Test that cuda_kernels_forward and torch_forward produce consistent outputs.
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_slow_vs_fast_forward(*config_and_inputs)

    def test_attention_outputs(self):
        r"""
        Overriding the test_attention_outputs test as the NemotronH model outputs attention only for its attention layers
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        # Count attention layers from hybrid pattern
        num_attention_layers = config.hybrid_override_pattern.count("*")

        for model_class in self.all_model_classes:
            print(f"Testing model class: {model_class}")
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
            self.assertEqual(len(attentions), num_attention_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), num_attention_layers)

            if num_attention_layers > 0:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions
            self.assertEqual(len(self_attentions), num_attention_layers)

            if num_attention_layers > 0:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

    @require_flash_attn
    @require_torch_accelerator
    @require_bitsandbytes
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_fp32_ln(self):
        r"""
        Overriding the test_flash_attn_2_fp32_ln test as the NemotronH model, like Zamba2, doesn't support
        right padding + use cache with FA2
        """
        from transformers import BitsAndBytesConfig

        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_input = inputs_dict[model.main_input_name]
                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
                # NOTE: NemotronH does not support right padding + use_cache with FA2.
                dummy_attention_mask[:, -1] = 1

                model = model_class.from_pretrained(
                    tmpdirname,
                    dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                )

                for _, param in model.named_parameters():
                    # upcast only layer norms
                    if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                        param.data = param.data.to(torch.float32)

                _ = model(dummy_input)
                # with attention mask
                _ = model(dummy_input, attention_mask=dummy_attention_mask)

    @unittest.skip(reason="NemotronH has its own special cache type")
    @parameterized.expand([(1, False), (1, True), (4, False)])
    def test_new_cache_format(self, num_beams, do_sample):
        pass

    @require_torch_accelerator
    def test_flex_attention_with_grads(self):
        """
        Overwriting as the base hidden size is big enough for compile.
        Manipulation of dims causes issues due to other constraints not being satisfied anymore.
        """
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config._attn_implementation = "flex_attention"

            model = model_class(config).to(device=torch_device)
            self.assertTrue(model.config._attn_implementation == "flex_attention")

            # Elaborate workaround for encoder-decoder models as some do not specify their main input
            dummy_inputs = {model.main_input_name: inputs_dict[model.main_input_name].to(torch_device)}
            if config.is_encoder_decoder:
                dummy_inputs["decoder_input_ids"] = inputs_dict["decoder_input_ids"].to(torch_device)
                dummy_inputs["decoder_attention_mask"] = inputs_dict["decoder_attention_mask"].to(torch_device)

            # If this does not raise an error, the test passes (see https://github.com/huggingface/transformers/pull/35605)
            _ = model(**dummy_inputs)

    def test_layers_block_type_validation(self):
        """Test that layers_block_type is validated correctly"""

        # Valid list - should work
        config = NemotronHConfig(
            vocab_size=100, hidden_size=32, layers_block_type=["mamba", "moe", "attention", "moe"]
        )
        self.assertEqual(len(config.layers_block_type), 4)
        self.assertEqual(config.num_hidden_layers, 4)

        # Invalid layer type - should raise error
        with self.assertRaises(ValueError):
            NemotronHConfig(
                vocab_size=100,
                hidden_size=32,
                layers_block_type=["mamba", "moe", "attention", "invalid"],  # "invalid" is not valid
            )

    def test_layers_block_type(self):
        """Test that layers_block_type works correctly and backward compatibility"""
        # Create config with explicit list
        config = NemotronHConfig(
            vocab_size=100, hidden_size=32, layers_block_type=["mamba", "moe", "attention", "moe"]
        )

        # Test direct access to layers_block_type
        self.assertEqual(config.layers_block_type[0], "mamba")
        self.assertEqual(config.layers_block_type[1], "moe")
        self.assertEqual(config.layers_block_type[2], "attention")
        self.assertEqual(config.layers_block_type[3], "moe")

        # Test that num_hidden_layers is derived from layers_block_type length
        self.assertEqual(config.num_hidden_layers, 4)

        # Test backward compatibility - hybrid_override_pattern property
        self.assertEqual(config.hybrid_override_pattern, "ME*E")

        # Test the model tester config
        config2 = self.model_tester.get_config()
        self.assertEqual(len(config2.layers_block_type), 5)
        self.assertEqual(config2.layers_block_type[0], "mamba")
        self.assertEqual(config2.layers_block_type[1], "moe")
        self.assertEqual(config2.layers_block_type[2], "mamba")
        self.assertEqual(config2.layers_block_type[3], "attention")
        self.assertEqual(config2.layers_block_type[4], "moe")

    def test_generate_with_and_without_cache(self):
        """Test that generation with and without cache produces identical outputs"""
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]

        # Create model
        model = NemotronHForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # Create input for generation (smaller sequence for faster test)
        input_ids = ids_tensor([1, 5], config.vocab_size)  # batch_size=1, seq_len=5
        input_ids = input_ids.to(torch_device)

        # Set seed for reproducibility
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        # Generate with cache
        with torch.no_grad():
            print("running generate with cache")
            output_with_cache = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                use_cache=True,
            )

        # Reset seed
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        # Generate without cache
        with torch.no_grad():
            print("running generate without cache")
            output_without_cache = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                use_cache=False,
            )

        print(f"output_with_cache: {output_with_cache}")
        print(f"output_without_cache: {output_without_cache}")

        # Outputs should be identical
        self.assertTrue(
            torch.equal(output_with_cache, output_without_cache),
            msg=f"Outputs differ:\n  With cache: {output_with_cache}\n  Without cache: {output_without_cache}",
        )

    def test_legacy_hybrid_override_pattern(self):
        """Test backward compatibility with legacy hybrid_override_pattern"""
        # Create config using legacy hybrid_override_pattern
        config = NemotronHConfig(vocab_size=100, hidden_size=32, hybrid_override_pattern="ME*E")

        # Test that it's converted to layers_block_type
        self.assertEqual(config.layers_block_type, ["mamba", "moe", "attention", "moe"])
        self.assertEqual(config.num_hidden_layers, 4)
        self.assertEqual(config.hybrid_override_pattern, "ME*E")

        # Test with longer pattern
        config2 = NemotronHConfig(vocab_size=100, hidden_size=32, hybrid_override_pattern="MEME*EME")
        self.assertEqual(
            config2.layers_block_type, ["mamba", "moe", "mamba", "moe", "attention", "moe", "mamba", "moe"]
        )
        self.assertEqual(config2.num_hidden_layers, 8)

    def test_num_hidden_layers_deprecated(self):
        """Test that num_hidden_layers is now derived from layers_block_type length"""
        # Test that num_hidden_layers is derived from layers_block_type
        config = NemotronHConfig(layers_block_type=["mamba", "moe", "attention", "moe", "mamba", "attention"])
        self.assertEqual(config.num_hidden_layers, 6)

        # Test that num_hidden_layers parameter is ignored when layers_block_type is provided
        config2 = NemotronHConfig(
            layers_block_type=["mamba", "moe", "attention"],
            num_hidden_layers=10,  # This should be ignored
        )
        # Should use layers_block_type length, not the parameter
        self.assertEqual(config2.num_hidden_layers, 3)

    def test_legacy_config_json_loading(self):
        """Test loading legacy config.json with hybrid_override_pattern and num_hidden_layers"""
        import json

        # Create a legacy config.json
        legacy_config = {
            "model_type": "nemotron_3",
            "vocab_size": 100,
            "hidden_size": 32,
            "num_hidden_layers": 6,
            "hybrid_override_pattern": "MEME*E",
            "num_attention_heads": 4,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = f"{tmpdir}/config.json"
            with open(config_path, "w") as f:
                json.dump(legacy_config, f)

            # Load the config
            config = NemotronHConfig.from_json_file(config_path)

            # Verify conversion
            self.assertEqual(len(config.layers_block_type), 6)
            self.assertEqual(config.num_hidden_layers, 6)
            self.assertEqual(config.layers_block_type, ["mamba", "moe", "mamba", "moe", "attention", "moe"])
            self.assertEqual(config.hybrid_override_pattern, "MEME*E")

    def test_mtp_backward_compatibility(self):
        """Test MTP backward compatibility with mtp_hybrid_override_pattern"""
        config = NemotronHConfig(
            layers_block_type=["mamba", "moe", "attention", "moe"],
            num_nextn_predict_layers=2,
            mtp_hybrid_override_pattern="*E",
        )

        # Verify conversion
        self.assertEqual(config.mtp_layers_block_type, ["attention", "moe"])
        self.assertEqual(config.mtp_hybrid_override_pattern, "*E")

    def test_config_roundtrip_save_load(self):
        """Test that config can be saved and loaded correctly"""
        # Create config with new format
        config1 = NemotronHConfig(
            vocab_size=100, hidden_size=32, layers_block_type=["mamba", "attention", "moe", "attention"]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            config1.save_pretrained(tmpdir)

            # Load
            config2 = NemotronHConfig.from_pretrained(tmpdir)

            # Verify
            self.assertEqual(config2.layers_block_type, ["mamba", "attention", "moe", "attention"])
            self.assertEqual(config2.num_hidden_layers, 4)
            self.assertEqual(config2.vocab_size, 100)
            self.assertEqual(config2.hidden_size, 32)

    def test_pattern_conversion_methods(self):
        """Test the pattern conversion utility methods"""
        # Test _pattern_to_list
        pattern = "M*EME*"
        layers_list = NemotronHConfig._pattern_to_list(pattern)
        self.assertEqual(layers_list, ["mamba", "attention", "moe", "mamba", "moe", "attention"])

        # Test _list_to_pattern
        layers_list = ["mamba", "moe", "attention", "moe"]
        pattern = NemotronHConfig._list_to_pattern(layers_list)
        self.assertEqual(pattern, "ME*E")

        # Test roundtrip
        original_pattern = "ME*ME*E"
        roundtrip_pattern = NemotronHConfig._list_to_pattern(NemotronHConfig._pattern_to_list(original_pattern))
        self.assertEqual(original_pattern, roundtrip_pattern)


@require_torch
class NemotronHModelIntegrationTest(unittest.TestCase):
    model = None
    tokenizer = None

    @classmethod
    @slow
    def setUpClass(cls):
        model_id = "dmax123/tiny-nemotron-dummy-weights"
        revision = "081dbac3061bb16c0c458c1798b1d9d7bc135c95"
        cls.model = NemotronHForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, revision=revision)
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def setUp(self):
        # Save original settings
        self._original_deterministic = torch.are_deterministic_algorithms_enabled()
        self._original_cudnn_deterministic = torch.backends.cudnn.deterministic
        self._original_cudnn_benchmark = torch.backends.cudnn.benchmark
        # Apply deterministic settings for NemotronH tests
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def tearDown(self):
        # Restore original settings
        torch.use_deterministic_algorithms(self._original_deterministic)
        torch.backends.cudnn.deterministic = self._original_cudnn_deterministic
        torch.backends.cudnn.benchmark = self._original_cudnn_benchmark

    @slow
    def test_simple_generate(self):
        self.model.to(torch_device)

        prompt = "Hey how are you doing?"
        EXPECTED_TOKENS_IDS = torch.tensor(
            [1045, 1429, 1073, 4525, 1605, 1261, 4249, 1044, 2081, 2224], dtype=torch.int32
        )

        messages = [{"role": "user", "content": prompt}]
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = tokenized_chat["input_ids"].to(torch_device)
        prompt_length = input_ids.shape[1]

        outputs = self.model.generate(input_ids, do_sample=False, max_new_tokens=10)

        generated_tokens = outputs[0][prompt_length:]
        self.assertTrue(torch.equal(generated_tokens.cpu(), EXPECTED_TOKENS_IDS.cpu()))
