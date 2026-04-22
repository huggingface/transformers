# Copyright 2024-2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch NemotronHSparse model."""

import tempfile
import unittest

import pytest
from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import AutoTokenizer, NemotronHSparseConfig, NemotronHSparseForCausalLM, is_torch_available
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

    from transformers import DynamicCache, NemotronHSparseForCausalLM, NemotronHSparseModel


class NemotronHSparseModelTester:
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
        hybrid_override_pattern="ME*ME",
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        moe_intermediate_size=40,
        moe_shared_expert_intermediate_size=40,
        mlp_hidden_act="relu2",
        mamba_hidden_act="silu",
        max_position_embeddings=512,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        ssm_state_size=16,
        mamba_num_heads=8,
        mamba_n_groups=8,
        mamba_head_dim=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=64,
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
        self.hybrid_override_pattern = hybrid_override_pattern
        # `E` is absorbed as an FFN tail; each `M`/`*` counts as one logical decoder layer.
        self.num_hidden_layers = sum(1 for c in hybrid_override_pattern if c in ("M", "*"))
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.mlp_hidden_act = mlp_hidden_act
        self.mamba_hidden_act = mamba_hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices

        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.mamba_n_groups = mamba_n_groups
        self.mamba_head_dim = mamba_head_dim
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_chunk_size = mamba_chunk_size

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
        return NemotronHSparseConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            hybrid_override_pattern=self.hybrid_override_pattern,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
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
        model = NemotronHSparseModel(config=config)
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
        model = NemotronHSparseForCausalLM(config=config)
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
        model = NemotronHSparseForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        outputs = model(
            input_ids,
            attention_mask=input_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 1), vocab_size=2)

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
        )["hidden_states"][0]

        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_mamba2_slow_vs_fast_forward(self, config, input_ids, *args):
        model = NemotronHSparseModel(config)
        model.eval()

        if not (is_mamba_ssm_available() and is_causal_conv1d_available()):
            self.parent.skipTest(
                "This test needs the Mamba2 fast path. Skipping as the necessary packages have not been found."
            )
        if torch_device != "cuda":
            self.parent.skipTest("This test needs the Mamba2 fast path. Skipping as we need a cuda capable device.")

        model.to(torch_device)

        mamba_layer_idx = None
        for idx, layer_type in enumerate(config.layer_types):
            if layer_type == "mamba":
                mamba_layer_idx = idx
                break

        if mamba_layer_idx is None:
            self.parent.skipTest("No mamba layer found in the model configuration.")

        token_emb = model.embed_tokens(input_ids.to(torch_device))
        mamba_mixer = model.layers[mamba_layer_idx].mamba

        outputs_fast = mamba_mixer.cuda_kernels_forward(token_emb)
        outputs_slow = mamba_mixer.torch_forward(token_emb)

        self.parent.assertTrue(torch.allclose(outputs_fast, outputs_slow, atol=1e-3, rtol=1e-3))

        cache_params = DynamicCache(config=config)
        outputs_fast_cached = mamba_mixer.cuda_kernels_forward(token_emb, cache_params=cache_params)

        cache_params_slow = DynamicCache(config=config)
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
class NemotronHSparseModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            NemotronHSparseModel,
            NemotronHSparseForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": NemotronHSparseModel,
            "text-generation": NemotronHSparseForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    def _get_conv_state_shape(self, batch_size: int, config):
        intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        conv_shape = (
            batch_size,
            intermediate_size + 2 * config.n_groups * config.ssm_state_size,
            config.conv_kernel,
        )
        return conv_shape

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.mamba_num_heads, config.mamba_head_dim, config.ssm_state_size)

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        if not isinstance(past_key_values, DynamicCache):
            raise ValueError("The cache does not use the correct Cache")

        config = config.get_text_config(decoder=True)

        num_attention_heads = getattr(config, "num_attention_heads", 1)
        num_kv_heads = getattr(config, "num_key_value_heads", num_attention_heads)
        hidden_size = getattr(config, "d_model", config.hidden_size)
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

        attention_shape = (batch_size, num_kv_heads, seq_length, head_dim)
        conv_shape = self._get_conv_state_shape(batch_size, config)
        recurrent_shape = self._get_recurrent_state_shape(batch_size, config)

        for layer, layer_type in zip(past_key_values.layers, config.layer_types):
            if layer_type == "attention":
                self.assertEqual(layer.keys.shape, attention_shape)
                self.assertEqual(layer.values.shape, attention_shape)
            elif layer_type == "mamba":
                self.assertEqual(layer.conv_states.shape, conv_shape)
                self.assertEqual(layer.recurrent_states.shape, recurrent_shape)
            else:
                raise ValueError("Unknown layer type.")

    def setUp(self):
        self.model_tester = NemotronHSparseModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=NemotronHSparseConfig, common_properties=["hidden_size", "num_attention_heads"]
        )
        self._original_deterministic = torch.are_deterministic_algorithms_enabled()
        self._original_cudnn_deterministic = torch.backends.cudnn.deterministic
        self._original_cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def tearDown(self):
        torch.use_deterministic_algorithms(self._original_deterministic)
        torch.backends.cudnn.deterministic = self._original_cudnn_deterministic
        torch.backends.cudnn.benchmark = self._original_cudnn_benchmark

    @unittest.skip(reason="NemotronHSparse needs at least 3 layers to test (mamba, moe, attention)")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("position_ids cannot be used to pad due to Mamba2 layers")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(reason="NemotronHSparse has hybrid cache.")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip(reason="Hybrid mamba/attention cache continuation needs separate fix.")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip(reason="A large nemotron3 would be necessary (and costly) for that")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_reverse_loading_mapping(self):
        super().test_reverse_loading_mapping(skip_base_model=True)

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
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mamba2_slow_vs_fast_forward(*config_and_inputs)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        num_attention_layers = config.hybrid_override_pattern.count("*")

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
            self.assertEqual(len(attentions), num_attention_layers)

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
        from transformers import BitsAndBytesConfig

        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                dummy_input = inputs_dict[model.main_input_name]
                dummy_attention_mask = inputs_dict.get("attention_mask", torch.ones_like(dummy_input))
                dummy_attention_mask[:, -1] = 1

                model = model_class.from_pretrained(
                    tmpdirname,
                    dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                )

                for _, param in model.named_parameters():
                    if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                        param.data = param.data.to(torch.float32)

                _ = model(dummy_input)
                _ = model(dummy_input, attention_mask=dummy_attention_mask)

    @require_torch_accelerator
    def test_flex_attention_with_grads(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config._attn_implementation = "flex_attention"

            model = model_class(config).to(device=torch_device)
            self.assertTrue(model.config._attn_implementation == "flex_attention")

            dummy_inputs = {model.main_input_name: inputs_dict[model.main_input_name].to(torch_device)}
            if config.is_encoder_decoder:
                dummy_inputs["decoder_input_ids"] = inputs_dict["decoder_input_ids"].to(torch_device)
                dummy_inputs["decoder_attention_mask"] = inputs_dict["decoder_attention_mask"].to(torch_device)

            _ = model(**dummy_inputs)

    def test_hybrid_override_pattern_validation(self):
        """`hybrid_override_pattern` only accepts M/*/E (MLP `-` rejected)."""
        config = NemotronHSparseConfig(vocab_size=100, hidden_size=32, hybrid_override_pattern="MEME*E")
        self.assertEqual(config.layer_types, ["mamba", "mamba", "attention"])
        self.assertEqual(config.num_hidden_layers, 3)

        with self.assertRaises((ValueError, StrictDataclassClassValidationError)):
            NemotronHSparseConfig(vocab_size=100, hidden_size=32, hybrid_override_pattern="M-*")

    def test_layer_types_property(self):
        config = NemotronHSparseConfig(vocab_size=100, hidden_size=32, hybrid_override_pattern="MEME*E")
        self.assertEqual(config.layer_types, ["mamba", "mamba", "attention"])
        self.assertEqual(config.layers_block_type, ["mamba", "mamba", "attention"])
        self.assertEqual(config.num_hidden_layers, 3)

    def test_mtp_kwargs_ignored(self):
        """MTP kwargs are silently dropped — MTP is not modeled in transformers."""
        config = NemotronHSparseConfig(
            hybrid_override_pattern="ME*E",
            num_nextn_predict_layers=2,
            mtp_hybrid_override_pattern="*E",
        )
        self.assertFalse(hasattr(config, "num_nextn_predict_layers"))
        self.assertFalse(hasattr(config, "mtp_hybrid_override_pattern"))

    def test_generate_with_and_without_cache(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]

        model = NemotronHSparseForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = ids_tensor([1, 5], config.vocab_size)
        input_ids = input_ids.to(torch_device)

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        with torch.no_grad():
            output_with_cache = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                use_cache=True,
            )

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        with torch.no_grad():
            output_without_cache = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                use_cache=False,
            )

        self.assertTrue(torch.equal(output_with_cache, output_without_cache))

    def test_config_roundtrip_save_load(self):
        config1 = NemotronHSparseConfig(vocab_size=100, hidden_size=32, hybrid_override_pattern="ME*EM*E")

        with tempfile.TemporaryDirectory() as tmpdir:
            config1.save_pretrained(tmpdir)
            config2 = NemotronHSparseConfig.from_pretrained(tmpdir)

            self.assertEqual(config2.hybrid_override_pattern, "ME*EM*E")
            self.assertEqual(config2.num_hidden_layers, 4)
            self.assertEqual(config2.vocab_size, 100)
            self.assertEqual(config2.hidden_size, 32)


@require_torch
class NemotronHSparseModelIntegrationTest(unittest.TestCase):
    model = None
    tokenizer = None

    @classmethod
    @slow
    def setUpClass(cls):
        model_id = "dmax123/tiny-nemotron-dummy-weights"
        revision = "081dbac3061bb16c0c458c1798b1d9d7bc135c95"
        cls.model = NemotronHSparseForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, revision=revision)
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def setUp(self):
        self._original_deterministic = torch.are_deterministic_algorithms_enabled()
        self._original_cudnn_deterministic = torch.backends.cudnn.deterministic
        self._original_cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def tearDown(self):
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
