# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch LLaMA model. """

import tempfile
import unittest

import pytest
from parameterized import parameterized

from transformers import LlamaConfig, StaticCache, is_torch_available, logging, set_seed
from transformers.testing_utils import (
    CaptureLogger,
    require_bitsandbytes,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        CodeLlamaTokenizer,
        LlamaForCausalLM,
        LlamaForQuestionAnswering,
        LlamaForSequenceClassification,
        LlamaModel,
        LlamaTokenizer,
    )
    from transformers.models.llama.modeling_llama import (
        LlamaDynamicNTKScalingRotaryEmbedding,
        LlamaLinearScalingRotaryEmbedding,
        LlamaRotaryEmbedding,
    )


class LlamaModelTester:
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
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
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

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones(self.batch_size, self.seq_length)).to(torch_device)

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
        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LlamaModel(config=config)
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
        model = LlamaModel(config)
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
        model = LlamaForCausalLM(config=config)
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
        model = LlamaForCausalLM(config=config)
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
class LlamaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (LlamaModel, LlamaForCausalLM, LlamaForSequenceClassification, LlamaForQuestionAnswering)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (LlamaForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": LlamaModel,
            "text-classification": LlamaForSequenceClassification,
            "text-generation": LlamaForCausalLM,
            "zero-shot": LlamaForSequenceClassification,
            "question-answering": LlamaForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = True

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    def setUp(self):
        self.model_tester = LlamaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlamaConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_llama_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = LlamaForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_llama_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = LlamaForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_llama_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = LlamaForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    @unittest.skip("Llama buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = LlamaModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = LlamaModel(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            self.assertTrue(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    def test_model_rope_scaling(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
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
        original_rope = LlamaRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        ).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        linear_scaling_rope = LlamaLinearScalingRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=scaling_factor,
        ).to(torch_device)
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
        ntk_scaling_rope = LlamaDynamicNTKScalingRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=scaling_factor,
        ).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short)
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())

    @require_flash_attn
    @require_torch_gpu
    @require_bitsandbytes
    @pytest.mark.flash_attn_test
    @require_read_token
    @slow
    def test_flash_attn_2_generate_padding_right(self):
        """
        Overwritting the common test as the test is flaky on tiny models
        """
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            load_in_4bit=True,
            device_map={"": 0},
        )

        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        texts = ["hi", "Hello this is a very long sentence"]

        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(0)

        output_native = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_native = tokenizer.batch_decode(output_native)

        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", load_in_4bit=True, device_map={"": 0}, attn_implementation="flash_attention_2"
        )

        output_fa_2 = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_fa_2 = tokenizer.batch_decode(output_fa_2)

        self.assertListEqual(output_native, output_fa_2)

    @require_flash_attn
    @require_torch_gpu
    @slow
    def test_use_flash_attention_2_true(self):
        """
        NOTE: this is the only test testing that the legacy `use_flash_attention=2` argument still works as intended.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            with tempfile.TemporaryDirectory() as tmp_dir:
                model = model_class(config)
                model.save_pretrained(tmp_dir)

                new_model = LlamaForCausalLM.from_pretrained(
                    tmp_dir, use_flash_attention_2=True, torch_dtype=torch.float16
                ).to("cuda")

                self.assertTrue(new_model.config._attn_implementation == "flash_attention_2")

                has_flash = False
                for name, submodule in new_model.named_modules():
                    if "FlashAttention" in submodule.__class__.__name__:
                        has_flash = True
                        break
                if not has_flash:
                    raise ValueError("The flash model should have flash attention layers")

    @require_torch_sdpa
    @slow
    def test_eager_matches_sdpa_generate(self):
        """
        Overwritting the common test as the test is flaky on tiny models
        """
        max_new_tokens = 30

        tokenizer = LlamaTokenizer.from_pretrained("saibo/llama-1B")

        model_sdpa = LlamaForCausalLM.from_pretrained(
            "saibo/llama-1B",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(torch_device)

        self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

        model_eager = LlamaForCausalLM.from_pretrained(
            "saibo/llama-1B",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(torch_device)

        self.assertTrue(model_eager.config._attn_implementation == "eager")

        for name, submodule in model_eager.named_modules():
            if "SdpaAttention" in submodule.__class__.__name__:
                raise ValueError("The eager model should not have SDPA attention layers")

        has_sdpa = False
        for name, submodule in model_sdpa.named_modules():
            if "SdpaAttention" in submodule.__class__.__name__:
                has_sdpa = True
                break
        if not has_sdpa:
            raise ValueError("The SDPA model should have SDPA attention layers")

        texts = [
            "hi here's a longer context, getting longer and",
            "Hello this is a very long sentence my friend, very long for real",
            "Today I am in Paris and",
        ]

        for padding_side in ["left", "right"]:
            tokenizer.padding_side = padding_side
            tokenizer.pad_token = tokenizer.eos_token

            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(torch_device)

            res_eager = model_eager.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            res_sdpa = model_sdpa.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            with self.subTest(f"{padding_side}"):
                torch.testing.assert_close(
                    res_eager,
                    res_sdpa,
                    msg=f"\n{tokenizer.batch_decode(res_eager)} \nvs\n{tokenizer.batch_decode(res_sdpa)}",
                )

    @unittest.skip("TODO @gante fix this for Llama")
    @parameterized.expand([(1, False), (1, True), (4, False)])
    def test_new_cache_format(self, num_beams, do_sample):
        pass


@require_torch_gpu
class LlamaIntegrationTest(unittest.TestCase):
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    @unittest.skip("Logits are not exactly the same, once we fix the instabalities somehow, will update!")
    @slow
    def test_model_7b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
        out = model(torch.tensor([input_ids]))
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-6.6550, -4.1227, -4.9859, -3.2406, 0.8262, -3.0033, 1.2964, -3.3699]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-12.8281, -7.4453, -0.4639, -8.0625, -7.2500, -8.0000, -6.4883, -7.7695, -7.8438, -7.0312, -6.2188, -7.1328, -1.8496, 1.9961, -8.6250, -6.7227, -12.8281, -6.9492, -7.0742, -7.7852, -7.5820, -7.9062, -6.9375, -7.9805, -8.3438, -8.1562, -8.0469, -7.6250, -7.7422, -7.3398,])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)

    @unittest.skip("Logits are not exactly the same, once we fix the instabalities somehow, will update!")
    @slow
    def test_model_13b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", device_map="auto")
        out = model(torch.tensor(input_ids))
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-2.0622, -1.2794, -1.1638, -0.9788, -1.4603, -1.0238, -1.7893, -1.4411]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-8.1406, -8.0547, 2.7461, -1.2344, -0.1448, -1.8262, -1.0020, -1.8154, -1.6895, -1.8516, -2.3574, -0.9277, 3.7598, 6.5742, -1.2998, -0.1177, -8.1406, -2.9688, -2.9199, -3.1699, -3.5254, -2.3555, -2.7988, -3.4141, -2.8262, -4.5195, -3.3379, -3.3164, -2.7832, -3.0273])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)

    @unittest.skip("Logits are not exactly the same, once we fix the instabalities somehow, will update!")
    @slow
    def test_model_13bf_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", device_map="auto")
        out = model(torch.tensor(input_ids))
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-0.8562, -1.8520, -0.7551, -0.4162, -1.5161, -1.2038, -2.4823, -2.3254]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-2.2227, 4.8828, 0.9023, -0.4578, -0.7871, -0.1033, -0.6221, -0.5786, -0.7803, -1.0674, -1.2920, -0.1570, 0.8008, 2.0723, -0.9497, 0.2771, -2.2227, -0.7612, -1.4346, -1.2061, -1.6426, -0.3000, -0.7139, -1.1934, -1.8691, -1.6973, -1.5947, -1.2705, -0.3523, -0.5513])  # fmt: skip
        torch.testing.assert_close(out.mean(-1), EXPECTED_SLICE, atol=1e-2, rtol=1e-2)

    @unittest.skip(
        "Logits are not exactly the same, once we fix the instabalities somehow, will update! Also it is gonna be a `too_slow` test"
    )
    @slow
    def test_model_70b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf", device_map="auto")
        out = model(torch.tensor(input_ids))

        EXPECTED_MEAN = torch.tensor(
            [[-4.2327, -3.3360, -4.6665, -4.7631, -1.8180, -3.4170, -1.4211, -3.1810]], dtype=torch.float32
        )
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        EXPECTED_SLICE = torch.tensor([-9.4922, -3.9551, 1.7998, -5.6758, -5.1055, -5.8984, -4.8320, -6.8086, -6.5391, -5.6172, -5.5820, -5.5352, 1.7881, 3.6289, -6.5117, -3.4785, -9.5000, -6.0352, -6.8125, -6.0195, -6.6836, -5.4727, -6.2812, -6.0391, -7.3398, -7.4297, -7.4844, -6.5820, -5.8789, -5.5312])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)

    @unittest.skip("Model is curently gated")
    @slow
    def test_model_13b_greedy_generation(self):
        EXPECTED_TEXT_COMPLETION = """Simply put, the theory of relativity states that 1) the laws of physics are the same everywhere in the universe and 2) the passage of time and the length of objects can vary depending on the observer\'s frame of reference.\n\nThe first part of the theory, that the laws of physics are the same everywhere, is known as the "princi"""
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-13b-chat-hf", device_map="sequential", use_safetensors=False
        )

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_torch_gpu
    @require_read_token
    def test_compile_static_cache(self):
        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = {
            7: [
                "Simply put, the theory of relativity states that 1) the speed of light is constant, 2) the speed of light is the same for all observers, and 3) the laws of physics are the same for all observers.",
                "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my p",
            ],
            8: [
                "Simply put, the theory of relativity states that 1) the speed of light is the same for all observers, and 2) the laws of physics are the same for all observers.\nThe first part of the theory of relativity",
                "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my p",
            ],
        }

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", pad_token="</s>", padding_side="right")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", device_map="sequential", torch_dtype=torch.float16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        def decode_one_tokens(model, cur_token, input_pos, cache_position):
            logits = model(
                cur_token, position_ids=input_pos, cache_position=cache_position, return_dict=False, use_cache=True
            )[0]
            new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
            return new_token

        batch_size, seq_length = inputs["input_ids"].shape
        with torch.no_grad():
            model._setup_cache(StaticCache, 2, max_cache_len=4096)
            cache_position = torch.arange(seq_length, device=torch_device)
            generated_ids = torch.zeros(
                batch_size, seq_length + NUM_TOKENS_TO_GENERATE + 1, dtype=torch.int, device=torch_device
            )
            generated_ids[:, cache_position] = inputs["input_ids"].to(torch_device).to(torch.int)

            logits = model(**inputs, cache_position=cache_position, return_dict=False, use_cache=True)[0]
            next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
            generated_ids[:, seq_length] = next_token[:, 0]

            decode_one_tokens = torch.compile(decode_one_tokens, mode="reduce-overhead", fullgraph=True)
            cache_position = torch.tensor([seq_length + 1], device=torch_device)
            for _ in range(1, NUM_TOKENS_TO_GENERATE):
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                    with CaptureLogger(logging.get_logger(__name__)) as cl:
                        next_token = decode_one_tokens(model, next_token.clone(), None, cache_position)
                        self.assertNotIn("skipping cudagraphs due to", cl.out)
                    generated_ids[:, cache_position] = next_token.int()
                cache_position += 1

        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION[self.cuda_compute_capability_major_version], text)


@require_torch
class CodeLlamaIntegrationTest(unittest.TestCase):
    PROMPTS = [
        '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
''',
        """# Installation instructions:
    ```bash
<FILL_ME>
    ```
This downloads the LLaMA inference code and installs the repository as a local pip package.
""",
        """class InterfaceManagerFactory(AbstractManagerFactory):
    def __init__(<FILL_ME>
def main():
    factory = InterfaceManagerFactory(start=datetime.now())
    managers = []
    for i in range(10):
        managers.append(factory.build(id=i))
""",
        """/-- A quasi-prefunctoid is 1-connected iff all its etalisations are 1-connected. -/
theorem connected_iff_etalisation [C D : precategoroid] (P : quasi_prefunctoid C D) :
π₁ P = 0 ↔ <FILL_ME> = 0 :=
begin
split,
{ intros h f,
    rw pi_1_etalisation at h,
    simp [h],
    refl
},
{ intro h,
    have := @quasi_adjoint C D P,
    simp [←pi_1_etalisation, this, h],
    refl
}
end
""",
    ]

    @require_torch_accelerator
    @slow
    @unittest.skip("Model is too large")
    def test_model_7b_logits(self):
        model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf").to(torch_device)
        tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
        # Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.
        # meaning by default this supports passing splitted list of inputs
        processed_text = tokenizer.batch_decode(tokenizer(self.PROMPTS)["input_ids"], add_special_tokens=False)
        # fmt: off
        EXPECTED_TEXT = [
            '<s> <PRE> def remove_non_ascii(s: str) -> str:\n    """  <SUF>\n    return result\n <MID>',
            '<s> <PRE> # Installation instructions:\n    ```bash\n <SUF>\n    ```\nThis downloads the LLaMA inference code and installs the repository as a local pip package.\n <MID>',
            '<s> <PRE> class InterfaceManagerFactory(AbstractManagerFactory):\n    def __init__( <SUF>\ndef main():\n    factory = InterfaceManagerFactory(start=datetime.now())\n    managers = []\n    for i in range(10):\n        managers.append(factory.build(id=i))\n <MID>',
            '<s> <PRE> /-- A quasi-prefunctoid is 1-connected iff all its etalisations are 1-connected. -/\ntheorem connected_iff_etalisation [C D : precategoroid] (P : quasi_prefunctoid C D) :\nπ₁ P = 0 ↔  <SUF> = 0 :=\nbegin\nsplit,\n{ intros h f,\n    rw pi_1_etalisation at h,\n    simp [h],\n    refl\n},\n{ intro h,\n    have := @quasi_adjoint C D P,\n    simp [←pi_1_etalisation, this, h],\n    refl\n}\nend\n <MID>'
        ]
        # fmt: on
        self.assertEqual(processed_text, EXPECTED_TEXT)
        processed_text_suffix_first = tokenizer.batch_decode(
            tokenizer(self.PROMPTS, suffix_first=True, add_special_tokens=False)["input_ids"]
        )

        # fmt: off
        EXPECTED_TEXT = [
            '<PRE> <SUF>\n    return result\n <MID> def remove_non_ascii(s: str) -> str:\n    """ ',
            '<PRE> <SUF>\n    ```\nThis downloads the LLaMA inference code and installs the repository as a local pip package.\n <MID> # Installation instructions:\n    ```bash\n',
            '<PRE> <SUF>\ndef main():\n    factory = InterfaceManagerFactory(start=datetime.now())\n    managers = []\n    for i in range(10):\n        managers.append(factory.build(id=i))\n <MID> class InterfaceManagerFactory(AbstractManagerFactory):\n    def __init__(',
            '<PRE> <SUF> = 0 :=\nbegin\nsplit,\n{ intros h f,\n    rw pi_1_etalisation at h,\n    simp [h],\n    refl\n},\n{ intro h,\n    have := @quasi_adjoint C D P,\n    simp [←pi_1_etalisation, this, h],\n    refl\n}\nend\n <MID> /-- A quasi-prefunctoid is 1-connected iff all its etalisations are 1-connected. -/\ntheorem connected_iff_etalisation [C D : precategoroid] (P : quasi_prefunctoid C D) :\nπ₁ P = 0 ↔ '
        ]
        EXPECTED_IDS = torch.tensor([[    1, 32007, 822, 3349, 29918, 5464, 29918, 294, 18869, 29898,29879, 29901, 851, 29897, 1599, 851, 29901, 13, 1678, 9995, 29871, 32008, 13, 1678, 736, 1121, 13, 32009, 15941, 1661, 29899, 28599, 2687, 4890, 515, 263, 1347, 29889, 13, 13, 1678, 826, 3174, 29901, 13, 4706, 269, 29901, 450, 1347, 304, 3349, 1661, 29899, 28599, 2687, 4890, 515, 29889, 13, 13, 1678, 16969, 29901, 13, 4706, 450, 1347, 411, 1661, 29899, 28599, 2687, 4890, 6206, 29889, 13, 1678, 9995, 13, 1678, 1121, 353, 5124, 13, 1678, 363, 274, 297, 269, 29901, 13, 4706, 565, 4356, 29898, 29883, 29897, 529, 29871, 29896, 29906, 29947, 29901, 13, 9651, 1121, 4619, 274, 32010, 2]])
        # fmt: on
        self.assertEqual(processed_text_suffix_first, EXPECTED_TEXT)
        input_ids = tokenizer(self.PROMPTS[0], return_tensors="pt")["input_ids"]
        generated_ids = model.generate(input_ids.to(torch_device), max_new_tokens=128)
        torch.testing.assert_close(generated_ids, EXPECTED_IDS)

        EXPECTED_INFILLING = [
            '<s> <PRE> def remove_non_ascii(s: str) -> str:\n    """  <SUF>\n    return result\n <MID>Remove non-ASCII characters from a string.\n\n    Args:\n        s: The string to remove non-ASCII characters from.\n\n    Returns:\n        The string with non-ASCII characters removed.\n    """\n    result = ""\n    for c in s:\n        if ord(c) < 128:\n            result += c <EOT></s>'
        ]
        infilling = tokenizer.batch_decode(generated_ids)
        self.assertEqual(infilling, EXPECTED_INFILLING)
