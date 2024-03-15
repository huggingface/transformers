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
""" Testing suite for the PyTorch InternLM2 model. """

import tempfile
import unittest

import pytest
from parameterized import parameterized

from transformers import InternLM2Config, is_torch_available, set_seed
from transformers.testing_utils import (
    require_bitsandbytes,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_gpu,
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
        InternLM2ForCausalLM,
        InternLM2ForSequenceClassification,
        InternLM2Model,
        InternLM2Tokenizer,
    )


class InternLM2ModelTester:
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
        return InternLM2Config(
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
        model = InternLM2Model(config=config)
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
        model = InternLM2Model(config)
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
        model = InternLM2ForCausalLM(config=config)
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
        model = InternLM2ForCausalLM(config=config)
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
class InternLM2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (InternLM2Model, InternLM2ForCausalLM, InternLM2ForSequenceClassification)
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (InternLM2ForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": InternLM2Model,
            "text-classification": InternLM2ForSequenceClassification,
            "text-generation": InternLM2ForCausalLM,
            "zero-shot": InternLM2ForSequenceClassification,
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
        self.model_tester = InternLM2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=InternLM2Config, hidden_size=37)

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

    def test_internlm2_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = InternLM2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_internlm2_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = InternLM2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_internlm2_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = InternLM2ForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    @unittest.skip("InternLM2 buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",)])
    def test_model_rope_scaling(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = InternLM2Model(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = InternLM2Model(config)
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
        model = InternLM2ForCausalLM.from_pretrained(
            " /mnt/inspurfs/xingshuhao/repo/huggingface_repo/internlm2-7b/",
            load_in_4bit=True,
            device_map={"": 0},
        )

        tokenizer = InternLM2Tokenizer.from_pretrained("/mnt/inspurfs/xingshuhao/repo/huggingface_repo/internlm2-7b/")

        texts = ["hi", "Hello this is a very long sentence"]

        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(0)

        output_native = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_native = tokenizer.batch_decode(output_native)

        model = InternLM2ForCausalLM.from_pretrained(
            "/mnt/inspurfs/xingshuhao/repo/huggingface_repo/internlm2-7b/", load_in_4bit=True, device_map={"": 0}, attn_implementation="flash_attention_2"
        )

        output_fa_2 = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_fa_2 = tokenizer.batch_decode(output_fa_2)

        self.assertListEqual(output_native, output_fa_2)

    @require_flash_attn
    @require_torch_gpu
    @slow
    def test_use_flash_attention_2_true(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            with tempfile.TemporaryDirectory() as tmp_dir:
                model = model_class(config)
                model.save_pretrained(tmp_dir)

                new_model = InternLM2ForCausalLM.from_pretrained(
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

    @unittest.skip("TODO")
    @parameterized.expand([(1, False), (1, True), (4, False)])
    def test_new_cache_format(self, num_beams, do_sample):
        pass

    @unittest.skip("TODO seems not supported now.")
    def test_torch_fx(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self._create_and_check_torch_fx_tracing(config, inputs_dict)

    @unittest.skip("TODO seems not supported now.")
    def test_torch_fx_output_loss(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self._create_and_check_torch_fx_tracing(config, inputs_dict, output_loss=True)


@require_torch
class InternLM2IntegrationTest(unittest.TestCase):

    @require_torch_gpu
    @slow
    def test_model_7b_logits(self):
        input_ids = [1, 9843, 346, 1226, 505, 395, 6576, 2049, 3514, 346]
        model = InternLM2ForCausalLM.from_pretrained("/mnt/inspurfs/xingshuhao/repo/huggingface_repo/internlm2-7b/", device_map="auto")
        out = model(torch.tensor([input_ids]).cuda()).logits
        out = out.cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-32.2436, 16.4387, 170.1120, 279.2027, 10.0092, -0.8451, 12.2949, 264.5546, 245.7853, 37.2034]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([
            -36.4534, -36.1433, -29.9111, -37.3252, -36.5012, -36.5528,
            -38.1726, -36.4655, -35.0742, -36.1164, -35.4240, -36.9896,
            -37.5202, -37.0715, -35.9560, -37.2607, -35.1858, -36.0715,
            -36.3952, -35.7251, -37.4291, -37.0344, -36.3430, -36.7009,
            -35.9350, -37.7175, -35.7385, -37.5043, -37.2610, -38.2035
        ])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)

    @require_torch_gpu
    @slow
    def test_model_20b_logits(self):
        input_ids = [1, 9843, 346, 1226, 505, 395, 6576, 2049, 3514, 346]
        model = InternLM2ForCausalLM.from_pretrained("/mnt/inspurfs/xingshuhao/repo/huggingface_repo/internlm2-20b/", device_map="auto")
        out = model(torch.tensor([input_ids]).cuda()).logits
        out = out.cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[0.3166, -0.7101, -0.1732, 3.7886, -0.2943, 2.5799, 2.0190, 3.3716, -0.2332, -0.5200]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([
            -1.8368946, -1.8470794, 1.4792165, -1.8458825, -1.8258708,
            -1.8340706, -1.8361576, -1.8298074, -1.8374101, -1.8282508,
            -1.8452281, -1.8541982, -1.8390937, -1.8511844, -1.827197,
            -1.8347681, -1.8478713, -1.8467027, -1.8257951, -1.8538132,
            -1.8401388, -1.8368533, -1.8511107, -1.85074, -1.8287741,
            -1.8403285, -1.834005, -1.8299413, -1.8222079, -1.8385823
        ])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)

    @require_torch_gpu
    @slow
    def test_model_chat_7b_logits(self):
        input_ids = [1, 9843, 346, 1226, 505, 395, 6576, 2049, 3514, 346]
        model = InternLM2ForCausalLM.from_pretrained("/mnt/inspurfs/xingshuhao/repo/huggingface_repo/internlm2-chat-7b/", device_map="auto")
        input_ids = torch.tensor([input_ids]).cuda()
        out = model(input_ids).logits
        out = out.cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[49.1164, 227.4299, 330.5746, 387.2284, 289.0935, 256.2116, 356.7249, 372.1883, 350.3138, 188.9861]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([
            42.0673, 42.8545, 51.3282, 43.2778, 41.8655, 43.0275, 43.1024,
            43.0847, 42.6107, 43.2770, 41.7264, 43.8559, 41.6691, 42.5620,
            42.2234, 42.3588, 43.1922, 40.5355, 41.9764, 43.1189, 41.5261,
            42.0903, 42.0941, 42.1555, 42.3925, 42.6166, 42.8195, 42.0701,
            43.1131, 43.4281
        ])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)

    @require_torch_gpu
    @slow
    def test_model_chat_20b_logits(self):
        input_ids = [1, 9843, 346, 1226, 505, 395, 6576, 2049, 3514, 346]
        model = InternLM2ForCausalLM.from_pretrained("/mnt/inspurfs/xingshuhao/repo/huggingface_repo/internlm2-chat-20b/", device_map="auto")
        out = model(torch.tensor([input_ids]).cuda()).logits
        out = out.cpu()

        EXPECTED_MEAN = torch.tensor(
            [[-2.7782e-01, 1.3930e-02, 8.2256e+00, 1.6851e+01, 1.3826e+01, 1.6862e+01, 1.5346e+01, 1.6360e+01, 1.4957e+01, 6.2845e+00]], dtype=torch.float32
        )
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        EXPECTED_SLICE = torch.tensor([
            -3.2300613, -3.2160933, 3.6487341, -3.2194245, -3.234827,
            -3.2349849, -3.2541847, -3.2256546, -3.2382178, -3.2377017,
            -3.2344182, -3.234288, -3.226959, -3.2184987, -3.2367303,
            -3.2436583, -3.2549813, -3.2267215, -3.2400918, -3.2260523,
            -3.2500873, -3.2475781, -3.2233024, -3.2291524, -3.2427764,
            -3.2538521, -3.2546685, -3.2423139, -3.2466452, -3.2423635
        ])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-5, rtol=1e-5)

    @require_torch_gpu
    @slow
    def test_model_7b_greedy_generation(self):
        EXPECTED_TEXT_COMPLETION = """Simply put, the theory of relativity states that “the laws of physics are the same for all non-accelerating observers, and that the speed of light in a vacuum is the same no matter the speed at which an observer travels.”\nWhat is the theory of relativity in simple terms?\nThe theory of relativity is a scientific theory that Albert Einstein"""
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = InternLM2Tokenizer.from_pretrained("/mnt/inspurfs/xingshuhao/repo/huggingface_repo/internlm2-7b/")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        model = InternLM2ForCausalLM.from_pretrained(
            "/mnt/inspurfs/xingshuhao/repo/huggingface_repo/internlm2-7b/", device_map="sequential", use_safetensors=False
        )

        # greedy generation outputs
        generated_ids = model.generate(input_ids.cuda(), max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
