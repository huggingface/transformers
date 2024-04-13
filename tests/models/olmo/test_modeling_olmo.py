# coding=utf-8
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
""" Testing suite for the PyTorch OLMo model. """

import unittest

from parameterized import parameterized

from transformers import OLMoConfig, StaticCache, is_torch_available, logging, set_seed
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.testing_utils import (
    CaptureLogger,
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
        OLMoForCausalLM,
        OLMoModel,
    )


class OLMoModelTester:
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
        return OLMoConfig(
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
        model = OLMoModel(config=config)
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
        model = OLMoModel(config)
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
        model = OLMoForCausalLM(config=config)
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
        model = OLMoForCausalLM(config=config)
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
class OLMoModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (OLMoModel, OLMoForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (OLMoForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": OLMoModel,
            "text-generation": OLMoForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    fx_compatible = False

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    def setUp(self):
        self.model_tester = OLMoModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OLMoConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip("OLMo does not support head pruning.")
    def test_headmasking(self):
        pass

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip("OLMo buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",)])
    def test_model_rope_scaling(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = OLMoModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = OLMoModel(config)
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

    @unittest.skip("TODO @gante fix this for OLMo")
    @parameterized.expand([(1, False), (1, True), (4, False)])
    def test_new_cache_format(self, num_beams, do_sample):
        pass


@require_torch
class OLMoIntegrationTest(unittest.TestCase):
    @slow
    def test_model_1b_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = OLMoForCausalLM.from_pretrained("allenai/OLMo-1B-hf")
        out = model(torch.tensor(input_ids)).logits
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[2.2869, 0.3315, 0.9876, 1.4146, 1.8804, 2.0430, 1.7055, 1.2065]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([2.5551, -1.1230, 11.0510, 12.4977, 7.9651, 7.2342, 6.1885, 7.8340, 9.9847, 12.6695, 12.2345, 10.7970, 8.4749, 14.2483, 12.9588, 13.9233, 11.0496, 5.5749, 7.4466, 7.7914, 6.8440, 5.8951, 4.8180, 4.1935, 4.5216, 4.7256, 3.9553, 12.2870, 12.4990, 8.1591])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-2, rtol=1e-2)

    @slow
    def test_model_7b_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B-hf")
        out = model(torch.tensor(input_ids)).logits
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[0.0266, -0.0012, -0.0589, -0.0868, 0.0284, 0.0609, 0.0836, 0.0552]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-1.5364, -1.5505, 6.6963, 4.6085, -0.5580, 0.3295, 2.0886, 4.2162, 3.3459, 4.6850, 6.9190, 2.5602, 2.6904, 9.5714, 9.6181, 9.7108, 6.0352, 1.0118, 3.6572, 3.8474, 1.9998, 1.7586, 1.6130, 0.8734, 0.5737, 1.2120, 0.2762, 5.5266, 5.8129, 1.8287])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-2, rtol=1e-2)

    @unittest.skip("Logits are not yet correct, will update!")
    @slow
    def test_model_7b_twin_2t_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B-Twin-2T-hf")
        out = model(torch.tensor(input_ids)).logits
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-0.3636, -0.3825, -0.4800, -0.3696, -0.8388, -0.9737, -0.9849, -0.8356]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-2.0833, -1.9234, 8.7312, 7.8049, 1.0372, 0.8941, 3.1548, 1.8502, 5.5511, 5.5793, 8.1166, 4.5906, 1.8691, 11.6377, 8.9858, 11.6447, 7.4549, 1.4725, 2.8399, 2.7568, 1.4011, 1.6958, 0.5572, 0.5231, 0.3068, 0.5364, 0.6769, 7.9636, 8.2379, 1.7950])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-2, rtol=1e-2)

    @unittest.skip("Model is curently gated")
    @slow
    def test_model_7b_greedy_generation(self):
        EXPECTED_TEXT_COMPLETION = """Simply put, the theory of relativity states that 1) the laws of physics are the same everywhere in the universe and 2) the passage of time and the length of objects can vary depending on the observer\'s frame of reference.\n\nThe first part of the theory, that the laws of physics are the same everywhere, is known as the "princi"""
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        model = OLMoForCausalLM.from_pretrained("allenai/OLMo-7B-hf", device_map="sequential", use_safetensors=False)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_torch_gpu
    @require_read_token
    def test_compile_static_cache(self):
        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that 1) the speed of light is constant, 2) the speed of light is the same for all observers, and 3) the laws of physics are the same for all observers.",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my p",
        ]
        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf", pad_token="</s>", padding_side="right")
        model = OLMoForCausalLM.from_pretrained("allenai/OLMo-1B-hf", device_map="sequential")
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
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
