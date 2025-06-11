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
"""Testing suite for the PyTorch openai model."""

import unittest

import torch
from packaging import version

from transformers import (
    AutoTokenizer,
    OpenAIMoeConfig,
    OpenAIMoeForCausalLM,
    OpenAIMoeModel,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


class OpenAIMoeModelTester:
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
        return OpenAIMoeConfig(
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
        model = OpenAIMoeModel(config=config)
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
class OpenAIMoeModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            OpenAIMoeModel,
            OpenAIMoeForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = OpenAIMoeForCausalLM if is_torch_available() else None

    def setUp(self):
        self.model_tester = OpenAIMoeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OpenAIMoeConfig, hidden_size=37)

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


@unittest.skip(reason="No available checkpoint for integration tests yet")
@require_torch_accelerator
class OpenAIMoeIntegrationTest(unittest.TestCase):
    def tearDown(self):
        # TODO (joao): automatic compilation, i.e. compilation when `cache_implementation="static"` is used, leaves
        # some memory allocated in the cache, which means some object is not being released properly. This causes some
        # unoptimal memory usage, e.g. after certain tests a 7B model in FP16 no longer fits in a 24GB GPU.
        # Investigate the root cause.
        cleanup(torch_device, gc_collect=False)

    @slow
    @require_read_token
    def test_openai_3_1_hard(self):
        """
        An integration test for openai 3.1. It tests against a long output to ensure the subtle numerical differences
        from openai 3.1.'s RoPE can be detected
        """
        # diff on `EXPECTED_TEXT`:
        # 2024-08-26: updating from torch 2.3.1 to 2.4.0 slightly changes the results.
        EXPECTED_TEXT = (
            "Tell me about the french revolution. The french revolution was a period of radical political and social "
            "upheaval in France that lasted from 1789 until 1799. It was a time of great change and upheaval, marked "
            "by the overthrow of the monarchy, the rise of the middle class, and the eventual establishment of the "
            "First French Republic.\nThe revolution began in 1789 with the Estates-General, a representative "
            "assembly that had not met since 1614. The Third Estate, which represented the common people, "
            "demanded greater representation and eventually broke away to form the National Assembly. This marked "
            "the beginning of the end of the absolute monarchy and the rise of the middle class.\n"
        )

        tokenizer = AutoTokenizer.from_pretrained("meta-openai/Meta-Openai-3.1-8B-Instruct")
        model = OpenAIMoeForCausalLM.from_pretrained(
            "meta-openai/Meta-Openai-3.1-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16
        )
        input_text = ["Tell me about the french revolution."]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)

    @slow
    @require_read_token
    def test_model_7b_logits_bf16(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = OpenAIMoeForCausalLM.from_pretrained(
            "meta-openai/Openai-2-7b-hf", device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))
        # Expected mean on dim = -1

        # fmt: off
        expected_means = Expectations(
            {
            ("xpu", 3): torch.tensor([[-6.5208, -4.1218, -4.9377, -3.2536,  0.8127, -2.9811,  1.2918, -3.3848]]),
            ("cuda", 7): torch.tensor([[-6.5061, -4.1147, -4.9669, -3.2038, 0.8069, -2.9694, 1.2864, -3.3786]]),
            ("cuda", 8): torch.tensor([[-6.5208, -4.1218, -4.9377, -3.2536,  0.8127, -2.9811,  1.2918, -3.3848]])
         })

        expected_mean = expected_means.get_expectation()
        self.assertTrue(
            torch.allclose(
                expected_mean.to(torch_device),
                out.logits.float().mean(-1),
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        expected_slices = Expectations(
            {
            ("xpu", 3): torch.tensor([[-12.5625,  -7.1250,  -0.6289,  -7.8750,  -6.9688,  -7.8125,  -6.5000, -7.4375,  -7.6562,  -6.9688,  -6.0312,  -7.0312,  -1.8203,   1.8750, -8.5000]]),
            ("cuda", 7): torch.tensor([[-12.5000, -7.0625, -0.6289, -7.8750, -6.9688, -7.8125, -6.4688, -7.4375, -7.6875, -6.9375, -6.0312, -7.0000, -1.8594, 1.8438, -8.5000]]),
            ("cuda", 8): torch.tensor([[-12.5625,  -7.1250,  -0.6289,  -7.8750,  -6.9688,  -7.8125,  -6.5000, -7.4375,  -7.6562,  -6.9688,  -6.0312,  -7.0312,  -1.8203,   1.8750, -8.5000]])
        })
        # fmt: on
        expected_slice = expected_slices.get_expectation()
        self.assertTrue(
            torch.allclose(
                expected_slice.to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @slow
    @require_read_token
    def test_model_7b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = OpenAIMoeForCausalLM.from_pretrained(
            "meta-openai/Openai-2-7b-hf", device_map="auto", torch_dtype=torch.float16
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        expected_means = Expectations(
          {
            ("xpu", 3): torch.tensor([[-6.6544, -4.1259, -4.9840, -3.2456,  0.8261, -3.0124,  1.2971, -3.3641]]),
            ("cuda", 7): torch.tensor([[-6.6420, -4.1227, -4.9809, -3.2041, 0.8261, -3.0052, 1.2957, -3.3648]]),
            ("cuda", 8): torch.tensor([[-6.6544, -4.1259, -4.9840, -3.2456,  0.8261, -3.0124,  1.2971, -3.3641]]),
        })

        expected_mean = expected_means.get_expectation()
        self.assertTrue(
            torch.allclose(
                expected_mean.to(torch_device),
                out.logits.float().mean(-1),
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        expected_slices = Expectations(
            {
              ("xpu", 3): torch.tensor([-12.8281,  -7.4609,  -0.4668,  -8.0703,  -7.2539,  -8.0078,  -6.4961, -7.7734,  -7.8516,  -7.0352,  -6.2188,  -7.1367,  -1.8564,   1.9922, -8.6328]),
              ("cuda", 7): torch.tensor([-12.8125, -7.3359, -0.4846, -8.0234, -7.2383, -7.9922, -6.4805, -7.7344, -7.8125, -7.0078, -6.1797, -7.1094, -1.8633, 1.9736, -8.6016]),
              ("cuda", 8): torch.tensor([-12.8281,  -7.4609,  -0.4668,  -8.0703,  -7.2539,  -8.0078,  -6.4961, -7.7734,  -7.8516,  -7.0352,  -6.2188,  -7.1367,  -1.8564,   1.9922, -8.6328])
        })
        # fmt: on

        expected_slice = expected_slices.get_expectation()
        self.assertTrue(
            torch.allclose(
                expected_slice.to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    @slow
    def test_model_7b_dola_generation(self):
        # ground truth text generated with dola_layers="low", repetition_penalty=1.2
        EXPECTED_TEXT_COMPLETION = (
            "Simply put, the theory of relativity states that 1) time and space are relative, and 2) the laws of "
            "physics are the same for all observers in uniform motion relative to one another.\n\nThe theory of "
            "relativity was developed by Albert Einstein in the early 20th century, and it revolutionized our "
            "understanding of space and time."
        )
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("meta-openai/Openai-2-7b-chat-hf")
        model = OpenAIMoeForCausalLM.from_pretrained(
            "meta-openai/Openai-2-7b-chat-hf", device_map="sequential", torch_dtype=torch.float16
        )
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=64, top_p=None, temperature=1, do_sample=False, dola_layers="low"
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_torch_accelerator
    @require_read_token
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        # Note on `EXPECTED_TEXT_COMPLETION`'s diff: the current value matches the original test if the original test
        # was changed to have a cache of 53 tokens (as opposed to 4096), on Ampere GPUs.
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that 1) the speed of light is constant in all inertial "
            "reference frames, and 2) the laws of physics are the same for all inertial reference frames.\nThe "
            "theory of relativ",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, "
            "my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my p",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("meta-openai/Openai-2-7b-hf", pad_token="</s>", padding_side="right")
        model = OpenAIMoeForCausalLM.from_pretrained(
            "meta-openai/Openai-2-7b-hf", device_map=torch_device, torch_dtype=torch.float16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        # Static Cache + compile (`generate()` internally compiles each decoding step when static cache is used)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)
