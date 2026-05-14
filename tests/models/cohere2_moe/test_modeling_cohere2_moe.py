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
"""Testing suite for the PyTorch Cohere2Moe model"""

import unittest

from parameterized import parameterized
from pytest import mark

from transformers import AutoConfig, AutoTokenizer, Cohere2MoeConfig, Cohere2VisionForConditionalGeneration, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    is_flash_attn_2_available,
    is_torch_xpu_available,
    is_kernels_available,
    require_flash_attn,
    require_torch,
    require_torch_large_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        Cohere2MoeForCausalLM,
        Cohere2MoeModel,
    )

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


class Cohere2MoeModelTester:
    config_class = Cohere2MoeConfig
    if is_torch_available():
        model_class = Cohere2MoeModel
        for_causal_lm_class = Cohere2MoeForCausalLM

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
        head_dim=8,
        intermediate_size=37,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=0,
        sliding_window=512,
        # One full-attention layer so the hybrid cache is exercised.
        layer_types=None,
        pad_token_id=0,
        scope=None,
        num_labels=3,
        num_choices=4,
        type_sequence_label_size=2,
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
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.sliding_window = sliding_window
        self.layer_types = layer_types if layer_types is not None else ["full_attention", "sliding_attention"]
        self.pad_token_id = pad_token_id
        self.scope = scope
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.type_sequence_label_size = type_sequence_label_size

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, None, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return self.config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            num_shared_experts=self.num_shared_experts,
            sliding_window=self.sliding_window,
            layer_types=self.layer_types,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = self.model_class(config=config)
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
class Cohere2MoeModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Cohere2MoeModel, Cohere2MoeForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Cohere2MoeModel,
            "text-generation": Cohere2MoeForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    _is_stateful = True

    # Avoid edge cases with causal_mask buffer during CPU offload
    model_split_percents = [0.5, 0.7, 0.8]

    def setUp(self):
        self.model_tester = Cohere2MoeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Cohere2MoeConfig, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)


@slow
@require_torch_large_accelerator
class Cohere2MoeIntegrationTest(unittest.TestCase):
    """Integration tests for the cohere2moe text backbone via the Command A+ Model.

    Cohere2VisionForConditionalGeneration wraps the cohere2moe language model; running it with
    text-only inputs exercises the text backbone without requiring a separate text-only checkpoint.
    """

    model_id = "/root/repos/moe/engines/mhlv2_bf16_clean"
    input_text = ["Hello I am doing", "Hi today"]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_model(self, dtype, attn_implementation="eager", text_config_overrides=None):
        """Load the vision model (cohere2moe backbone) distributed across all available GPUs.

        text_config_overrides: optional dict of attributes to set on config.text_config before loading
        (e.g. {"sliding_window": 1024}).
        """
        if text_config_overrides:
            config = AutoConfig.from_pretrained(self.model_id)
            for k, v in text_config_overrides.items():
                setattr(config.text_config, k, v)
        else:
            config = None
        kwargs = dict(torch_dtype=dtype, attn_implementation=attn_implementation, device_map="auto")
        if config is not None:
            kwargs["config"] = config
        return Cohere2VisionForConditionalGeneration.from_pretrained(self.model_id, **kwargs).eval()

    def test_model_bf16(self):
        EXPECTED_TEXTS = [
            '<BOS_TOKEN>Hello I am doing a project on the history of the internet. I am trying to ARexx script a program that',
            '<PAD><PAD><BOS_TOKEN>Hi today we are going to discuss about the concept of "The law of karma" and "Reincarnation',
        ]

        model = self._load_model(torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to("cuda:0")

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_fp16(self):
        # fmt: off
        EXPECTED_TEXTS = Expectations(
            {
                (None, None): [
                    '<BOS_TOKEN>Hello I am doing a project on the history of the internet. I am trying to ARexx script a program that',
                    '<PAD><PAD><BOS_TOKEN>Hi today we are going to discuss about the concept of "Self-Confidence". Self-confidence is a term that',
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        # fmt: on

        model = self._load_model(torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to("cuda:0")

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_flash_attn
    @mark.flash_attn_test
    def test_model_flash_attn(self):
        # fmt: off
        EXPECTED_TEXTS = [
            '<BOS_TOKEN>Hello I am doing a project on the history of the internet. I am trying to ARexx script a program that will display a comment and then a progress bar that moves across the2009-09-30\n\nHello, I am doing a project on the history of the internet. I am trying to ARexx script a program that will display a comment and then a progress bar that moves across the screen. I have a question about the "wait" command. I have been using "wait 1"',
            '<PAD><PAD><BOS_TOKEN>Hi today we are going to discuss about the concept of "Self-Confidence". Self-confidence is a term that many people use to describe a state of mind where one feels confident in their abilities, decisions, and actions. It\'s a feeling of trust in one\'s own judgment and abilities. Self-confidence is not about being arrogant or overconfident; it\'s about having a realistic and positive view of oneself and one\'s capabilities.\n\nSelf-confidence can be developed and improved over time through various practices such as setting and achieving goals, learning',
        ]
        # fmt: on

        model = self._load_model(torch.float16, attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to("cuda:0")

        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @parameterized.expand([("flash_attention_2",), ("sdpa",), ("eager",)])
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Verify that generation beyond the sliding window produces coherent output
        with all supported attention backends.
        """
        if (
            attn_implementation == "flash_attention_2"
            and not is_flash_attn_2_available()
            and not (is_torch_xpu_available() and is_kernels_available())
        ):
            self.skipTest("FlashAttention2 is required for this test.")

        EXPECTED_COMPLETIONS = [
            " but I think it's a nice place. This is a nice place. This is a nice place.",
            ", green, yellow, orange, purple, pink, brown, black, white.\n\nWe need to",
        ]

        input_text = [
            "This is a nice place. " * 200 + "I really enjoy the scenery,",
            "A list of colors: red, blue",
        ]
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to("cuda:0")

        model = self._load_model(torch.float16, attn_implementation=attn_implementation, text_config_overrides={"sliding_window": 1024})

        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.text_config.sliding_window)

        out = model.generate(**inputs, max_new_tokens=20)[:, input_size:]
        output_text = tokenizer.batch_decode(out)

        self.assertEqual(output_text, EXPECTED_COMPLETIONS)
