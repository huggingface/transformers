# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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


import math
import unittest

from parameterized import parameterized

from transformers import AutoTokenizer, Mamba2Config, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Mamba2ForCausalLM,
        Mamba2Model,
    )
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_0
else:
    is_torch_greater_or_equal_than_2_0 = False


class Mamba2ModelTester:
    config_classs = Mamba2Config
    model_class = Mamba2Model
    for_causal_lm = Mamba2ForCausalLM

    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=32,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        tie_word_embeddings=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1
        self.tie_word_embeddings = tie_word_embeddings


@unittest.skipIf(
    not is_torch_greater_or_equal_than_2_0, reason="See https://github.com/huggingface/transformers/pull/24204"
)
@require_torch
class Mamba2ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Mamba2Model, Mamba2ForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (Mamba2ForCausalLM,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = Mamba2ModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Mamba2Config, n_embd=37, common_properties=["hidden_size", "num_hidden_layers"]
        )

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for name, param in model.named_parameters():
                if "dt_proj.bias" in name:
                    dt = torch.exp(
                        torch.tensor([0, 1]) * (math.log(config.time_step_max) - math.log(config.time_step_min))
                        + math.log(config.time_step_min)
                    ).clamp(min=config.time_step_floor)
                    inv_dt = dt + torch.log(-torch.expm1(-dt))
                    if param.requires_grad:
                        self.assertTrue(param.data.max().item() <= inv_dt[1])
                        self.assertTrue(param.data.min().item() >= inv_dt[0])
                elif "A_log" in name:
                    A = torch.arange(1, config.state_size + 1, dtype=torch.float32)[None, :]
                    self.assertTrue(torch.allclose(param.data, torch.log(A), atol=1e-5, rtol=1e-5))
                elif "D" in name:
                    if param.requires_grad:
                        # check if it's a ones like
                        self.assertTrue(torch.allclose(param.data, torch.ones_like(param.data), atol=1e-5, rtol=1e-5))


@require_torch
class Mamba2IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "state-spaces/mamba2-2.8b-hf"  # FIXME add correct model id here
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    @parameterized.expand([(torch_device,), ("cpu",)])
    def test_simple_generate(self, device):
        tokenizer = AutoTokenizer.from_pretrained("mistralai/mamba-codestral-7B-v0.1")
        tokenizer.pad_token = tokenizer.eos_token

        model = Mamba2ForCausalLM.from_pretrained("mistralai/mamba-codestral-7B-v0.1", torch_dtype=torch.float16)
        model.to(device)
        input_ids = tokenizer("[INST]Write a hello world program in C++.[/INST]", return_tensors="pt")["input_ids"].to(
            device
        )

        out = model.generate(input_ids, do_sample=False, use_cache=True, max_new_tokens=10)
        output_sentence = tokenizer.decode(out[0, :])

        ground_truth_sentence = """Sure, here is a simple "Hello, World!" program in C++:
                                ```cpp
                                #include <iostream>

                                int main() {
                                    std::cout << "Hello, World!";
                                    return 0;
                                }
                                ```

                                This program will output the text "Hello, World!" when run. Let me break it down for you:

                                - `#include <iostream>`: This is a preprocessor directive that tells the compiler to include the iostream standard library.
                                - `int main()`: This is the main function where the program starts executing.
                                - `std::cout << "Hello, World!";`: This line is where the magic happens. `std::cout` is an object in the standard library that is used for outputting text to the console. The text "Hello, World!" is what we want to output.
                                - `return 0;`: This line indicates that the program has run successfully. In Unix-like operating systems, the convention is that a return value of 0 indicates success, while a non-zero value indicates failure.
                                """
        # TODO finish up integration test for all cases (cpu, gpu, kernels, no kernels)
        self.assertEqual(output_sentence, ground_truth_sentence)
