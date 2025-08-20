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
"""Testing suite for the PyTorch Llasa model."""

import copy
import unittest

from parameterized import parameterized

# TODO remove when `XCodec2Model` is integrated into `transformers`
from xcodec2.modeling_xcodec2 import XCodec2Model

from transformers.models.llasa import LlasaConfig
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import LlasaForCausalLM, LlasaModel, LlasaProcessor, LlasaTokenizer


@require_torch
class LlasaModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=7,
        max_length=50,
        is_training=True,
        vocab_size=100,
        hidden_size=16,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=8,
        hidden_act="silu",
        eos_token_id=97,  # special tokens all occur after eos
        pad_token_id=98,
        bos_token_id=99,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.max_length = max_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def get_config(self):
        return LlasaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            hidden_act=self.hidden_act,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
        )

    def prepare_config_and_inputs(self) -> tuple[LlasaConfig, dict]:
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = input_ids.ne(self.pad_token_id)

        config = self.get_config()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self) -> tuple[LlasaConfig, dict]:
        return self.prepare_config_and_inputs()


@require_torch
class LlasaModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            LlasaModel,
            LlasaForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (LlasaForCausalLM,)
    pipeline_model_mapping = (
        {
            "feature-extraction": LlasaModel,
            # "text-generation": LlasaForCausalLM,
            "text-to-speech": LlasaForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = LlasaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlasaConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        """
        Overrides [ModelTesterMixin._prepare_for_class] to handle third input_ids dimension (namely adding labels).
        """
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (
                    self.model_tester.batch_size,
                    self.model_tester.seq_length,
                ),
                dtype=torch.long,
                device=torch_device,
            )

        return inputs_dict


"""
Gist to reproduce expected outputs for integration tests: https://gist.github.com/ebezzam/1863ec8eb7ec4afff02c26bdcb7691f9
"""

# fmt: off
input_text = "How much wood would a woodchuck chuck if a woodchuck could chuck speech tokens?"
EXPECTED_OUTPUT = {
    "bezzam/Llasa-1B": torch.tensor([168962, 172806, 193286, 188417, 184321, 167938, 167937, 173058, 189447,
        148418, 172641, 172642, 172978, 156321, 173042, 168578, 135208, 136200,
        150625, 165080, 131293, 151880, 131554, 134602, 188927, 187895, 190151,
        161262, 128750, 149197, 132828, 191964, 163363, 138639, 144335, 140242,
        155680, 179808, 186739, 185651, 186050, 186674, 186095, 150951, 151983,
        145749, 151789, 152213, 135384]),
    "bezzam/Llasa-3B": torch.tensor([152563, 140021, 135905, 140260, 172790, 167539, 189298, 193077, 173123,
        193378, 193714, 189495, 168769, 155437, 150284, 138253, 151564, 139297,
        167772, 152752, 169048, 131520, 147864, 128690, 145114, 165359, 157083,
        166250, 141662, 182586, 169250, 181566, 169361, 191175, 182706, 145882,
        147970, 167326, 128433, 138696, 171372, 193411, 131711, 155406, 144339,
        144598, 143457, 166944, 179244]),
    "bezzam/Llasa-8B": torch.tensor([152562, 135925, 152545, 140260, 168694, 168562, 193718, 193137, 173187,
        193463, 193479, 152385, 139293, 152584, 150625, 185240, 131485, 129481,
        129502, 182687, 165255, 185734, 161129, 128346, 152973, 136664, 192062,
        193279, 148035, 159818, 139158, 155424, 163872, 189493, 191047, 190851,
        174499, 167611, 147111, 146938, 166373, 149976, 137560, 190849, 178759,
        170784, 186678, 191047, 185911])
}
MAX_LENGTH = 50
TEMPERATURE = 0.8
# fmt: on


class LlasaForCausalLMIntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_accelerator
    @parameterized.expand([(model_repo,) for model_repo in EXPECTED_OUTPUT.keys()])
    def test_generate(self, model_repo):
        # load processor (tokenizer + audio codec)
        processor = LlasaProcessor(
            LlasaTokenizer.from_pretrained(model_repo),
            XCodec2Model.from_pretrained("HKUSTAudio/xcodec2").eval().to(torch_device),
        )
        # # -- TODO: use below when `XCodec2Model` integrated into `transformers`
        # processor = LlasaProcessor.from_pretrained(model_repo)

        model = LlasaForCausalLM.from_pretrained(model_repo)
        model.eval().to(torch_device)

        with torch.no_grad():
            encoded_text = processor(input_text).to(torch_device)
            outputs = model.generate(
                encoded_text["input_ids"],
                do_sample=False,
                max_new_tokens=MAX_LENGTH,
                top_p=1,
                temperature=TEMPERATURE,
            )
            generated_ids = outputs[0][encoded_text["input_ids"].shape[1] : -1]

        assert torch.equal(generated_ids, EXPECTED_OUTPUT[model_repo].to(torch_device))
