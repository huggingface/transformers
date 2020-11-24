# coding=utf-8
# Copyright 2020 HuggingFace Inc. team.
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
import unittest

from tests.test_configuration_common import ConfigTester
from tests.test_modeling_tf_bart import TFBartModelTester
from tests.test_modeling_tf_common import TFModelTesterMixin
from transformers import (
    BlenderbotConfig,
    BlenderbotSmallTokenizer,
    TFAutoModelForSeq2SeqLM,
    TFBlenderbotForConditionalGeneration,
    is_tf_available,
)
from transformers.file_utils import cached_property
from transformers.testing_utils import is_pt_tf_cross_test, require_tf, require_tokenizers, slow


class TFBlenderbotModelTester(TFBartModelTester):
    config_updates = dict(
        normalize_before=True,
        static_position_embeddings=True,
        do_blenderbot_90_layernorm=True,
        normalize_embeddings=True,
    )
    config_cls = BlenderbotConfig


@require_tf
class TFBlenderbotModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFBlenderbotForConditionalGeneration,) if is_tf_available() else ()
    all_generative_model_classes = (TFBlenderbotForConditionalGeneration,) if is_tf_available() else ()
    is_encoder_decoder = True
    test_pruning = False

    def setUp(self):
        self.model_tester = TFBlenderbotModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlenderbotConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_inputs_embeds(self):
        # inputs_embeds not supported
        pass

    def test_saved_model_with_hidden_states_output(self):
        # Should be uncommented during patrick TF refactor
        pass

    def test_saved_model_with_attentions_output(self):
        # Should be uncommented during patrick TF refactor
        pass


@is_pt_tf_cross_test
@require_tokenizers
class TFBlenderbot90MIntegrationTests(unittest.TestCase):
    src_text = [
        "Social anxiety\nWow, I am never shy. Do you have anxiety?\nYes. I end up sweating and blushing and feel like   i'm going to throw up.\nand why is that?"
    ]
    model_name = "facebook/blenderbot-90M"

    @cached_property
    def tokenizer(self):
        return BlenderbotSmallTokenizer.from_pretrained(self.model_name)

    @cached_property
    def model(self):
        model = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_name, from_pt=True)
        return model

    @slow
    def test_90_generation_from_long_input(self):
        model_inputs = self.tokenizer(self.src_text, return_tensors="tf")
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            num_beams=2,
            use_cache=True,
        )
        generated_words = self.tokenizer.batch_decode(generated_ids.numpy(), skip_special_tokens=True)[0]
        assert generated_words in (
            "i don't know. i just feel like i'm going to throw up. it's not fun.",
            "i'm not sure. i just feel like i've been feeling like i have to be in a certain place",
            "i'm not sure. i just feel like i've been in a bad situation.",
        )
