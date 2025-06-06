# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import XLNetConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        XLNetForMultipleChoice,
        XLNetForQuestionAnswering,
        XLNetForQuestionAnsweringSimple,
        XLNetForSequenceClassification,
        XLNetForTokenClassification,
        XLNetLMHeadModel,
        XLNetModel,
    )


class XLNetModelTester(CausalLMModelTester):
    config_class = XLNetConfig
    if is_torch_available():
        base_model_class = XLNetModel
        causal_lm_class = XLNetLMHeadModel
        question_answering_class = XLNetForQuestionAnsweringSimple
        sequence_classification_class = XLNetForSequenceClassification
        token_classification_class = XLNetForTokenClassification


@require_torch
class XLNetModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            XLNetModel,
            XLNetLMHeadModel,
            XLNetForTokenClassification,
            XLNetForSequenceClassification,
            XLNetForQuestionAnswering,
            XLNetForQuestionAnsweringSimple,
            XLNetForMultipleChoice,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": XLNetModel,
            "question-answering": XLNetForQuestionAnsweringSimple,
            "text-classification": XLNetForSequenceClassification,
            "text-generation": XLNetLMHeadModel,
            "token-classification": XLNetForTokenClassification,
            "zero-shot": XLNetForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = XLNetModelTester

    @unittest.skip(reason="xlnet cannot keep gradients in attentions or hidden states")
    def test_retain_grad_hidden_states_attentions(self):
        return


@require_torch
class XLNetModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_xlnet_base_cased(self):
        model = XLNetLMHeadModel.from_pretrained("xlnet/xlnet-base-cased")
        model.to(torch_device)
        # fmt: off
        input_ids = torch.tensor(
            [
                [
                    67, 2840, 19, 18, 1484, 20, 965, 29077, 8719, 1273, 21, 45, 273, 17, 10, 15048, 28, 27511, 21, 4185, 11, 41, 2444, 9, 32, 1025, 20, 8719, 26, 23, 673, 966, 19, 29077, 20643, 27511, 20822, 20643, 19, 17, 6616, 17511, 18, 8978, 20, 18, 777, 9, 19233, 1527, 17669, 19, 24, 673, 17, 28756, 150, 12943, 4354, 153, 27, 442, 37, 45, 668, 21, 24, 256, 20, 416, 22, 2771, 4901, 9, 12943, 4354, 153, 51, 24, 3004, 21, 28142, 23, 65, 20, 18, 416, 34, 24, 2958, 22947, 9, 1177, 45, 668, 3097, 13768, 23, 103, 28, 441, 148, 48, 20522, 19, 12943, 4354, 153, 12860, 34, 18, 326, 27, 17492, 684, 21, 6709, 9, 8585, 123, 266, 19, 12943, 4354, 153, 6872, 24, 3004, 20, 18, 9225, 2198, 19, 12717, 103, 22, 401, 24, 6348, 9, 12943, 4354, 153, 1068, 2768, 2286, 19, 33, 104, 19, 176, 24, 9313, 19, 20086, 28, 45, 10292, 9, 4, 3,
                ]
            ],
            dtype=torch.long,
            device=torch_device,
        )
        # fmt: on
        #  In 1991, the remains of Russian Tsar Nicholas II and his family
        #  (except for Alexei and Maria) are discovered.
        #  The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
        #  remainder of the story. 1883 Western Siberia,
        #  a young Grigori Rasputin is asked by his father and a group of men to perform magic.
        #  Rasputin has a vision and denounces one of the men as a horse thief. Although his
        #  father initially slaps him for making such an accusation, Rasputin watches as the
        #  man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
        #  the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
        #  with people, even a bishop, begging for his blessing. """

        # fmt: off
        expected_output_ids = [
            67, 2840, 19, 18, 1484, 20, 965, 29077, 8719, 1273, 21, 45, 273, 17, 10, 15048, 28, 27511, 21, 4185, 11, 41, 2444, 9, 32, 1025, 20, 8719, 26, 23, 673, 966, 19, 29077, 20643, 27511, 20822, 20643, 19, 17, 6616, 17511, 18, 8978, 20, 18, 777, 9, 19233, 1527, 17669, 19, 24, 673, 17, 28756, 150, 12943, 4354, 153, 27, 442, 37, 45, 668, 21, 24, 256, 20, 416, 22, 2771, 4901, 9, 12943, 4354, 153, 51, 24, 3004, 21, 28142, 23, 65, 20, 18, 416, 34, 24, 2958, 22947, 9, 1177, 45, 668, 3097, 13768, 23, 103, 28, 441, 148, 48, 20522, 19, 12943, 4354, 153, 12860, 34, 18, 326, 27, 17492, 684, 21, 6709, 9, 8585, 123, 266, 19, 12943, 4354, 153, 6872, 24, 3004, 20, 18, 9225, 2198, 19, 12717, 103, 22, 401, 24, 6348, 9, 12943, 4354, 153, 1068, 2768, 2286, 19, 33, 104, 19, 176, 24, 9313, 19, 20086, 28, 45, 10292, 9, 4, 3, 19, 12943, 4354, 153, 27, 442, 22, 2771, 4901, 9, 69, 27, 442, 22, 2771, 24, 11335, 20, 18, 9225, 2198, 9, 69, 27, 442, 22, 2771, 24, 11335, 20, 18, 9225, 2198, 9, 69, 27, 442, 22, 2771,
        ]
        # fmt: on
        #  In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria)
        #  are discovered. The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich,
        #  narrates the remainder of the story. 1883 Western Siberia, a young Grigori Rasputin
        #  is asked by his father and a group of men to perform magic. Rasputin has a vision and
        #  denounces one of the men as a horse thief. Although his father initially slaps
        #  him for making such an accusation, Rasputin watches as the man is chased outside and beaten.
        #  Twenty years later, Rasputin sees a vision of the Virgin Mary, prompting him to become a priest.
        #  Rasputin quickly becomes famous, with people, even a bishop, begging for his blessing.
        #  <sep><cls>, Rasputin is asked to perform magic. He is asked to perform a ritual of the Virgin Mary.
        #  He is asked to perform a ritual of the Virgin Mary. He is asked to perform

        output_ids = model.generate(input_ids, max_length=200, do_sample=False)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
