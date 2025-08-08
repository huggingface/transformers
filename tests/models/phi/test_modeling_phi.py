# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""Testing suite for the PyTorch Phi model."""

import unittest

from transformers import PhiConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        PhiForCausalLM,
        PhiForSequenceClassification,
        PhiForTokenClassification,
        PhiModel,
    )
    from transformers.models.phi.modeling_phi import PhiRotaryEmbedding


class PhiModelTester(CausalLMModelTester):
    config_class = PhiConfig
    if is_torch_available():
        base_model_class = PhiModel
        causal_lm_class = PhiForCausalLM
        sequence_class = PhiForSequenceClassification
        token_class = PhiForTokenClassification


@require_torch
class PhiModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (PhiModel, PhiForCausalLM, PhiForSequenceClassification, PhiForTokenClassification)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": PhiModel,
            "text-classification": PhiForSequenceClassification,
            "token-classification": PhiForTokenClassification,
            "text-generation": PhiForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    model_tester_class = PhiModelTester
    rotary_embedding_layer = PhiRotaryEmbedding

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79292/workflows/fa2ba644-8953-44a6-8f67-ccd69ca6a476/jobs/1012905
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True


@slow
@require_torch
class PhiIntegrationTest(unittest.TestCase):
    def test_model_phi_1_logits(self):
        input_ids = {
            "input_ids": torch.tensor(
                [[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], dtype=torch.long, device=torch_device
            )
        }

        model = PhiForCausalLM.from_pretrained("microsoft/phi-1").to(torch_device)
        model.eval()

        output = model(**input_ids).logits

        EXPECTED_OUTPUT = torch.tensor([[2.2671,  6.7684, -2.0107, -1.2440, -1.5335, -2.3828,  6.9186,  6.4245, 3.1548,  0.9998,  0.0760,  4.4653,  4.9857,  4.2956,  1.2308, -1.4178, 0.1361,  0.5191, -0.5699, -2.2201, -3.0750, -3.9600, -4.5936, -3.7394, -2.7777,  6.1874, -0.4148, -1.5684, -0.5967,  0.2395], [1.7004,  4.0383,  0.0546,  0.4530, -0.3619, -0.9021,  1.8355,  1.3587, 1.2406,  2.5775, -0.8834,  5.1910,  4.2565,  4.1406,  3.0752, -0.9099, 1.1595,  0.0264,  0.3243, -1.1803, -1.3945, -2.1406, -3.9939, -1.4438, -2.9546,  3.9204,  1.0851, -1.0598, -1.7819, -0.4827]]).to(torch_device)  # fmt: skip

        torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2, :30], rtol=1e-4, atol=1e-4)

    def test_model_phi_1_5_logits(self):
        input_ids = {
            "input_ids": torch.tensor(
                [[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], dtype=torch.long, device=torch_device
            )
        }

        model = PhiForCausalLM.from_pretrained("microsoft/phi-1_5").to(torch_device)
        model.eval()

        output = model(**input_ids).logits

        EXPECTED_OUTPUT = torch.tensor([[12.2922, 13.3507,  8.6963,  9.1355,  9.3502,  9.2667, 14.2027, 13.1363, 13.5446, 11.1337,  9.9279, 16.7195, 13.0768, 14.9141, 11.9965,  8.0233, 10.3129, 10.6118, 10.0204,  9.3827,  8.8344,  8.2806,  8.0153,  8.0540, 7.0964, 16.5743, 11.1256,  9.6987, 11.4770, 10.5440], [12.3323, 14.6050,  8.9986,  8.1580,  9.5654,  6.6728, 12.5966, 12.6662, 12.2784, 11.7522,  8.2039, 16.3102, 11.2203, 13.6088, 12.0125,  9.1021, 9.8216, 10.0987,  9.0926,  8.4260,  8.8009,  7.6547,  6.8075,  7.7881, 7.4501, 15.7451, 10.5053,  8.3129, 10.0027,  9.2612]]).to(torch_device)  # fmt: skip

        torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2, :30], rtol=1e-4, atol=1e-4)

    def test_model_phi_2_logits(self):
        input_ids = {
            "input_ids": torch.tensor(
                [[1212, 318, 281, 1672, 2643, 290, 428, 318, 257, 1332]], dtype=torch.long, device=torch_device
            )
        }

        model = PhiForCausalLM.from_pretrained("microsoft/phi-2").to(torch_device)
        model.eval()

        output = model(**input_ids).logits

        EXPECTED_OUTPUT = torch.tensor([[6.4830,  6.1644,  3.4055,  2.2848,  5.4654,  2.8360,  5.5975,  5.5391, 7.3101,  4.2498,  2.5913, 10.3885,  6.4359,  8.7982,  5.6534,  0.5150, 2.7498,  3.1930,  2.4334,  1.7781,  1.5613,  1.3067,  0.8291,  0.5633, 0.6522,  9.8191,  5.5771,  2.7987,  4.2845,  3.7030], [6.0642,  7.8242,  3.4634,  1.9259,  4.3169,  2.0913,  6.0446,  3.6804, 6.6736,  4.0727,  2.1791, 11.4139,  5.6795,  7.5652,  6.2039,  2.7174, 4.3266,  3.6930,  2.8058,  2.6721,  2.3047,  2.0848,  2.0972,  2.0441, 1.3160,  9.2085,  4.5557,  3.0296,  2.6045,  2.4059]]).to(torch_device)  # fmt: skip

        torch.testing.assert_close(EXPECTED_OUTPUT, output[0, :2, :30], rtol=1e-3, atol=1e-3)

    def test_phi_2_generation(self):
        model = PhiForCausalLM.from_pretrained("microsoft/phi-2")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

        inputs = tokenizer(
            "Can you help me write a formal email to a potential business partner proposing a joint venture?",
            return_tensors="pt",
            return_attention_mask=False,
        )

        outputs = model.generate(**inputs, max_new_tokens=30)
        output_text = tokenizer.batch_decode(outputs)

        EXPECTED_OUTPUT = [
            "Can you help me write a formal email to a potential business partner proposing a joint venture?\nInput: Company A: ABC Inc.\nCompany B: XYZ Ltd.\nJoint Venture: A new online platform for e-commerce"
        ]

        self.assertListEqual(output_text, EXPECTED_OUTPUT)
