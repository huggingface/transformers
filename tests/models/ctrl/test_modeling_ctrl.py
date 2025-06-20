# Copyright 2018 Salesforce and HuggingFace Inc. team.
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

from transformers import CTRLConfig, is_torch_available
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        CTRLForSequenceClassification,
        CTRLLMHeadModel,
        CTRLModel,
    )


class CTRLModelTester(CausalLMModelTester):
    config_class = CTRLConfig
    if is_torch_available():
        base_model_class = CTRLModel
        causal_lm_class = CTRLLMHeadModel
        sequence_classification_class = CTRLForSequenceClassification

    def __init__(self, parent, num_hidden_layers=1, **kwargs):
        super().__init__(parent=parent, num_hidden_layers=num_hidden_layers, **kwargs)


@require_torch
class CTRLModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (CTRLModel, CTRLLMHeadModel, CTRLForSequenceClassification) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": CTRLModel,
            "text-classification": CTRLForSequenceClassification,
            "text-generation": CTRLLMHeadModel,
            "zero-shot": CTRLForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = CTRLModelTester

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
        if pipeline_test_case_name == "ZeroShotClassificationPipelineTests":
            # Get `tokenizer does not have a padding token` error for both fast/slow tokenizers.
            # `CTRLConfig` was never used in pipeline tests, either because of a missing checkpoint or because a tiny
            # config could not be created.
            return True

        return False


@require_torch
class CTRLModelLanguageGenerationTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_lm_generate_ctrl(self):
        model = CTRLLMHeadModel.from_pretrained("Salesforce/ctrl")
        model.to(torch_device)
        input_ids = torch.tensor(
            [[11859, 0, 1611, 8]], dtype=torch.long, device=torch_device
        )  # Legal the president is
        expected_output_ids = [
            11859,
            0,
            1611,
            8,
            5,
            150,
            26449,
            2,
            19,
            348,
            469,
            3,
            2595,
            48,
            20740,
            246533,
            246533,
            19,
            30,
            5,
        ]  # Legal the president is a good guy and I don't want to lose my job. \n \n I have a

        output_ids = model.generate(input_ids, do_sample=False)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
