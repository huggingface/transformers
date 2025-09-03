# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Falcon model."""

import unittest

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    FalconConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    require_bitsandbytes,
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        FalconForCausalLM,
        FalconForQuestionAnswering,
        FalconForSequenceClassification,
        FalconForTokenClassification,
        FalconModel,
    )


class FalconModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = FalconConfig
        base_model_class = FalconModel
        causal_lm_class = FalconForCausalLM
        sequence_class = FalconForSequenceClassification
        token_class = FalconForTokenClassification

    def __init__(self, parent, new_decoder_architecture=True):
        super().__init__(parent)
        self.new_decoder_architecture = new_decoder_architecture


@require_torch
class FalconModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = FalconModelTester
    all_model_classes = (
        (
            FalconModel,
            FalconForCausalLM,
            FalconForSequenceClassification,
            FalconForTokenClassification,
            FalconForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": FalconModel,
            "text-classification": FalconForSequenceClassification,
            "token-classification": FalconForTokenClassification,
            "text-generation": FalconForCausalLM,
            "zero-shot": FalconForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
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


@require_torch
class FalconLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_falcon(self):
        tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
        model = FalconForCausalLM.from_pretrained("Rocketknight1/falcon-rw-1b")
        model.eval()
        model.to(torch_device)
        inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)

        EXPECTED_OUTPUT = (
            "My favorite food is pizza. I love it so much that I have a pizza party every year for my birthday."
        )

        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=19)
        output_str = tokenizer.batch_decode(output_ids)[0]

        self.assertEqual(output_str, EXPECTED_OUTPUT)

    @slow
    @require_bitsandbytes
    def test_lm_generate_falcon_11b(self):
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-11B", padding_side="left")
        model = FalconForCausalLM.from_pretrained(
            "tiiuae/falcon-11B", device_map={"": torch_device}, load_in_8bit=True
        )
        model.eval()
        inputs = tokenizer(
            "Two roads diverged in a yellow wood,", return_tensors="pt", return_token_type_ids=False
        ).to(torch_device)

        EXPECTED_OUTPUT = "Two roads diverged in a yellow wood,\nAnd sorry I could not travel both\n"

        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=9)
        output_str = tokenizer.batch_decode(output_ids)[0]

        self.assertEqual(output_str, EXPECTED_OUTPUT)

    @slow
    def test_lm_generation_big_models(self):
        # The big models are way too big for the CI, so we use tiny random models that resemble their
        # architectures but with much smaller and fewer layers
        for repo in ["Rocketknight1/tiny-random-falcon-7b", "Rocketknight1/tiny-random-falcon-40b"]:
            tokenizer = AutoTokenizer.from_pretrained(repo)
            model = FalconForCausalLM.from_pretrained(repo)
            model.eval()
            model.to(torch_device)
            inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)

            # We just test that these run without errors - the models are randomly initialized
            # and so the actual text outputs will be garbage
            model.generate(**inputs, do_sample=False, max_new_tokens=4)
            model.generate(**inputs, do_sample=True, max_new_tokens=4)
            model.generate(**inputs, num_beams=2, max_new_tokens=4)

    @slow
    def test_lm_generation_use_cache(self):
        # The big models are way too big for the CI, so we use tiny random models that resemble their
        # architectures but with much smaller and fewer layers
        with torch.no_grad():
            for repo in [
                "Rocketknight1/falcon-rw-1b",
                "Rocketknight1/tiny-random-falcon-7b",
                "Rocketknight1/tiny-random-falcon-40b",
            ]:
                tokenizer = AutoTokenizer.from_pretrained(repo)
                model = FalconForCausalLM.from_pretrained(repo)
                model.eval()
                model.to(device=torch_device)
                inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)

                # Test results are the same with and without cache
                outputs_no_cache = model.generate(**inputs, do_sample=False, max_new_tokens=20, use_cache=False)
                outputs_cache = model.generate(**inputs, do_sample=False, max_new_tokens=20, use_cache=True)
                self.assertTrue((outputs_cache - outputs_no_cache).sum().item() == 0)

    @require_bitsandbytes
    @slow
    def test_batched_generation(self):
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            "tiiuae/falcon-7b",
            device_map={"": torch_device},
            load_in_4bit=True,
        )

        test_text = "A sequence: 1, 2"  # should generate the rest of the sequence

        unpadded_inputs = tokenizer([test_text], return_tensors="pt").to(f"{torch_device}:0")
        unpadded_gen_out = model.generate(**unpadded_inputs, max_new_tokens=20)
        unpadded_gen_text = tokenizer.batch_decode(unpadded_gen_out, skip_special_tokens=True)

        dummy_text = "This is a longer text " * 2  # forces left-padding on `test_text`
        padded_inputs = tokenizer([test_text, dummy_text], return_tensors="pt", padding=True).to(f"{torch_device}:0")
        padded_gen_out = model.generate(**padded_inputs, max_new_tokens=20)
        padded_gen_text = tokenizer.batch_decode(padded_gen_out, skip_special_tokens=True)

        expected_output = "A sequence: 1, 2, 3, 4, 5, 6, 7, 8, "
        self.assertLess(unpadded_inputs.input_ids.shape[-1], padded_inputs.input_ids.shape[-1])  # left-padding exists
        self.assertEqual(unpadded_gen_text[0], expected_output)
        self.assertEqual(padded_gen_text[0], expected_output)

    @slow
    def test_falcon_alibi_sdpa_matches_eager(self):
        input_ids = torch.randint(0, 1000, (5, 20))

        config = FalconConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            new_decoder_architecture=True,
            alibi=True,
        )

        falcon = FalconForCausalLM(config)
        falcon = falcon.eval()

        with torch.no_grad():
            # output_attentions=True dispatches to eager path
            falcon_output_eager = falcon(input_ids, output_attentions=True)[0]
            falcon_output_sdpa = falcon(input_ids)[0]

        torch.testing.assert_close(falcon_output_eager, falcon_output_sdpa, rtol=1e-3, atol=1e-3)
