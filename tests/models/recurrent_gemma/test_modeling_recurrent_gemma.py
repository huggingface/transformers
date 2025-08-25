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
"""Testing suite for the PyTorch RecurrentGemma model."""

import unittest

import pytest
from parameterized import parameterized

from transformers import AutoModelForCausalLM, AutoTokenizer, RecurrentGemmaConfig, is_torch_available, set_seed
from transformers.testing_utils import (
    Expectations,
    require_bitsandbytes,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import RecurrentGemmaConfig, RecurrentGemmaForCausalLM, RecurrentGemmaModel


from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class RecurrentGemmaModelTester(CausalLMModelTester):
    config_class = RecurrentGemmaConfig
    if is_torch_available():
        base_model_class = RecurrentGemmaModel
        causal_lm_class = RecurrentGemmaForCausalLM


@require_torch
class RecurrentGemmaModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (RecurrentGemmaModel, RecurrentGemmaForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": RecurrentGemmaModel,
            "text-generation": RecurrentGemmaForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    has_attentions = False
    model_tester_class = RecurrentGemmaModelTester

    @unittest.skip(reason="RecurrentGemma only supports sdpa")
    def test_eager_matches_sdpa_generate(self):
        pass

    @unittest.skip(reason="RecurrentGemma does not return the cache")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip(reason="RecurrentGemma does not return the cache")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="RecurrentGemma does not return the cache")
    def test_contrastive_generate(self):
        pass

    @unittest.skip(reason="SQRBound is known to have issues with gc")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Past key values are not returned")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip(reason="Past key values are not returned")
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="Past key values are not returned")
    def test_model_parallel_beam_search(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip(reason="Rely on `past_key_values` to crop the assistant pkv. Not supported")
    def test_assisted_decoding_matches_greedy_search(self):
        pass

    @unittest.skip(reason="RecurrentGemma's output different if you pad left or right. This is expected")
    def test_left_padding_compatibility(self):
        pass

    @pytest.mark.generate
    @unittest.skip(reason="Relies on `past_key_values` returned by the model. Not supported with recurrent gemma")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(reason="TODO @arthurzucker not super important and failing.")
    def test_initialization(self):
        pass

    @unittest.skip(reason="RecurrentGemma is unusual and fails a lot of generation tests")
    @pytest.mark.generate
    def test_beam_sample_generate_dict_output(self):
        pass

    @unittest.skip(reason="RecurrentGemma is unusual and fails a lot of generation tests")
    @pytest.mark.generate
    def test_beam_search_generate_dict_output(self):
        pass

    @unittest.skip(reason="RecurrentGemma is unusual and fails a lot of generation tests")
    @pytest.mark.generate
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="RecurrentGemma is unusual and fails a lot of generation tests")
    @pytest.mark.generate
    def test_constrained_beam_search_generate_dict_output(self):
        pass

    @unittest.skip(reason="RecurrentGemma is unusual and fails a lot of generation tests")
    @pytest.mark.generate
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip(reason="RecurrentGemma is unusual and fails a lot of generation tests")
    @pytest.mark.generate
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="RecurrentGemma is unusual and fails a lot of generation tests")
    @pytest.mark.generate
    def test_group_beam_search_generate(self):
        pass

    @unittest.skip(reason="RecurrentGemma is unusual and fails a lot of generation tests")
    @pytest.mark.generate
    def test_group_beam_search_generate_dict_output(self):
        pass

    @unittest.skip(reason="RecurrentGemma is unusual and fails a lot of generation tests")
    @pytest.mark.generate
    def test_constrained_beam_search_generate(self):
        pass

    @unittest.skip(reason="RecurrentGemma is unusual and fails a lot of generation tests")
    @pytest.mark.generate
    def test_greedy_generate_dict_outputs(self):
        pass

    @unittest.skip(reason="RecurrentGemma is unusual and fails a lot of generation tests")
    @pytest.mark.generate
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="RecurrentGemma is unusual and fails a lot of generation tests")
    def test_model_outputs_equivalence(self):
        pass


@require_torch_accelerator
@slow
class RecurrentGemmaIntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]
    input_long_text = ['<bos><s>Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a phone at the wreckage site. The two publications described the supposed video, but did not post it on their websites. The publications said that they watched the video, which was found by a source close to the investigation. "One can hear cries of \'My God\' in several languages," Paris Match reported. "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt, editor-in-chief of Bild online. An official with France\'s accident investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col.']  # fmt: skip
    model_id = "google/recurrentgemma-2b"

    @require_read_token
    def test_2b_generate(self):
        EXPECTED_TEXTS = ['Hello I am doing a project on the topic of "The impact of the internet on the society" and I am looking for some information on the topic. I am looking for some information on the impact of the internet on the society. I am looking for some information on the impact of the internet on the society. I am looking for some', 'Hi today is a new app that allows you to make money by watching videos.\n\nThe app is very simple to use and you can earn money by watching videos.\n\nThe app is available for both Android and iOS devices and you can download it from the Google Play Store or the App Store.\n\nOnce you have downloaded the app']  # fmt: skip
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
        ).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.padding_side = "right"

        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

        tokenizer.padding_side = "left"
        EXPECTED_TEXTS = ['Hello I am doing a project on the topic of "The impact of the internet on the society" and I am looking for some information on the topic. I am looking for some information on the impact of the internet on the society. I am looking for some information on the impact of the internet on the society. I am looking for some', 'Hi today I’m going to show you how to make a simple and easy to make a <strong>DIY</strong> <strong>DIY</strong> <strong>DIY</strong> <strong>DIY</strong> <strong>DIY</strong> <strong>DIY</strong> <strong>DIY</strong> <strong>DIY</strong> <strong>DIY</strong> <strong>DIY</strong> <strong>DIY</strong> <strong>DIY']  # fmt: skip

        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        del model
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float16).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        del model
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_2b_sample(self):
        set_seed(0)
        expectations = Expectations(
            {
                (None, None): [
                    "What is Deep learning ?\n\nDeep learning is the next frontier in computer vision. It is an Artificial Intelligence (AI) discipline that is rapidly being adopted across industries. The success of Deep"
                ],
                ("cuda", 8): [
                    "What is Deep learning ?\n\nDeep learning is the next frontier in computer vision, it’s an incredibly powerful branch of artificial intelligence.\n\nWhat is Dalle?\n\nDalle is",
                ],
            }
        )
        EXPECTED_TEXT = expectations.get_expectation()
        model = AutoModelForCausalLM.from_pretrained(self.model_id).to(torch_device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        inputs = tokenizer("What is Deep learning ?", return_tensors="pt", padding=True).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=32, do_sample=True)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_bitsandbytes
    @require_read_token
    def test_model_2b_8bit(self):
        EXPECTED_TEXTS = ['Hello I am doing a project on the topic of "The impact of social media on the society" and I am looking', "Hi today I'm going to show you how to make a simple and easy to make a 3D"]  # fmt: skip

        model = AutoModelForCausalLM.from_pretrained(
            "gg-hf/recurrent-gemma-2b-hf", device_map={"": torch_device}, load_in_8bit=True, dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_read_token
    def test_long_context(self):
        EXPECTED_GENERATION = [' Jean-Paul Delannoy told CNN that the BEA is "not aware of any video footage that could have been taken on board the plane." He added that the BEA is "not aware of any video footage that could have been taken on board the plane." The BEA is the French equivalent of the National Transportation Safety Board']  # fmt: skip

        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float16).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        inputs = tokenizer(self.input_long_text, return_tensors="pt").to(torch_device)
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        output_text = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        print(output_text)
        self.assertEqual(output_text, EXPECTED_GENERATION)

    @require_read_token
    def test_longer_than_window(self):
        EXPECTED_GENERATION = [" Robin's comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French Alps. All 150 on board were killed. Paris Match and Bild reported that the"]  # fmt: skip

        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float16).to(torch_device)
        model.config.attention_window_size = 256  # Make the attention window size shorter than the current prompt
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        inputs = tokenizer(self.input_long_text, return_tensors="pt").to(torch_device)
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        output_text = tokenizer.batch_decode(output[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        self.assertEqual(output_text, EXPECTED_GENERATION)
