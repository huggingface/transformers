# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from transformers import GPTJConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    tooslow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        GPTJForCausalLM,
        GPTJForQuestionAnswering,
        GPTJForSequenceClassification,
        GPTJModel,
    )


class GPTJModelTester(CausalLMModelTester):
    config_class = GPTJConfig
    if is_torch_available():
        base_model_class = GPTJModel
        causal_lm_class = GPTJForCausalLM
        question_answering_class = GPTJForQuestionAnswering
        sequence_classification_class = GPTJForSequenceClassification


@require_torch
class GPTJModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (GPTJModel, GPTJForCausalLM, GPTJForSequenceClassification, GPTJForQuestionAnswering)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GPTJModel,
            "question-answering": GPTJForQuestionAnswering,
            "text-classification": GPTJForSequenceClassification,
            "text-generation": GPTJForCausalLM,
            "zero-shot": GPTJForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = GPTJModelTester
    fx_compatible = True
    test_pruning = False
    test_missing_keys = False
    test_model_parallel = False
    test_head_masking = False

    # TODO: Fix the failed tests
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
        if (
            pipeline_test_case_name == "QAPipelineTests"
            and tokenizer_name is not None
            and not tokenizer_name.endswith("Fast")
        ):
            # `QAPipelineTests` fails for a few models when the slower tokenizer are used.
            # (The slower tokenizers were never used for pipeline tests before the pipeline testing rework)
            # TODO: check (and possibly fix) the `QAPipelineTests` with slower tokenizer
            return True

        return False


@require_torch
class GPTJModelLanguageGenerationTest(unittest.TestCase):
    @tooslow
    def test_lm_generate_gptj(self):
        # Marked as @tooslow due to GPU OOM
        for checkpointing in [True, False]:
            model = GPTJForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16
            )
            if checkpointing:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            model.to(torch_device)
            input_ids = torch.tensor([[464, 3290]], dtype=torch.long, device=torch_device)  # The dog
            # The dog is a man's best friend. It is a loyal companion, and it is a friend
            expected_output_ids = [464, 3290, 318, 257, 582, 338, 1266, 1545, 13, 632, 318, 257, 9112, 15185, 11, 290, 340, 318, 257, 1545]  # fmt: skip
            output_ids = model.generate(input_ids, do_sample=False)
            self.assertListEqual(output_ids[0].tolist(), expected_output_ids)

    @tooslow
    def test_gptj_sample(self):
        # Marked as @tooslow due to GPU OOM (issue #13676)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", revision="float16")
        model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16)
        model.to(torch_device)

        torch.manual_seed(0)
        tokenized = tokenizer("Today is a nice day and", return_tensors="pt", return_token_type_ids=True)
        input_ids = tokenized.input_ids.to(torch_device)
        output_ids = model.generate(input_ids, do_sample=True)
        output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        token_type_ids = tokenized.token_type_ids.to(torch_device)
        output_seq = model.generate(input_ids=input_ids, do_sample=True, num_return_sequences=5)
        output_seq_tt = model.generate(
            input_ids=input_ids, token_type_ids=token_type_ids, do_sample=True, num_return_sequences=5
        )
        output_seq_strs = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
        output_seq_tt_strs = tokenizer.batch_decode(output_seq_tt, skip_special_tokens=True)

        if torch_device != "cpu":
            # currently this expect value is only for `cuda`
            EXPECTED_OUTPUT_STR = (
                "Today is a nice day and I've already been enjoying it. I walked to work with my wife"
            )
        else:
            EXPECTED_OUTPUT_STR = "Today is a nice day and one of those days that feels a bit more alive. I am ready"

        self.assertEqual(output_str, EXPECTED_OUTPUT_STR)
        self.assertTrue(
            all(output_seq_strs[idx] != output_seq_tt_strs[idx] for idx in range(len(output_seq_tt_strs)))
        )  # token_type_ids should change output

    @tooslow
    def test_contrastive_search_gptj(self):
        article = (
            "DeepMind Technologies is a British artificial intelligence subsidiary of Alphabet Inc. and "
            "research laboratory founded in 2010. DeepMind was acquired by Google in 2014. The company is based"
        )

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16
        ).to(torch_device)
        input_ids = tokenizer(article, return_tensors="pt").input_ids.to(torch_device)

        outputs = model.generate(input_ids, penalty_alpha=0.6, top_k=4, max_length=256)
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(
            generated_text,
            [
                "DeepMind Technologies is a British artificial intelligence subsidiary of Alphabet Inc. and research "
                "laboratory founded in 2010. DeepMind was acquired by Google in 2014. The company is based in London, "
                "United Kingdom with offices in Mountain View, San Francisco, New York City, Paris, Tokyo, Seoul, "
                "Beijing, Singapore, Tel Aviv, Dublin, Sydney, and Melbourne.[1]\n\nContents\n\nIn 2010, Google's "
                "parent company, Alphabet, announced a $500 million investment in DeepMind, with the aim of creating "
                "a company that would apply deep learning to problems in healthcare, energy, transportation, and "
                "other areas.[2]\n\nOn April 23, 2014, Google announced that it had acquired DeepMind for $400 "
                "million in cash and stock.[3] The acquisition was seen as a way for Google to enter the "
                "fast-growing field of artificial intelligence (AI), which it had so far avoided due to concerns "
                'about ethical and social implications.[4] Google co-founder Sergey Brin said that he was "thrilled" '
                'to have acquired DeepMind, and that it would "help us push the boundaries of AI even further."'
                "[5]\n\nDeepMind's founders, Demis Hassabis and Mustafa Suleyman, were joined by a number of Google "
                "employees"
            ],
        )
