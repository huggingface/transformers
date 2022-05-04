# coding=utf-8
# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch OPT model. """

import unittest


from transformers import OPTConfig, is_torch_available, BartTokenizerFast



if is_torch_available():
    import torch

    from transformers import (
        AutoModelForSequenceClassification,
        OPTForCausalLM,
        OPTForConditionalGeneration,
        OPTForQuestionAnswering,
        OPTForSequenceClassification,
        OPTModel,
        OPTTokenizer,
        pipeline,
    )

class OPTEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.path_model = "/home/younes/Desktop/Work/data/opt-350-m/"
        self.path_logits_meta = "/home/younes/Desktop/Work/metaseq-conversion/logits_metaseq.p"
    
    def test_load_model(self):
        try:
            _ = OPTForCausalLM.from_pretrained(self.path_model)
        except BaseException:
            self.fail("Failed loading model")
    
    def test_logits(self):
        model = OPTForCausalLM.from_pretrained(self.path_model)
        model = model.eval()
        tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large")
        prompts = [
            "Today is a beautiful day and I want to",
            "In the city of",
            "Paris is the capital of France and",
            "Computers and mobile phones have taken",
        ]
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids
        logits = model(input_ids)[0]
        logits_meta = torch.load(self.path_logits_meta)
        assert torch.allclose(logits, logits_meta.permute(1, 0, 2))

if __name__ == "__main__":
    unittest.main()