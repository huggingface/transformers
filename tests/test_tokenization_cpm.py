# coding=utf-8
# Copyright 2018 HuggingFace Inc. team.
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


from transformers.models.cpm.tokenization_cpm import CpmTokenizer
from transformers.testing_utils import custom_tokenizers

from .test_modeling_xlnet import XLNetModelTest


@custom_tokenizers
class CpmTokenizationTest(XLNetModelTest):
    def test_pre_tokenization(self):
        tokenizer = CpmTokenizer.from_pretrained("TsinghuaAI/CPM-Generate")
        text = "Hugging Face大法好，谁用谁知道。"
        normalized_text = "Hugging Face大法好,谁用谁知道。<unk>"
        bpe_tokens = "▁Hu gg ing ▁ ▂ ▁F ace ▁大法 ▁好 ▁ , ▁谁 ▁用 ▁谁 ▁知 道 ▁ 。".split()

        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]

        input_bpe_tokens = [13789, 13283, 1421, 8, 10, 1164, 13608, 16528, 63, 8, 9, 440, 108, 440, 121, 90, 8, 12, 0]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

        reconstructed_text = tokenizer.decode(input_bpe_tokens)
        self.assertEqual(reconstructed_text, normalized_text)
