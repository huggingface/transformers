# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Team. All rights reserved.
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
import tempfile

from transformers import AutoTokenizer, AddedToken, PreTrainedTokenizerFast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.create_fast_tokenizer import SentencePieceExtractor
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
)
from tests.test_tokenization_common import TokenizerTesterMixin

input_string = """Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet...) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between Jax, PyTorch and TensorFlow.🤗 Transformers 提供了可以轻松地下载并且训练先进的预训练模型的 API 和工具。使用预训练模型可以减少计算消耗和碳排放，并且节省从头训练所需要的时间和资源。```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-tokenizer")
tokenizer("世界，你好！")```<|im_start|>Hello, world!<|im_end|>
"""

expected_tokens = ['Transform', 'ers', 'Ġ(', 'formerly', 'Ġknown', 'Ġas', 'Ġpy', 'torch', '-transform', 'ers', 'Ġand', 'Ġpy', 'torch', '-pre', 'trained', '-b', 'ert', ')', 'Ġprovides', 'Ġgeneral', '-purpose', 'Ġarchitectures', 'Ġ(', 'BERT', ',', 'ĠG', 'PT', '-', '2', ',', 'ĠRo', 'BERT', 'a', ',', 'ĠX', 'LM', ',', 'ĠDist', 'il', 'B', 'ert', ',', 'ĠXL', 'Net', '...)', 'Ġfor', 'ĠNatural', 'ĠLanguage', 'ĠUnderstanding', 'Ġ(', 'N', 'LU', ')', 'Ġand', 'ĠNatural', 'ĠLanguage', 'ĠGeneration', 'Ġ(', 'NL', 'G', ')', 'Ġwith', 'Ġover', 'Ġ', '3', '2', '+', 'Ġpretrained', 'Ġmodels', 'Ġin', 'Ġ', '1', '0', '0', '+', 'Ġlanguages', 'Ġand', 'Ġdeep', 'Ġinteroper', 'ability', 'Ġbetween', 'ĠJ', 'ax', ',', 'ĠPy', 'T', 'orch', 'Ġand', 'ĠTensorFlow', '.', 'ðŁ¤Ĺ', 'ĠTransformers', 'ĠæıĲ', 'ä¾Ľ', 'äºĨ', 'åı¯ä»¥', 'è½»æĿ¾', 'åľ°', 'ä¸ĭè½½', 'å¹¶ä¸Ķ', 'è®Ńç»ĥ', 'åħĪè¿ĽçļĦ', 'é¢Ħ', 'è®Ńç»ĥ', 'æ¨¡åŀĭ', 'çļĦ', 'ĠAPI', 'ĠåĴĮ', 'å·¥åħ·', 'ãĢĤ', 'ä½¿çĶ¨', 'é¢Ħ', 'è®Ńç»ĥ', 'æ¨¡åŀĭ', 'åı¯ä»¥', 'åĩıå°ĳ', 'è®¡ç®Ĺ', 'æ¶ĪèĢĹ', 'åĴĮ', 'ç¢³', 'æİĴæĶ¾', 'ï¼Įå¹¶', 'ä¸Ķ', 'èĬĤçľģ', 'ä»İ', 'å¤´', 'è®Ńç»ĥ', 'æīĢéľĢè¦ģ', 'çļĦæĹ¶éĹ´', 'åĴĮ', 'èµĦæºĲ', 'ãĢĤ', '```', 'python', 'Ċ', 'tokenizer', 'Ġ=', 'ĠAuto', 'Tokenizer', '.from', '_pre', 'trained', '("', 'Q', 'wen', '/Q', 'wen', '-token', 'izer', '")Ċ', 'tokenizer', '("', 'ä¸ĸçķĮ', 'ï¼Į', 'ä½łå¥½', 'ï¼ģ', '")', '```', '<|im_start|>', 'Hello', ',', 'Ġworld', '!', '<|im_end|>', 'Ċ']
expected_token_ids = [8963, 388, 320, 69514, 3881, 438, 4510, 27414, 32852, 388, 323, 4510, 27414, 21334, 35722, 1455, 529, 8, 5707, 4586, 58238, 77235, 320, 61437, 11, 479, 2828, 12, 17, 11, 11830, 61437, 64, 11, 1599, 10994, 11, 27604, 321, 33, 529, 11, 29881, 6954, 32574, 369, 18448, 11434, 45451, 320, 45, 23236, 8, 323, 18448, 11434, 23470, 320, 30042, 38, 8, 448, 916, 220, 18, 17, 10, 80669, 4119, 304, 220, 16, 15, 15, 10, 15459, 323, 5538, 94130, 2897, 1948, 619, 706, 11, 5355, 51, 21584, 323, 94986, 13, 144834, 80532, 93685, 83744, 34187, 73670, 104261, 29490, 62189, 103937, 104034, 102830, 98841, 104034, 104949, 9370, 5333, 58143, 102011, 1773, 37029, 98841, 104034, 104949, 73670, 101940, 100768, 104997, 33108, 100912, 105054, 90395, 100136, 106831, 45181, 64355, 104034, 113521, 101975, 33108, 85329, 1773, 73594, 12669, 198, 85593, 284, 8979, 37434, 6387, 10442, 35722, 445, 48, 16948, 45274, 16948, 34841, 3135, 1138, 85593, 445, 99489, 3837, 108386, 6313, 899, 73594, 151644, 9707, 11, 1879, 0, 151645, 198]

@require_tokenizers
class Qwen2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    tokenizer_class = Qwen2TokenizerFast
    rust_tokenizer_class = Qwen2TokenizerFast
    test_slow_tokenizer = True
    test_rust_tokenizer = False # we're going to just test the fast one I'll remove this
    space_between_special_tokens = False
    from_pretrained_kwargs = {}
    test_seq2seq = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tok_auto]

    def test_integration_expected_tokens(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.tokenize(input_string), expected_tokens)

    def test_integration_expected_token_ids(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.encode(input_string), expected_token_ids)

    def test_save_and_reload(self):
        for tok in self.tokenizers:
            with self.subTest(f"{tok.__class__.__name__}"):
                original_tokens = tok.tokenize(input_string)
                original_ids = tok.encode(input_string)

                # Save tokenizer to temporary directory
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tok.save_pretrained(tmp_dir)

                    # Reload tokenizer from saved directory
                    reloaded_tok = tok.__class__.from_pretrained(tmp_dir)

                    # Test that reloaded tokenizer produces same results
                    reloaded_tokens = reloaded_tok.tokenize(input_string)
                    reloaded_ids = reloaded_tok.encode(input_string)

                    self.assertEqual(original_tokens, reloaded_tokens)
                    self.assertEqual(original_ids, reloaded_ids)
