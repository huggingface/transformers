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

from transformers import is_ov_available
from transformers.testing_utils import require_ov

if is_ov_available():
    from transformers import AutoTokenizer, OVAutoModelForQuestionAnswering


@require_ov
class OVBertSQuADTest(unittest.TestCase):
    def test_inference_no_head_absolute_embedding(self):
        tok = AutoTokenizer.from_pretrained("dkurt/bert-large-uncased-whole-word-masking-squad-int8-0001")
        model = OVAutoModelForQuestionAnswering.from_pretrained("dkurt/bert-large-uncased-whole-word-masking-squad-int8-0001")

        context = """
        Soon her eye fell on a little glass box that
        was lying under the table: she opened it, and
        found in it a very small cake, on which the
        words “EAT ME” were beautifully marked in
        currants. “Well, I’ll eat it,” said Alice, “ and if
        it makes me grow larger, I can reach the key ;
        and if it makes me grow smaller, I can creep
        under the door; so either way I’ll get into the
        garden, and I don’t care which happens !”
        """

        question = "Where Alice should go?"

        input_ids = tok.encode(question + " " + tok.sep_token + " " + context, return_tensors="pt")

        outputs = model(input_ids)

        start_pos = outputs.start_logits.argmax()
        end_pos = outputs.end_logits.argmax() + 1

        answer_ids = input_ids[0, start_pos:end_pos]
        answer = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(answer_ids))

        self.assertEqual(answer, "the garden")
