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

from transformers import is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device


if is_torch_available():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

# https://github.com/google-research/t5x/blob/main/docs/models.md#umt5-checkpoints

@require_torch
@require_sentencepiece
@require_tokenizers
class Umt5IntegrationTest(unittest.TestCase):
    @slow
    def test_small_integration_test(self):
        """
        For comparison run the kaggle notbook available here : TODO share view link only
        """

        model = AutoModelForSeq2SeqLM.from_pretrained("google/umt5-small", return_dict=True).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
        input_text = [
            'Bonjour monsieur <extra_id_0> bien <extra_id_1>.',
            'No se como puedo <extra_id_0>.',
            'This is the reason why we <extra_id_0> them.',
            'The <extra_id_0> walks in <extra_id_1>, sits'
        ]
        input_ids = tokenizer(input_text, return_tensors="pt", padding = True).input_ids
        EXPECTED_INPUT_IDS = torch.tensor([
                [ 38530, 210703, 256299,   1410, 256298,    274,      1,      0, 0,      0],
                [   826,    321,    671,  25922, 256299,    274,      1,      0, 0,      0]
                [  1460,    339,    312,  19014,  10620,    758, 256299,   2355, 274,      1]
                [   517, 256299,  14869,    281,    301, 256298,    275,   9433, 281,      1]
            ]
        )
        self.assertEqual(input_ids, EXPECTED_INPUT_IDS)
        EXPECTED_GENERATED_ID = torch.tensor([
                [  ],
                [  ],
                [  ],
                [  ]
            ]
        )
        generated_ids = model.generate(input_ids.to(torch_device))
        self.assertTrue(generated_ids, )
        EXPECTED_FILLING ="""<pad><extra_id_0> et<extra_id_1> [eod] <extra_id_2><extra_id_55>.. [eod] 💐 💐 💐 💐 💐 💐 💐 💐 💐 💐 💐 <extra_id_56> ajšietosto<extra_id_56> lleux<extra_id_19><extra_id_6> ajšie</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
<pad><extra_id_0>.<extra_id_1>.,\n...spech \n<extra_id_20> <extra_id_21></s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
<pad><extra_id_0> are not going to be a part of the world. We are not going to be a part of<extra_id_1> and<extra_id_2> \n<extra_id_48>.<extra_id_48></s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
<pad><extra_id_0> city<extra_id_1> the city<extra_id_2> verkarzekł<extra_id_3>..,,<extra_id_4><extra_id_56>ajšie\n海外取寄せ品lleuxlleuxmogulleuxajšieM Loadingüentlleuxчинностіonomy嵋ABRmnázilleuxajšiewaнуарajšielemba嵋zdrowi Issumnázi\nmmersciedad泷嵋 https Httpajšiealulleuxlleuxambiquemündeijstajšiequippedrakutenblogzytelni seanwaitajšie
"""
        filling = tokenizer.batch_decode(generated_ids)
        self.assertTrue(filling, EXPECTED_FILLING)