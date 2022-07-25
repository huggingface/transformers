# coding=utf-8
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
from transformers.testing_utils import require_torch, slow
from transformers.trainer_utils import set_seed


if is_torch_available():
    import torch

    from transformers import JukeboxModel, JukeboxTokenizer  # ,JUKEBOX_PRETRAINED_MODEL_ARCHIVE_LIST


@require_torch
class Jukebox1bModelTester(unittest.TestCase):
    all_model_classes = (JukeboxModel,) if is_torch_available() else ()
    metas = dict(
        artist="Zac Brown Band",
        genres="Country",
        lyrics="""I met a traveller from an antique land,
    Who said "Two vast and trunkless legs of stone
    Stand in the desert. . . . Near them, on the sand,
    Half sunk a shattered visage lies, whose frown,
    And wrinkled lip, and sneer of cold command,
    Tell that its sculptor well those passions read
    Which yet survive, stamped on these lifeless things,
    The hand that mocked them, and the heart that fed;
    And on the pedestal, these words appear:
    My name is Ozymandias, King of Kings;
    Look on my Works, ye Mighty, and despair!
    Nothing beside remains. Round the decay
    Of that colossal Wreck, boundless and bare
    The lone and level sands stretch far away
    """,
    )
    # fmt: off
    EXPECTED_OUTPUT_2 = [
        1864, 1536, 1213, 1869, 1321, 1597, 519, 947, 1177, 789, 1434, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 1007, 1472, 255, 1228,
        555, 1272, 1379, 1423, 1673, 427, 1683, 1321, 475, 416, 1177, 1827,
        1106, 1127, 1494, 812
    ]
    
    EXPECTED_OUTPUT_1 = [
        1125, 1585, 1485, 2020, 1141, 1680, 381, 539, 1368, 642, 1585, 284,
        717, 1544, 1045, 1320, 711, 193, 1440, 1193, 416, 1125, 539, 1544,
        593, 1274, 1181, 1658, 1181, 1145, 2037, 1125, 556, 1014, 1045, 1858,
        1749, 1803, 1440, 1145, 416, 416, 1372, 1079, 1045, 1320, 1764, 158,
        2020, 1543, 2037, 416, 539, 2047, 1446, 885, 1749, 2047, 118, 1348,
        1585, 284, 529, 2047, 1228, 556, 732, 2047, 307, 1323, 2037, 1446,
        591, 1803, 58, 591, 529, 1079, 642, 591
    ]
    
    EXPECTED_OUTPUT_0 = [
        1979, 1613, 290, 1843, 844, 1427, 293, 616, 1771, 632, 591, 290,
        234, 842, 589, 948, 983, 616, 1613, 1613, 290, 632, 89, 632,
        290, 1022, 983, 1612, 1353, 581, 1353, 755, 185, 307, 632, 1979,
        854, 1120, 1572, 719
    ]
    # fmt: on

    def prepare_inputs(self, model_id):
        tokenizer = JukeboxTokenizer.from_pretrained(model_id)
        tokens = tokenizer(**self.metas)["input_ids"]
        return tokens

    @require_torch
    def test_sampling(self):
        model_id = "ArthurZ/jukebox-1b-lyrics"
        model = JukeboxModel.from_pretrained(model_id, cond_res_scale=[None, True, False]).eval()

        labels = self.prepare_inputs(model_id)

        set_seed(0)
        zs = [torch.zeros(1, 0, dtype=torch.long).cpu() for _ in range(3)]
        zs = model._sample(zs, labels, [2], sample_tokens=10)
        assert torch.allclose(zs[-1][0], self.EXPECTED_OUTPUT_2)

        zs[-1] = self.EXPECTED_OUTPUT_2.unsqueeze(0)
        set_seed(0)
        zs[-1] = torch.cat((zs[-1], torch.zeros(1, 1000000 - zs[-1].shape[-1]).cpu()), dim=-1).long()
        zs = model._sample(zs, labels, [1], sample_tokens=10)
        assert torch.allclose(zs[-2][0, :80], self.EXPECTED_OUTPUT_1)

        zs[-2] = self.EXPECTED_OUTPUT_1.unsqueeze(0)

        set_seed(0)
        zs[-2] = torch.cat((zs[-2], torch.zeros(1, 1000000 - zs[-2].shape[-1]).cpu()), dim=-1).long()
        zs = model._sample(zs, labels, [0], sample_tokens=10)
        assert torch.allclose(zs[0][0, :40], self.EXPECTED_OUTPUT_0)

    @slow
    @require_torch
    def test_slow_sampling(self):
        model_id = "ArthurZ/jukebox-1b-lyrics"
        model = JukeboxModel.from_pretrained(model_id).eval().to("cuda")

        labels = [i.cuda() for i in self.prepare_inputs(model_id)]
        set_seed(0)
        zs = [torch.zeros(1, 0, dtype=torch.long).cuda() for _ in range(3)]
        zs = model._sample(zs, labels, [2], sample_tokens=10)
        print(zs[-1][0].cpu())
        print(self.EXPECTED_OUTPUT_2)
        assert torch.allclose(zs[-1][0].cpu(), self.EXPECTED_OUTPUT_2)

    def test_vqvae(self):
        # implemented vavae decoding test at 3 levels using the expected outputs
        pass


@require_torch
class Jukebox5bModelTester(unittest.TestCase):
    all_model_classes = (JukeboxModel,) if is_torch_available() else ()
    metas = dict(
        artist="Zac Brown Band",
        genres="Country",
        lyrics="""I met a traveller from an antique land,
    Who said "Two vast and trunkless legs of stone
    Stand in the desert. . . . Near them, on the sand,
    Half sunk a shattered visage lies, whose frown,
    And wrinkled lip, and sneer of cold command,
    Tell that its sculptor well those passions read
    Which yet survive, stamped on these lifeless things,
    The hand that mocked them, and the heart that fed;
    And on the pedestal, these words appear:
    My name is Ozymandias, King of Kings;
    Look on my Works, ye Mighty, and despair!
    Nothing beside remains. Round the decay
    Of that colossal Wreck, boundless and bare
    The lone and level sands stretch far away
    """,
    )

    # fmt: off
    EXPECTED_OUTPUT_2 = [
        1489, 653, 653, 653, 653, 653, 653, 653, 653, 653, 1489, 653,
        653, 653, 653, 653, 653, 653, 653, 653
    ]
    
    EXPECTED_OUTPUT_1 = [
        1125, 416, 1125, 1125, 1125, 1125, 416, 416, 416, 416, 1585, 284,
        717, 1544, 1045, 1320, 711, 193, 1440, 1193, 416, 1125, 539, 1544,
        593, 1274, 1181, 1658, 1181, 1145, 2037, 1125, 556, 1014, 1045, 1858,
        1749, 1803, 1440, 1145, 416, 416, 1372, 1079, 1045, 1320, 1764, 158,
        2020, 1543, 2037, 416, 539, 2047, 1446, 885, 1749, 2047, 118, 1348,
        1585, 284, 529, 2047, 1228, 556, 732, 2047, 307, 1323, 2037, 1446,
        591, 1803, 58, 591, 529, 1079, 642, 591
    ]
    
    EXPECTED_OUTPUT_0 = [
        1755, 1061, 234, 1755, 290, 1572, 234, 491, 992, 417, 591, 290,
        234, 842, 589, 948, 983, 616, 1613, 1613, 290, 632, 89, 632,
        290, 1022, 983, 1612, 1353, 581, 1353, 755, 185, 307, 632, 1979,
        854, 1120, 1572, 719, 491, 34, 755, 632, 844, 755, 1802, 225,
        2013, 1814, 1148, 616, 185, 1979, 1460, 983, 1168, 1613, 34, 1242,
        632, 34, 34, 1982, 1510, 554, 983, 1784, 526, 1691, 1268, 1268,
        290, 755, 34, 307, 222, 234, 648, 526
    ]
    # fmt: on

    def prepare_inputs(self, model_id):
        tokenizer = JukeboxTokenizer.from_pretrained(model_id)
        tokens = tokenizer(**self.metas)["input_ids"]
        return tokens

    def test_sampling(self):
        model_id = "ArthurZ/jukebox-5b-lyrics"
        model = JukeboxModel.from_pretrained(model_id).eval()

        labels = self.prepare_inputs(model_id)
        set_seed(0)
        zs = [torch.zeros(1, 0, dtype=torch.long).cpu() for _ in range(3)]
        zs = model._sample(zs, labels, [2], sample_tokens=10)
        assert torch.allclose(zs[-1][0], self.EXPECTED_OUTPUT_2)

        zs[-1] = self.EXPECTED_OUTPUT_2.unsqueeze(0)
        set_seed(0)
        zs[-1] = torch.cat((zs[-1], torch.zeros(1, 1000000 - zs[-1].shape[-1]).cpu()), dim=-1).long()
        zs = model._sample(zs, labels, [1], sample_tokens=10)
        assert torch.allclose(zs[-2][0, :80], self.EXPECTED_OUTPUT_1)

        zs[-2] = self.EXPECTED_OUTPUT_1.unsqueeze(0)

        set_seed(0)
        zs[-2] = torch.cat((zs[-2], torch.zeros(1, 1000000 - zs[-2].shape[-1]).cpu()), dim=-1).long()
        zs = model._sample(zs, labels, [0], sample_tokens=10)
        assert torch.allclose(zs[0][0, :80], self.EXPECTED_OUTPUT_0)

    @slow
    def test_slow_sampling(self):
        model_id = "ArthurZ/jukebox-5b-lyrics"
        model = JukeboxModel.from_pretrained(model_id).eval().to("cuda")

        labels = [i.cuda() for i in self.prepare_inputs(model_id)]
        set_seed(0)
        zs = [torch.zeros(1, 0, dtype=torch.long).cuda() for _ in range(3)]
        zs = model._sample(zs, labels, [2], sample_tokens=10)
        assert torch.allclose(zs[-1][0].cpu(), self.EXPECTED_OUTPUT_2)

    def test_vqvae(self):
        # implement vavae decoding test at 3 levels using the expected outputs
        pass


if __name__ == "__main__":
    tester = Jukebox5bModelTester()
    tester.test_1b_lyrics()
    tester.test_slow_sampling()
