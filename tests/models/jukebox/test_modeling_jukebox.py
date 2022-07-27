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
from transformers.testing_utils import require_accelerate, require_torch, slow
from transformers.trainer_utils import set_seed


if is_torch_available():
    import torch

    from transformers import JukeboxModel, JukeboxTokenizer  # ,JUKEBOX_PRETRAINED_MODEL_ARCHIVE_LIST


@require_torch
class Jukebox1bModelTester(unittest.TestCase):
    all_model_classes = (JukeboxModel,) if is_torch_available() else ()
    model_id = "ArthurZ/jukebox-1b-lyrics"
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
        1434, 324, 1489, 756, 1224, 1150, 1489, 353, 2033, 1622, 1536, 519,
        475, 1996, 1643, 701, 1229, 1434, 10, 1420, 1306, 178, 409, 2038,
        1355, 286, 897, 1804, 253, 1643, 1685, 1769, 1002, 1597, 30, 539,
        376, 427, 1179, 286
    ]

    EXPECTED_OUTPUT_1 = [
        1125, 1125, 904, 2037, 1125, 1274, 317, 642, 1274, 317, 851, 642,
        642, 747, 867, 502, 502, 416, 1125, 1125, 317, 1274, 317, 1125,
        416, 1125, 1125, 1125, 1125, 317, 855, 844, 94, 855, 502, 1714,
        1107, 747, 1353, 573
    ]

    EXPECTED_OUTPUT_0 = [
        1755, 1061, 808, 1755, 992, 1572, 185, 491, 992, 417, 234, 89,
        234, 417, 234, 234, 234, 417, 1638, 1638, 677, 1659, 541, 1659,
        946, 579, 992, 556, 844, 329, 926, 556, 293, 579, 946, 1659,
        1562, 579, 1372, 290
    ]

    EXPECTED_Y_COND = [1058304, 0, 786432, 7169, 507, 76, 27, 40, 30, 76]

    EXPECTED_GPU_OUTPUTS_0 = [
        591, 1979, 89, 1332, 1572, 755, 844, 1022, 234, 1174, 1962, 1174,
        1755, 676, 58, 1756, 844, 739, 185, 1332, 806, 1180, 774, 842,
        306, 442, 1797, 734, 1081, 109, 806, 1492, 926, 2008, 844, 2008,
        992, 89, 1353, 637
    ]
    EXPECTED_GPU_OUTPUTS_1 = [
        1125, 2037, 317, 1372, 2037, 851, 1274, 1125, 642, 502, 1274, 851,
        1125, 502, 317, 1125, 880, 904, 317, 1125, 642, 502, 844, 851,
        416, 317, 1585, 642, 1125, 58, 697, 1125, 1585, 2037, 502, 2037,
        851, 317, 1125, 642
    ]
    EXPECTED_GPU_OUTPUTS_2 = [
        1489, 1489, 324, 1489, 1600, 1150, 1489, 1489, 947, 1357, 1600, 1417,
        1481, 1003, 141, 1165, 1303, 904, 303, 1369, 395, 461, 994, 1283,
        269, 35, 1699, 241, 1369, 35, 1303, 583, 825, 1941, 1089, 1944,
        581, 35, 1153, 1153
    ]
    EXPECTED_VQVAE = [
        -0.0168, -0.0083, -0.0062, -0.0078, -0.0095, -0.0108, -0.0117, -0.0124,
        -0.0138, -0.0149, -0.0148, -0.0140, -0.0136, -0.0130, -0.0125, -0.0120,
        -0.0129, -0.0148, -0.0151, -0.0138, -0.0130, -0.0129, -0.0125, -0.0116,
        -0.0119, -0.0130, -0.0129, -0.0116, -0.0113, -0.0118, -0.0112, -0.0104,
        -0.0114, -0.0127, -0.0122, -0.0103, -0.0083, -0.0070, -0.0060, -0.0051
    ]
    EXPECTED_PRIMED_0 = [
        390, 1160, 1002, 1907, 1788, 1788, 1788, 1907, 1002, 1002, 1854, 1002,
        1002, 1002, 1002, 1002, 1002, 1160, 1160, 1606, 596, 596, 1160, 1002,
        1516, 596, 1002, 1002, 1002, 1907, 1788, 1788, 1788, 1854, 1788, 1907,
        1907, 1788, 596, 1626
    ]
    EXPECTED_PRIMED_1 = [
        1236, 1668, 1484, 1920, 1848, 1409, 139, 864, 1828, 1272, 1599, 824,
        1672, 139, 555, 1484, 824, 1920, 555, 596, 1579, 1599, 1231, 1599,
        1637, 1407, 212, 824, 1599, 116, 1433, 824, 258, 1599, 1433, 1895,
        1063, 1433, 1433, 1599
    ]
    EXPECTED_PRIMED_2 = [
        1684, 1873, 1119, 1189, 395, 611, 1901, 972, 890, 1337, 1392, 1927,
        96, 972, 672, 780, 1119, 890, 158, 771, 1073, 1927, 353, 1331,
        1269, 1459, 1333, 1645, 812, 1577, 1337, 606, 353, 981, 1466, 619,
        197, 391, 302, 1930
    ]
    # fmt: on

    def prepare_inputs(self):
        tokenizer = JukeboxTokenizer.from_pretrained(self.model_id)
        tokens = tokenizer(**self.metas)["input_ids"]
        return tokens

    @require_torch
    def test_sampling(self):
        model = JukeboxModel.from_pretrained(self.model_id, min_duration = 0).eval()
        labels = self.prepare_inputs()

        set_seed(0)
        zs = [torch.zeros(1, 0, dtype=torch.long).cpu() for _ in range(3)]
        zs = model._sample(zs, labels, [2], sample_length=40*model.priors[-1].raw_to_tokens, save_results=False)
        assert torch.allclose(zs[-1][0], torch.tensor(self.EXPECTED_OUTPUT_2))

        zs[-1] = torch.tensor(self.EXPECTED_OUTPUT_2).unsqueeze(0)
        set_seed(0)
        zs[-1] = torch.cat((zs[-1], torch.zeros(1, 1000000 - zs[-1].shape[-1]).cpu()), dim=-1).long()
        zs = model._sample(zs, labels, [1], sample_length=40*model.priors[-2].raw_to_tokens, save_results=False)
        assert torch.allclose(zs[-2][0], torch.tensor(self.EXPECTED_OUTPUT_1))

        zs[-2] = torch.tensor(self.EXPECTED_OUTPUT_1).unsqueeze(0)

        set_seed(0)
        zs[-2] = torch.cat((zs[-2], torch.zeros(1, 1000000 - zs[-2].shape[-1]).cpu()), dim=-1).long()
        zs = model._sample(zs, labels, [0], sample_length=40*model.priors[-3].raw_to_tokens, save_results=False)
        assert torch.allclose(zs[0][0], torch.tensor(self.EXPECTED_OUTPUT_0))

    @slow
    @require_torch
    def test_slow_sampling(self):
        torch.backends.cuda.matmul.allow_tf32 = False

        model = JukeboxModel.from_pretrained(self.model_id,min_duration=0).eval()

        labels = [i.cuda() for i in self.prepare_inputs()]
        set_seed(0)
        zs = [torch.zeros(1, 0, dtype=torch.long).cuda() for _ in range(3)]

        top_prior = model.priors[-1]
        start = 0
        z_conds = top_prior.get_z_conds(zs, start=start, end=start + top_prior.n_ctx)
        y = top_prior.get_y(labels[-1].clone(), start, 1058304, 0)

        self.assertIsNone(z_conds)
        self.assertListEqual(y.cpu().numpy()[0][:10].tolist(), self.EXPECTED_Y_COND)

        set_seed(0)
        zs = [torch.zeros(1, 0, dtype=torch.long).cuda() for _ in range(3)]
        zs = model._sample(zs, labels, [2], sample_length=40*model.priors[-1].raw_to_tokens, save_results=False)
        assert torch.allclose(zs[-1][0].cpu(), torch.tensor(self.EXPECTED_GPU_OUTPUTS_2))

        zs[-1] = torch.tensor(self.EXPECTED_GPU_OUTPUTS_2).unsqueeze(0)
        set_seed(0)
        zs[-1] = torch.cat((zs[-1].cuda(), torch.zeros(1, 1000000 - zs[-1].shape[-1]).cuda()), dim=-1).long()
        zs = model._sample(zs, labels, [1], sample_length=40*model.priors[-2].raw_to_tokens, save_results=False)
        assert torch.allclose(zs[-2][0].cpu(), torch.tensor(self.EXPECTED_GPU_OUTPUTS_1))

        zs[-2] = torch.tensor(self.EXPECTED_GPU_OUTPUTS_1).unsqueeze(0)

        set_seed(0)
        zs[-2] = torch.cat((zs[-2].cuda(), torch.zeros(1, 1000000 - zs[-2].shape[-1]).cuda()), dim=-1).long()
        zs = model._sample(zs, labels, [0], sample_length=40*model.priors[-3].raw_to_tokens, save_results=False)
        assert torch.allclose(zs[0][0].cpu(), torch.tensor(self.EXPECTED_GPU_OUTPUTS_0))

    @slow
    def test_primed_sampling(self):
        torch.backends.cuda.matmul.allow_tf32 = False

        model = JukeboxModel.from_pretrained(self.model_id, min_duration=0.5).eval()
        set_seed(0)
        waveform = torch.rand((1, 5120, 1))
        tokens = [i.cuda() for i in self.prepare_inputs()]

        zs = [None, None, model.vqvae.encode(waveform, start_level=2, bs_chunks=waveform.shape[0])[0].cuda()]
        zs = model._sample(zs, tokens, sample_levels=[2], save_results=False, sample_length_in_seconds=1)
        assert torch.allclose(zs[-1][0][:40].cpu(), torch.tensor(self.EXPECTED_PRIMED_0))

        upper_2 = torch.cat((zs[-1], torch.zeros(1, 1000000 - zs[-1].shape[-1]).cuda()), dim=-1).long()
        zs = [None, model.vqvae.encode(waveform, start_level=1, bs_chunks=waveform.shape[0])[0].cuda(), upper_2]
        zs = model._sample(zs, tokens, sample_levels=[1], save_results=False, sample_length_in_seconds=1)
        assert torch.allclose(zs[1][0][:40].cpu(), torch.tensor(self.EXPECTED_PRIMED_1))

        upper_1 = torch.cat((zs[1], torch.zeros(1, 1000000 - zs[1].shape[-1]).cuda()), dim=-1).long()
        zs = [model.vqvae.encode(waveform, start_level=0, bs_chunks=waveform.shape[0])[0].cuda(), upper_1, upper_2]
        zs = model._sample(zs, tokens, sample_levels=[0], save_results=False, sample_length_in_seconds=1)
        assert torch.allclose(zs[0][0][:40].cpu(), torch.tensor(self.EXPECTED_PRIMED_2))

    @slow
    def test_vqvae(self):
        zs = torch.tensor(self.EXPECTED_OUTPUT_2)
        with torch.no_grad():
            x = self.vqvae.decode(zs, start_level=2, bs_chunks=zs.shape[0])
        assert torch.allclose(x.cpu(), torch.tensor(self.EXPECTED_GPU_OUTPUTS_0))

        zs.to("gpu")
        self.vqvae.to("gpu")
        with torch.no_grad():
            x = self.vqvae.decode(zs, start_level=2, bs_chunks=zs.shape[0])
        assert torch.allclose(x.cpu(), torch.tensor(self.EXPECTED_GPU_OUTPUTS_0))


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
        model = JukeboxModel.from_pretrained(model_id,min_duration=0).eval()

        labels = self.prepare_inputs(model_id)
        set_seed(0)
        zs = [torch.zeros(1, 0, dtype=torch.long).cpu() for _ in range(3)]
        zs = model._sample(zs, labels, [2], sample_length=40*model.priors[-1].raw_to_tokens, save_results=False)
        assert torch.allclose(zs[-1][0], torch.tensor(self.EXPECTED_OUTPUT_2))

        zs[-1] = torch.tensor(self.EXPECTED_OUTPUT_2).unsqueeze(0)
        set_seed(0)
        zs[-1] = torch.cat((zs[-1], torch.zeros(1, 1000000 - zs[-1].shape[-1]).cpu()), dim=-1).long()
        zs = model._sample(zs, labels, [1], sample_length=40*model.priors[-2].raw_to_tokens, save_results=False)
        assert torch.allclose(zs[-2][0, :80], torch.tensor(self.EXPECTED_OUTPUT_1))

        zs[-2] = torch.tensor(self.EXPECTED_OUTPUT_1).unsqueeze(0)

        set_seed(0)
        zs[-2] = torch.cat((zs[-2], torch.zeros(1, 1000000 - zs[-2].shape[-1]).cpu()), dim=-1).long()
        zs = model._sample(zs, labels, [0], sample_length=40*model.priors[-3].raw_to_tokens, save_results=False)
        assert torch.allclose(zs[0][0, :40], torch.tensor(self.EXPECTED_OUTPUT_0))

    @slow
    def test_slow_sampling(self):
        model_id = "ArthurZ/jukebox-5b-lyrics"
        model = JukeboxModel.from_pretrained(model_id).eval().to("cuda")

        labels = [i.cuda() for i in self.prepare_inputs(model_id)]
        set_seed(0)
        zs = [torch.zeros(1, 0, dtype=torch.long).cuda() for _ in range(3)]
        zs = model._sample(zs, labels, [2], sample_tokens=10, save_results=False)
        assert torch.allclose(zs[-1][0].cpu(), torch.tensor(self.EXPECTED_OUTPUT_2))

    def test_vqvae(self):
        # test encoding of an audio
        # test decoding
        # implement vavae decoding test at 3 levels using the expected outputs
        pass


if __name__ == "__main__":
    tester = Jukebox1bModelTester()
    tester.test_slow_sampling()
