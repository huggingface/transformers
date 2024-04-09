# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from unittest import skip

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    slow,
    torch_device,
)
from transformers.trainer_utils import set_seed


if is_torch_available():
    import torch

    from transformers import JukeboxModel, JukeboxPrior, JukeboxTokenizer


@require_torch
class Jukebox1bModelTester(unittest.TestCase):
    all_model_classes = (JukeboxModel,) if is_torch_available() else ()
    model_id = "openai/jukebox-1b-lyrics"
    metas = {
        "artist": "Zac Brown Band",
        "genres": "Country",
        "lyrics": """I met a traveller from an antique land,
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
    }
    # fmt: off
    EXPECTED_OUTPUT_2 = [
        1864, 1536, 1213, 1870, 1357, 1536, 519, 880, 1323, 789, 1082, 534,
        1000, 1445, 1105, 1130, 967, 515, 1434, 1620, 534, 1495, 283, 1445,
        333, 1307, 539, 1631, 1528, 375, 1434, 673, 627, 710, 778, 1883,
        1405, 1276, 1455, 1228
    ]

    EXPECTED_OUTPUT_2_PT_2 = [
        1489, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653
    ]

    EXPECTED_OUTPUT_1 = [
        1125, 1751, 697, 1776, 1141, 1476, 391, 697, 1125, 684, 867, 416,
        844, 1372, 1274, 717, 1274, 844, 1299, 1419, 697, 1370, 317, 1125,
        191, 1440, 1370, 1440, 1370, 282, 1621, 1370, 368, 349, 867, 1872,
        1262, 869, 1728, 747
    ]
    EXPECTED_OUTPUT_1_PT_2 = [
        416, 416, 1125, 1125, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416
    ]

    EXPECTED_OUTPUT_0 = [
        1755, 842, 307, 1843, 1022, 1395, 234, 1554, 806, 739, 1022, 442,
        616, 556, 268, 1499, 933, 457, 1440, 1837, 755, 985, 308, 902,
        293, 1443, 1671, 1141, 1533, 555, 1562, 1061, 287, 417, 1022, 2008,
        1186, 1015, 1777, 268
    ]
    EXPECTED_OUTPUT_0_PT_2 = [
        854, 842, 1353, 114, 1353, 842, 185, 842, 185, 114, 591, 842,
        185, 417, 185, 842, 307, 842, 591, 842, 185, 842, 307, 842,
        591, 842, 1353, 842, 185, 842, 591, 842, 591, 114, 591, 842,
        185, 842, 591, 89
    ]

    EXPECTED_Y_COND = [1058304, 0, 786432, 7169, 507, 76, 27, 40, 30, 76]

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
    EXPECTED_VQVAE_ENCODE = [
        390, 1160, 1002, 1907, 1788, 1788, 1788, 1907, 1002, 1002, 1854, 1002,
        1002, 1002, 1002, 1002, 1002, 1160, 1160, 1606, 596, 596, 1160, 1002,
        1516, 596, 1002, 1002, 1002, 1907, 1788, 1788, 1788, 1854, 1788, 1907,
        1907, 1788, 596, 1626
    ]
    EXPECTED_VQVAE_DECODE = [
        -0.0492, -0.0524, -0.0565, -0.0640, -0.0686, -0.0684, -0.0677, -0.0664,
        -0.0605, -0.0490, -0.0330, -0.0168, -0.0083, -0.0075, -0.0051, 0.0025,
        0.0136, 0.0261, 0.0386, 0.0497, 0.0580, 0.0599, 0.0583, 0.0614,
        0.0740, 0.0889, 0.1023, 0.1162, 0.1211, 0.1212, 0.1251, 0.1336,
        0.1502, 0.1686, 0.1883, 0.2148, 0.2363, 0.2458, 0.2507, 0.2531
    ]
    EXPECTED_AUDIO_COND = [
        0.0256, -0.0544, 0.1600, -0.0032, 0.1066, 0.0825, -0.0013, 0.3440,
        0.0210, 0.0412, -0.1777, -0.0892, -0.0164, 0.0285, -0.0613, -0.0617,
        -0.0137, -0.0201, -0.0175, 0.0215, -0.0627, 0.0520, -0.0730, 0.0970,
        -0.0100, 0.0442, -0.0586, 0.0207, -0.0015, -0.0082
    ]
    EXPECTED_META_COND = [
        0.0415, 0.0877, 0.0022, -0.0055, 0.0751, 0.0334, 0.0324, -0.0068,
        0.0011, 0.0017, -0.0676, 0.0655, -0.0143, 0.0399, 0.0303, 0.0743,
        -0.0168, -0.0394, -0.1113, 0.0124, 0.0442, 0.0267, -0.0003, -0.1536,
        -0.0116, -0.1837, -0.0180, -0.1026, -0.0777, -0.0456
    ]
    EXPECTED_LYRIC_COND = [
        76, 27, 40, 30, 76, 46, 44, 47, 40, 37, 38, 31, 45, 45, 76, 38, 31, 33,
        45, 76, 41, 32, 76, 45, 46, 41, 40, 31, 78, 76
    ]
    # fmt: on

    def prepare_inputs(self):
        tokenizer = JukeboxTokenizer.from_pretrained(self.model_id)
        tokens = tokenizer(**self.metas)["input_ids"]
        return tokens

    @slow
    def test_sampling(self):
        model = JukeboxModel.from_pretrained(self.model_id, min_duration=0).eval()
        labels = self.prepare_inputs()

        set_seed(0)
        zs = [torch.zeros(1, 0, dtype=torch.long).cpu() for _ in range(3)]
        zs = model._sample(zs, labels, [0], sample_length=40 * model.priors[0].raw_to_tokens, save_results=False)
        self.assertIn(zs[0][0].detach().cpu().tolist(), [self.EXPECTED_OUTPUT_2, self.EXPECTED_OUTPUT_2_PT_2])

        set_seed(0)
        zs = model._sample(zs, labels, [1], sample_length=40 * model.priors[1].raw_to_tokens, save_results=False)
        self.assertIn(zs[1][0].detach().cpu().tolist(), [self.EXPECTED_OUTPUT_1, self.EXPECTED_OUTPUT_1_PT_2])

        set_seed(0)
        zs = model._sample(zs, labels, [2], sample_length=40 * model.priors[2].raw_to_tokens, save_results=False)
        self.assertIn(zs[2][0].detach().cpu().tolist(), [self.EXPECTED_OUTPUT_0, self.EXPECTED_OUTPUT_0_PT_2])

    @slow
    def test_conditioning(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = JukeboxModel.from_pretrained(self.model_id, min_duration=0).eval()

        labels = self.prepare_inputs()
        set_seed(0)
        zs = [torch.zeros(1, 0, dtype=torch.long) for _ in range(3)]

        top_prior = model.priors[0]
        start = 0
        music_token_conds = top_prior.get_music_tokens_conds(zs, start=start, end=start + top_prior.n_ctx)
        metadata = top_prior.get_metadata(labels[0].clone(), start, 1058304, 0)

        self.assertIsNone(music_token_conds)
        self.assertListEqual(metadata.numpy()[0][:10].tolist(), self.EXPECTED_Y_COND)

        audio_conditioning, metadata_conditioning, lyric_tokens = top_prior.get_cond(music_token_conds, metadata)
        torch.testing.assert_close(
            audio_conditioning[0][0][:30].detach(), torch.tensor(self.EXPECTED_AUDIO_COND), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            metadata_conditioning[0][0][:30].detach(), torch.tensor(self.EXPECTED_META_COND), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            lyric_tokens[0, :30].detach(), torch.tensor(self.EXPECTED_LYRIC_COND), atol=1e-4, rtol=1e-4
        )

    @slow
    def test_primed_sampling(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        model = JukeboxModel.from_pretrained(self.model_id, min_duration=0).eval()
        set_seed(0)
        waveform = torch.rand((1, 5120, 1))
        tokens = list(self.prepare_inputs())

        zs = [model.vqvae.encode(waveform, start_level=2, bs_chunks=waveform.shape[0])[0], None, None]
        zs = model._sample(
            zs, tokens, sample_levels=[0], save_results=False, sample_length=40 * model.priors[0].raw_to_tokens
        )
        torch.testing.assert_close(zs[0][0][:40], torch.tensor(self.EXPECTED_PRIMED_0))

        upper_2 = torch.cat((zs[0], torch.zeros(1, 2048 - zs[0].shape[-1])), dim=-1).long()
        zs = [upper_2, model.vqvae.encode(waveform, start_level=1, bs_chunks=waveform.shape[0])[0], None]
        zs = model._sample(
            zs, tokens, sample_levels=[1], save_results=False, sample_length=40 * model.priors[1].raw_to_tokens
        )
        torch.testing.assert_close(zs[1][0][:40], torch.tensor(self.EXPECTED_PRIMED_1))

        upper_1 = torch.cat((zs[1], torch.zeros(1, 2048 - zs[1].shape[-1])), dim=-1).long()
        zs = [upper_2, upper_1, model.vqvae.encode(waveform, start_level=0, bs_chunks=waveform.shape[0])[0]]
        zs = model._sample(
            zs, tokens, sample_levels=[2], save_results=False, sample_length=40 * model.priors[2].raw_to_tokens
        )
        torch.testing.assert_close(zs[2][0][:40].cpu(), torch.tensor(self.EXPECTED_PRIMED_2))

    @slow
    def test_vqvae(self):
        model = JukeboxModel.from_pretrained(self.model_id, min_duration=0).eval()
        set_seed(0)
        x = torch.rand((1, 5120, 1))
        with torch.no_grad():
            zs = model.vqvae.encode(x, start_level=2, bs_chunks=x.shape[0])
        torch.testing.assert_close(zs[0][0], torch.tensor(self.EXPECTED_VQVAE_ENCODE))

        with torch.no_grad():
            x = model.vqvae.decode(zs, start_level=2, bs_chunks=x.shape[0])
        torch.testing.assert_close(x[0, :40, 0], torch.tensor(self.EXPECTED_VQVAE_DECODE), atol=1e-4, rtol=1e-4)


@require_torch
class Jukebox5bModelTester(unittest.TestCase):
    all_model_classes = (JukeboxModel,) if is_torch_available() else ()
    model_id = "openai/jukebox-5b-lyrics"
    metas = {
        "artist": "Zac Brown Band",
        "genres": "Country",
        "lyrics": """I met a traveller from an antique land,
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
    }

    # fmt: off
    EXPECTED_OUTPUT_2 = [
        1489, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        1489, 1489, 1489, 1489, 1150, 1853, 1509, 1150, 1357, 1509, 6, 1272
    ]
    EXPECTED_OUTPUT_2_PT_2 = [
        1489, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653
    ]

    EXPECTED_OUTPUT_1 = [
        1125, 416, 1125, 1125, 1125, 1125, 1125, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416
    ]
    EXPECTED_OUTPUT_1_PT_2 = [
        416, 416, 1125, 1125, 416, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416
    ]

    EXPECTED_OUTPUT_0 = [
        1755, 1061, 234, 1755, 1061, 1755, 185, 290, 307, 307, 616, 616,
        616, 616, 616, 616, 307, 290, 417, 1755, 234, 1755, 185, 290,
        290, 290, 307, 616, 616, 616, 616, 616, 290, 234, 234, 1755,
        234, 234, 1755, 234, 185, 185, 307, 616, 616, 616, 616, 290,
        1755, 1755, 1755, 234, 234, 1755, 1572, 290, 307, 616, 34, 616
    ]
    EXPECTED_OUTPUT_0_PT_2 = [
        854, 842, 1353, 114, 1353, 842, 185, 842, 185, 114, 591, 842, 185,
        417, 185, 842, 307, 842, 591, 842, 185, 842, 185, 842, 591, 842,
        1353, 842, 185, 842, 591, 842, 591, 114, 591, 842, 185, 842, 591,
        89, 591, 842, 591, 842, 591, 417, 1372, 842, 1372, 842, 34, 842,
        185, 89, 591, 842, 185, 842, 591, 632
    ]

    EXPECTED_GPU_OUTPUTS_2 = [
        1489, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653
    ]
    EXPECTED_GPU_OUTPUTS_2_PT_2 = [
        1489, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653, 653,
        653, 653, 653, 653, 653, 653, 653, 1853, 1177, 1536, 1228,
        710, 475, 1489, 1229, 1224, 231, 1224, 252, 1434, 653, 475,
        1106, 1877, 1599, 1228, 1600, 1683, 1182, 1853, 475, 1864,
        252, 1229, 1434, 2001
    ]

    EXPECTED_GPU_OUTPUTS_1 = [
        1125, 1125, 416, 1125, 1125, 416, 1125, 1125, 416, 416, 1125, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416,
        416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416, 416
    ]
    EXPECTED_GPU_OUTPUTS_0 = [
        491, 1755, 34, 1613, 1755, 417, 992, 1613, 222, 842, 1353, 1613,
        844, 632, 185, 1613, 844, 632, 185, 1613, 185, 842, 677, 1613,
        185, 114, 1353, 1613, 307, 89, 844, 1613, 307, 1332, 234, 1979,
        307, 89, 1353, 616, 34, 842, 185, 842, 34, 842, 185, 842,
        307, 114, 185, 89, 34, 1268, 185, 89, 34, 842, 185, 89
    ]
    # fmt: on

    def prepare_inputs(self, model_id):
        tokenizer = JukeboxTokenizer.from_pretrained(model_id)
        tokens = tokenizer(**self.metas)["input_ids"]
        return tokens

    @slow
    def test_sampling(self):
        model = JukeboxModel.from_pretrained(self.model_id, min_duration=0).eval()
        labels = self.prepare_inputs(self.model_id)

        set_seed(0)
        zs = [torch.zeros(1, 0, dtype=torch.long).cpu() for _ in range(3)]
        zs = model._sample(zs, labels, [0], sample_length=60 * model.priors[0].raw_to_tokens, save_results=False)
        self.assertIn(zs[0][0].detach().cpu().tolist(), [self.EXPECTED_OUTPUT_2, self.EXPECTED_OUTPUT_2_PT_2])

        set_seed(0)
        zs = model._sample(zs, labels, [1], sample_length=60 * model.priors[1].raw_to_tokens, save_results=False)
        self.assertIn(zs[1][0].detach().cpu().tolist(), [self.EXPECTED_OUTPUT_1, self.EXPECTED_OUTPUT_1_PT_2])

        set_seed(0)
        zs = model._sample(zs, labels, [2], sample_length=60 * model.priors[2].raw_to_tokens, save_results=False)
        self.assertIn(zs[2][0].detach().cpu().tolist(), [self.EXPECTED_OUTPUT_0, self.EXPECTED_OUTPUT_0_PT_2])

    @slow
    @require_torch_accelerator
    @skip("Not enough GPU memory on CI runners")
    def test_slow_sampling(self):
        model = JukeboxModel.from_pretrained(self.model_id, min_duration=0).eval()
        labels = [i.to(torch_device) for i in self.prepare_inputs(self.model_id)]

        set_seed(0)
        model.priors[0].to(torch_device)
        zs = [torch.zeros(1, 0, dtype=torch.long).to(torch_device) for _ in range(3)]
        zs = model._sample(zs, labels, [0], sample_length=60 * model.priors[0].raw_to_tokens, save_results=False)
        torch.testing.assert_close(zs[0][0].cpu(), torch.tensor(self.EXPECTED_GPU_OUTPUTS_2))
        model.priors[0].cpu()

        set_seed(0)
        model.priors[1].to(torch_device)
        zs = model._sample(zs, labels, [1], sample_length=60 * model.priors[1].raw_to_tokens, save_results=False)
        torch.testing.assert_close(zs[1][0].cpu(), torch.tensor(self.EXPECTED_GPU_OUTPUTS_1))
        model.priors[1].cpu()

        set_seed(0)
        model.priors[2].to(torch_device)
        zs = model._sample(zs, labels, [2], sample_length=60 * model.priors[2].raw_to_tokens, save_results=False)
        torch.testing.assert_close(zs[2][0].cpu(), torch.tensor(self.EXPECTED_GPU_OUTPUTS_0))

    @slow
    @require_torch_accelerator
    @require_torch_fp16
    def test_fp16_slow_sampling(self):
        prior_id = "ArthurZ/jukebox_prior_0"
        model = JukeboxPrior.from_pretrained(prior_id, min_duration=0).eval().half().to(torch_device)

        labels = self.prepare_inputs(prior_id)[0].to(torch_device)
        metadata = model.get_metadata(labels, 0, 7680, 0)
        set_seed(0)
        outputs = model.sample(1, metadata=metadata, sample_tokens=60)
        self.assertIn(outputs[0].cpu().tolist(), [self.EXPECTED_GPU_OUTPUTS_2, self.EXPECTED_GPU_OUTPUTS_2_PT_2])
