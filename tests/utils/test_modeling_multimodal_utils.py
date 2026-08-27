# Copyright 2026 HuggingFace Inc.
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

from transformers.testing_utils import is_torch_available, require_torch


if is_torch_available():
    import torch

    from transformers import (
        Ernie4_5_VLMoeConfig,
        HunYuanVLConfig,
        Qwen2_5_VLConfig,
        Qwen2_5OmniThinkerConfig,
        Qwen2VLConfig,
        Qwen3OmniMoeThinkerConfig,
        Qwen3VLConfig,
    )
    from transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe import Ernie4_5_VLMoeModel
    from transformers.models.hunyuan_vl.modeling_hunyuan_vl import (
        get_mrope_position_ids as hunyuan_get_mrope_position_ids,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        get_mrope_position_ids as qwen2_5_omni_get_mrope_position_ids,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        get_mrope_position_ids as qwen3_omni_get_mrope_position_ids,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel


# NOTE: expected positions below were produced by each model's own `get_rope_index` on the same inputs.
# Every family defines that as a module-level function in its own `modeling_*.py`, pure in `(config, inputs)`,
# so the tests call it directly rather than through a model — see each one's docstring for what the blocks of
# positions mean.


@require_torch
class GetRopeIndexTest(unittest.TestCase):
    # a tiny token vocabulary standing in for the omni configs' placeholder/opening tokens
    TOKEN_IDS = {
        "image_token_id": 2,
        "video_token_id": 3,
        "audio_token_id": 1,
        "vision_start_token_id": 7,
        "audio_start_token_id": 4,
    }
    TEXT, VISION_END, AUDIO_END = 20, 8, 5

    def hunyuan_config(self, num_axes=3):
        """A real `HunYuanVLConfig`, with the two values this layout reads set for the tiny test grids."""
        config = HunYuanVLConfig()
        config.vision_config.spatial_merge_size = 1
        text_config = config.get_text_config()
        text_config.rope_parameters = {**(text_config.rope_parameters or {}), "mrope_section": [1] * num_axes}
        return config

    def omni_config(self, config, **knobs):
        """A real thinker config, with the placeholder ids swapped for this file's tiny vocabulary."""
        config.vision_config.spatial_merge_size = 1
        for key, value in {**self.TOKEN_IDS, **knobs}.items():
            setattr(config, key, value)
        return config

    def modality_runs(self, runs):
        """`(input_ids, mm_token_type_ids)` for a list of `(modality, length)` runs (0=text, 1=image, 2=video)."""
        token_types = []
        for modality, length in runs:
            token_types += [modality] * length
        input_ids = torch.arange(10, 10 + len(token_types)).view(1, -1)
        return input_ids, torch.tensor([token_types])

    def test_qwen2_vl_image(self):
        input_ids, token_types = self.modality_runs([(0, 2), (1, 4), (0, 1)])
        # meta device: the default config is production-sized and `get_rope_index` reads no weights
        with torch.device("meta"):
            model = Qwen2VLModel(Qwen2VLConfig(vision_config={"spatial_merge_size": 1}))
        position_ids, deltas = model.get_rope_index(input_ids, token_types, image_grid_thw=torch.tensor([[1, 2, 2]]))
        # the image's 2x2 patches share one temporal position and span 2 height / 2 width positions, so the
        # text token after them resumes at 4 rather than at 3
        self.assertEqual(
            position_ids.tolist(),
            [[[0, 1, 2, 2, 2, 2, 4]], [[0, 1, 2, 2, 3, 3, 4]], [[0, 1, 2, 3, 2, 3, 4]]],
        )
        self.assertEqual(deltas.tolist(), [[-2]])

    def test_qwen2_vl_video_temporal_scaling(self):
        input_ids, token_types = self.modality_runs([(0, 1), (2, 8), (0, 1)])
        # meta device: the default config is production-sized and `get_rope_index` reads no weights
        with torch.device("meta"):
            model = Qwen2_5_VLModel(Qwen2_5_VLConfig(vision_config={"spatial_merge_size": 1, "tokens_per_second": 2}))
        position_ids, deltas = model.get_rope_index(
            input_ids,
            token_types,
            video_grid_thw=torch.tensor([[2, 2, 2]]),
            second_per_grid_ts=torch.tensor([2.0]),
        )
        # temporal ids are `tokens_per_second * second_per_grid_ts = 4` apart: 1 for the first frame, 5 for
        # the second (qwen2_5_vl's layout; without `tokens_per_second` they would be 1 apart)
        self.assertEqual(
            position_ids.tolist(),
            [
                [[0, 1, 1, 1, 1, 5, 5, 5, 5, 3]],
                [[0, 1, 1, 2, 2, 1, 1, 2, 2, 3]],
                [[0, 1, 2, 1, 2, 1, 2, 1, 2, 3]],
            ],
        )
        self.assertEqual(deltas.tolist(), [[-4]])

    def test_qwen2_vl_split_video_frames(self):
        # processors that separate frames with timestamps emit one visual run per frame, so the single
        # `T=2` grid is expanded to two `T=1` grids, one per run
        input_ids, token_types = self.modality_runs([(0, 1), (2, 4), (0, 1), (2, 4), (0, 1)])
        # the real qwen3_vl-family override, run on a config-only carrier — every `get_rope_index`
        # is a pure function of `(self.config, inputs)`, so no weights are needed: an uninitialized
        # instance (generated overrides call zero-arg `super()`, which wants a real instance) carrying
        # nothing but a config drives it, which is also how the exporters rebuild positions

        # meta device: the default config is production-sized and `get_rope_index` reads no weights
        with torch.device("meta"):
            model = Qwen3VLModel(Qwen3VLConfig(vision_config={"spatial_merge_size": 1}))
        position_ids, deltas = model.get_rope_index(
            input_ids,
            token_types,
            video_grid_thw=torch.tensor([[2, 2, 2]]),
        )
        self.assertEqual(
            position_ids.tolist(),
            [
                [[0, 1, 1, 1, 1, 3, 4, 4, 4, 4, 6]],
                [[0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6]],
                [[0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6]],
            ],
        )
        self.assertEqual(deltas.tolist(), [[-4]])

    def test_qwen2_vl_video_temporal_merge(self):
        # ernie's temporal backbone merge: a `T=2` grid collapses to one temporal position (4 tokens)
        input_ids, token_types = self.modality_runs([(0, 1), (2, 4), (0, 1)])
        # meta device: the default config is production-sized and `get_rope_index` reads no weights
        with torch.device("meta"):
            model = Ernie4_5_VLMoeModel(
                Ernie4_5_VLMoeConfig(vision_config={"spatial_merge_size": 1, "temporal_merge_size": 2})
            )
        position_ids, deltas = model.get_rope_index(
            input_ids,
            token_types,
            video_grid_thw=torch.tensor([[2, 2, 2]]),
        )
        self.assertEqual(
            position_ids.tolist(),
            [[[0, 1, 1, 1, 1, 3]], [[0, 1, 1, 2, 2, 3]], [[0, 1, 2, 1, 2, 3]]],
        )
        self.assertEqual(deltas.tolist(), [[-2]])

    def test_qwen2_vl_ignores_padding(self):
        input_ids, token_types = self.modality_runs([(0, 2), (1, 4), (0, 1)])
        attention_mask = torch.ones_like(input_ids)
        attention_mask[:, :2] = 0
        # meta device: the default config is production-sized and `get_rope_index` reads no weights
        with torch.device("meta"):
            model = Qwen2VLModel(Qwen2VLConfig(vision_config={"spatial_merge_size": 1}))
        position_ids, deltas = model.get_rope_index(
            input_ids,
            token_types,
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            attention_mask=attention_mask,
        )
        # the two masked text tokens are dropped before laying out, so the image starts at 0 (not 2) and the
        # trailing text at 2 (not 4); masked slots keep 0
        self.assertEqual(
            position_ids.tolist(),
            [[[0, 0, 0, 0, 0, 0, 2]], [[0, 0, 0, 0, 1, 1, 2]], [[0, 0, 0, 1, 0, 1, 2]]],
        )
        self.assertEqual(deltas.tolist(), [[-2]])

    def test_hunyuan_vl_span_with_boundary_tokens(self):
        # 8 image-span tokens for a 6-token grid (2 rows x (2 width + 1 newline)): the span carries the two
        # image-boundary tokens, which keep their 1D positions
        input_ids, token_types = self.modality_runs([(0, 1), (1, 8), (0, 1)])
        position_ids, deltas = hunyuan_get_mrope_position_ids(
            self.hunyuan_config(),
            input_ids,
            token_types,
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )
        self.assertEqual(
            position_ids.tolist(),
            [
                [[0, 1, 0, 1, 2, 0, 1, 2, 8, 9]],  # width, spanning w + 1 columns
                [[0, 1, 0, 0, 0, 1, 1, 1, 8, 9]],  # height, repeated over the extra column
                [[0, 1, 0, 0, 0, 0, 0, 0, 8, 9]],  # per-image ordinal
            ],
        )
        self.assertEqual(deltas.tolist(), [[0]])

    def test_hunyuan_vl_extra_axis_keeps_1d_positions(self):
        # with 4 axes only the last three are replaced; two bare spans consume both grids and the ordinal
        # counts up per image
        input_ids, token_types = self.modality_runs([(0, 1), (1, 6), (0, 1), (1, 6)])
        position_ids, deltas = hunyuan_get_mrope_position_ids(
            self.hunyuan_config(num_axes=4),
            input_ids,
            token_types,
            image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]),
        )
        self.assertEqual(
            position_ids.tolist(),
            [
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
                [[0, 0, 1, 2, 0, 1, 2, 7, 0, 1, 2, 0, 1, 2]],
                [[0, 0, 0, 0, 1, 1, 1, 7, 0, 0, 0, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0, 0, 7, 1, 1, 1, 1, 1, 1]],
            ],
        )
        self.assertEqual(deltas.tolist(), [[0]])

    def test_hunyuan_vl_rejects_span_grid_mismatch(self):
        input_ids, token_types = self.modality_runs([(0, 1), (1, 5)])
        with self.assertRaises(ValueError):
            hunyuan_get_mrope_position_ids(
                self.hunyuan_config(),
                input_ids,
                token_types,
                image_grid_thw=torch.tensor([[1, 2, 2]]),
            )

    def audio_then_image_ids(self):
        """text, an audio span of 3 tokens, a 2x2 image span, then trailing text."""
        text, audio_end, vision_end = self.TEXT, self.AUDIO_END, self.VISION_END
        return torch.tensor([[text, text, 4, 1, 1, 1, audio_end, 7, 2, 2, 2, 2, vision_end, text]])

    def audio_in_video_ids(self):
        """text, then one span opening with both start tokens: 2x(2x2) video, 3 audio tokens, both ends."""
        text, audio_end, vision_end = self.TEXT, self.AUDIO_END, self.VISION_END
        return torch.tensor([[text, 7, 4, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, audio_end, vision_end, text]])

    def test_qwen2_5_omni_audio_then_image(self):
        input_ids = self.audio_then_image_ids()
        position_ids, deltas = qwen2_5_omni_get_mrope_position_ids(
            self.omni_config(Qwen2_5OmniThinkerConfig(), position_id_per_seconds=25, seconds_per_chunk=2),
            input_ids,
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            audio_seqlens=torch.tensor([12]),
            attention_mask=torch.ones_like(input_ids),
        )
        self.assertEqual(
            position_ids.tolist(),
            [
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 10, 11]],
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 11]],
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11]],
            ],
        )
        self.assertEqual(deltas.tolist(), [[-2]])

    def test_qwen2_5_omni_audio_in_video(self):
        input_ids = self.audio_in_video_ids()
        position_ids, deltas = qwen2_5_omni_get_mrope_position_ids(
            self.omni_config(Qwen2_5OmniThinkerConfig(), position_id_per_seconds=25, seconds_per_chunk=2),
            input_ids,
            video_grid_thw=torch.tensor([[2, 2, 2]]),
            second_per_grid_ts=torch.tensor([1.0]),
            audio_seqlens=torch.tensor([12]),
            use_audio_in_video=True,
            attention_mask=torch.ones_like(input_ids),
        )
        # the video's two temporal grids are 25 positions apart, and the audio track follows the video
        # inside the single time chunk both fit in
        self.assertEqual(
            position_ids.tolist(),
            [
                [[0, 1, 1, 2, 2, 2, 2, 27, 27, 27, 27, 2, 3, 4, 5, 5, 6]],
                [[0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 3, 4, 5, 5, 6]],
                [[0, 1, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 4, 5, 5, 6]],
            ],
        )
        self.assertEqual(deltas.tolist(), [[11]])

    def test_qwen3_omni_audio_then_image(self):
        input_ids = self.audio_then_image_ids()
        position_ids, deltas = qwen3_omni_get_mrope_position_ids(
            self.omni_config(Qwen3OmniMoeThinkerConfig(), position_id_per_seconds=25),
            input_ids,
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            audio_seqlens=torch.tensor([12]),
            attention_mask=torch.ones_like(input_ids),
        )
        self.assertEqual(position_ids.dtype, torch.float)
        self.assertEqual(
            position_ids.tolist(),
            [
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 10, 11]],
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 11]],
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 9, 10, 11]],
            ],
        )
        self.assertEqual(deltas.tolist(), [[-2]])

    def test_qwen3_omni_audio_in_video(self):
        input_ids = self.audio_in_video_ids()
        position_ids, deltas = qwen3_omni_get_mrope_position_ids(
            self.omni_config(Qwen3OmniMoeThinkerConfig(), position_id_per_seconds=25),
            input_ids,
            video_grid_thw=torch.tensor([[2, 2, 2]]),
            second_per_grid_ts=torch.tensor([1.0]),
            audio_seqlens=torch.tensor([12]),
            use_audio_in_video=True,
            attention_mask=torch.ones_like(input_ids),
        )
        # the audio tokens (positions 3, 4) merge in ahead of the video's second temporal grid (28)
        self.assertEqual(
            position_ids.tolist(),
            [
                [[0, 1, 2, 3, 3, 3, 3, 3, 4, 28, 28, 28, 28, 29, 30, 31, 32]],
                [[0, 1, 2, 3, 3, 4, 4, 3, 4, 3, 3, 4, 4, 29, 30, 31, 32]],
                [[0, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 29, 30, 31, 32]],
            ],
        )
        self.assertEqual(deltas.tolist(), [[16]])

    def test_text_only_positions_count_up(self):
        # No vision grids: the layout falls back to plain 1D positions on every axis.
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]])
        position_ids, deltas = qwen2_5_omni_get_mrope_position_ids(
            self.omni_config(Qwen2_5OmniThinkerConfig()), None, None, attention_mask=attention_mask
        )
        # padded slots keep position 1, the rest count up from 0
        self.assertEqual(position_ids.tolist(), [[[1, 1, 0, 1, 2]]] * 3)
        self.assertEqual(deltas.tolist(), [[0]])
