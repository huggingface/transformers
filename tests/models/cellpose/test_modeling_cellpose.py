# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Cellpose model."""

import os
import re
import tempfile
import unittest

import numpy as np

from transformers import CellposeConfig, pipeline
from transformers.image_utils import load_image
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file

    from transformers import CellposeForInstanceSegmentation, CellposeImageProcessor, CellposeModel

if is_vision_available():
    from PIL import Image


class CellposeModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=16,
        patch_size=4,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        mlp_dim=37,
        output_channels=8,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_dim = mlp_dim
        self.output_channels = output_channels
        self.is_training = is_training

    def get_config(self):
        return CellposeConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            mlp_dim=self.mlp_dim,
            output_channels=self.output_channels,
        )

    def prepare_config_and_inputs_for_common(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        return self.get_config(), {"pixel_values": pixel_values}


@require_torch
class CellposeModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CellposeModel, CellposeForInstanceSegmentation) if is_torch_available() else ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = CellposeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CellposeConfig, has_text_modality=False, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = CellposeForInstanceSegmentation(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        size = self.model_tester.image_size
        grid = size // self.model_tester.patch_size
        self.assertEqual(outputs.logits.shape, (self.model_tester.batch_size, 3, size, size))
        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)
        self.assertEqual(outputs.hidden_states[0].shape, (self.model_tester.batch_size, grid, grid, 32))

    def test_pixel_shuffle_matches_original_unpatchify(self):
        # The original code unpatchifies with `conv_transpose2d` on an identity kernel `W2`
        patch_size, num_outputs = 4, 3
        features = torch.randn(2, num_outputs * patch_size**2, 5, 5)
        w2 = torch.eye(num_outputs * patch_size**2).reshape(-1, num_outputs, patch_size, patch_size)
        torch.testing.assert_close(
            F.pixel_shuffle(features, patch_size), F.conv_transpose2d(features, w2, stride=patch_size)
        )

    @require_vision
    def test_image_segmentation_pipeline(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.id2label = {0: "cell"}
        model = CellposeForInstanceSegmentation(config).eval()
        segmenter = pipeline(
            "image-segmentation", model=model, image_processor=CellposeImageProcessor(tile_size=config.image_size)
        )
        image = Image.fromarray((np.random.default_rng(0).random((40, 50, 3)) * 255).astype(np.uint8))
        cells = segmenter(image)  # random weights: checks the plumbing, cells may be empty
        self.assertIsInstance(cells, list)
        for cell in cells:
            self.assertEqual(cell["label"], "cell")
            self.assertEqual(cell["mask"].size, (50, 40))

    def test_load_original_checkpoint(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        reference = CellposeForInstanceSegmentation(config)
        head_dim = config.hidden_size // config.num_attention_heads
        grid = config.image_size // config.patch_size

        renames = [
            (r"^readout\.", "out."),
            (r"layers\.(\d+)\.layer_norm", r"blocks.\1.norm"),
            (r"layers\.", "blocks."),
            (r"patch_embed\.projection\.", "patch_embed.proj."),
            (r"neck\.conv1\.", "neck.0."),
            (r"neck\.layer_norm1\.", "neck.1."),
            (r"neck\.conv2\.", "neck.2."),
            (r"neck\.layer_norm2\.", "neck.3."),
        ]
        original, expected = {}, {}
        for key, value in reference.state_dict().items():
            original_key = key
            for pattern, replacement in renames:
                original_key = re.sub(pattern, replacement, original_key)
            if "rel_pos" in key:
                # SAM tables were trained for 14x14 windows (or the 64x64 global grid): any length must load
                value = torch.randn(2 * 7 - 1 if "layers.0." in key else 2 * grid * 2 - 1, head_dim)
                expected[key] = F.interpolate(value.T[None], size=2 * grid - 1, mode="linear")[0].T
            else:
                expected[key] = value
            original[original_key] = value.contiguous()
        original["W2"] = torch.eye(3 * config.patch_size**2).reshape(-1, 3, config.patch_size, config.patch_size)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            save_file(original, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})
            model, info = CellposeForInstanceSegmentation.from_pretrained(tmp_dir, output_loading_info=True)

        self.assertEqual(info["missing_keys"], set())
        self.assertEqual(info["unexpected_keys"], set())
        state_dict = model.state_dict()
        for key, value in expected.items():
            torch.testing.assert_close(state_dict[key], value, msg=key)

    def test_reverse_loading_mapping(self):
        # `out` -> `readout` only exists on the model with head
        super().test_reverse_loading_mapping(skip_base_model=True)

    @unittest.skip(reason="Cellpose does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="The input embeddings are a patch embedding module, not an `nn.Embedding`")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="SAM attention returns `(batch_size * num_heads, seq, seq)` weights; covered by sam tests")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="Hidden states are `(batch_size, height, width, hidden_size)` maps, checked in test_model")
    def test_hidden_states_output(self):
        pass


@require_torch
class CellposeModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        # Reference: original cellpose `cpsam_v2` on the same tiles, max |diff| 3e-5; masks identical at flow_threshold=0.4
        repo = "ArthurZ/cellpose-sam-v2"
        processor = CellposeImageProcessor.from_pretrained(repo)
        model = CellposeForInstanceSegmentation.from_pretrained(repo).to(torch_device).eval()
        image = np.array(load_image("https://www.cellpose.org/static/images/img02.png"))
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(pixel_values=inputs.pixel_values.to(torch_device))
        self.assertEqual(outputs.logits.shape, (inputs.pixel_values.shape[0], 3, 256, 256))
        expected_slice = torch.tensor(
            [
                [-0.04291397, -0.06741032, -0.08836612],
                [-0.04544753, 0.04835206, 0.00611734],
                [-8.55818272, -7.33160400, -5.81300449],
            ]
        ).to(torch_device)
        torch.testing.assert_close(outputs.logits[0, :, 100, 100:103], expected_slice, rtol=1e-4, atol=1e-4)
        masks = processor.post_process_instance_segmentation(outputs, target_sizes=[image.shape[:2]])[0]["segmentation"]
        self.assertEqual(int(masks.max()), 99)

    @slow
    @require_vision
    def test_image_segmentation_pipeline(self):
        segmenter = pipeline("image-segmentation", model="ArthurZ/cellpose-sam-v2", device=torch_device)
        cells = segmenter("https://www.cellpose.org/static/images/img02.png")
        self.assertEqual(len(cells), 99)
        self.assertEqual({cell["label"] for cell in cells}, {"cell"})
