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

import unittest

from transformers import CellposeConfig, pipeline
from transformers.image_utils import load_image
from transformers.testing_utils import Expectations, require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch

    from transformers import CellposeForInstanceSegmentation, CellposeImageProcessor, CellposeModel

IMAGE_URL = "https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/cellpose_img02.png"


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
class CellposeModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (CellposeModel, CellposeForInstanceSegmentation) if is_torch_available() else ()
    pipeline_model_mapping = {"image-segmentation": CellposeForInstanceSegmentation} if is_torch_available() else {}
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

    def test_reverse_loading_mapping(self):
        # `out` -> `readout` only exists on the model with head
        super().test_reverse_loading_mapping(skip_base_model=True)

    @unittest.skip(reason="The input embeddings are a patch embedding module, not an `nn.Embedding`")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="SAM attention returns `(batch_size * num_heads, seq, seq)` weights; covered by sam tests")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="Hidden states are `(batch_size, height, width, hidden_size)` maps, checked in test_model")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(
        reason="SAM attention adds the relative position bias as a float `attn_mask`, which flash SDPA rejects"
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@require_torch
class CellposeModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        # Reference: original cellpose `cpsam_v2` on the same tiles, max |diff| 3e-5; masks identical at flow_threshold=0.4
        repo = "ArthurZ/cellpose-sam-v2"
        processor = CellposeImageProcessor.from_pretrained(repo)
        model = CellposeForInstanceSegmentation.from_pretrained(repo).to(torch_device).eval()
        image = load_image(url_to_local_path(IMAGE_URL))
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(pixel_values=inputs.pixel_values.to(torch_device))
        self.assertEqual(outputs.logits.shape, (inputs.pixel_values.shape[0], 3, 256, 256))
        expectations = Expectations(
            {
                (None, None): [
                    [-0.04291397, -0.06741032, -0.08836612],
                    [-0.04544753, 0.04835206, 0.00611734],
                    [-8.55818272, -7.33160400, -5.81300449],
                ],
            }
        )
        expected_slice = torch.tensor(expectations.get_expectation()).to(torch_device)
        torch.testing.assert_close(outputs.logits[0, :, 100, 100:103], expected_slice, rtol=1e-4, atol=1e-4)
        masks = processor.post_process_instance_segmentation(outputs, target_sizes=[(image.height, image.width)])[0][
            "segmentation"
        ]
        self.assertEqual(int(masks.max()), 99)

    @slow
    @require_vision
    def test_image_segmentation_pipeline(self):
        segmenter = pipeline("image-segmentation", model="ArthurZ/cellpose-sam-v2", device=torch_device)
        cells = segmenter(url_to_local_path(IMAGE_URL))
        self.assertEqual(len(cells), 93)  # the pipeline filters scores below `threshold=0.9`
        self.assertEqual({cell["label"] for cell in cells}, {"cell"})
