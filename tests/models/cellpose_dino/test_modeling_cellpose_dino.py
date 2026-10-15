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
"""Testing suite for the PyTorch Cellpose DINOv3 model."""

import os
import re
import tempfile
import unittest

import numpy as np

from transformers import CellposeDinoConfig
from transformers.image_utils import load_image
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch
    from safetensors.torch import save_file

    from transformers import CellposeDinoForInstanceSegmentation, CellposeDinoModel, CellposeImageProcessor


class CellposeDinoModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=16,
        patch_size=4,
        patch_stride=2,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        num_register_tokens=2,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_register_tokens = num_register_tokens
        self.is_training = is_training
        self.seq_length = (image_size // patch_stride) ** 2 + 1 + num_register_tokens

    def get_config(self):
        return CellposeDinoConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            num_register_tokens=self.num_register_tokens,
        )

    def prepare_config_and_inputs_for_common(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        return self.get_config(), {"pixel_values": pixel_values}


@require_torch
class CellposeDinoModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CellposeDinoModel, CellposeDinoForInstanceSegmentation) if is_torch_available() else ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = CellposeDinoModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=CellposeDinoConfig, has_text_modality=False, hidden_size=32
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = CellposeDinoForInstanceSegmentation(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        size = self.model_tester.image_size
        self.assertEqual(outputs.logits.shape, (self.model_tester.batch_size, 3, size, size))
        self.assertEqual(
            outputs.hidden_states[-1].shape, (self.model_tester.batch_size, self.model_tester.seq_length, 32)
        )

    def test_load_original_checkpoint(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        reference = CellposeDinoForInstanceSegmentation(config)
        for name, param in reference.named_parameters():
            if name.endswith("bias"):
                torch.nn.init.normal_(param)
        reference_state_dict = reference.state_dict()

        renames = [
            (r"^readout\.", "out."),
            (r"embeddings\.cls_token", "cls_token"),
            (r"embeddings\.register_tokens", "storage_tokens"),
            (r"embeddings\.patch_embeddings\.", "patch_embed.proj."),
            (r"model\.layer\.", "blocks."),
            (r"\.attention\.o_proj\.", ".attn.proj."),
            (r"\.layer_scale(\d)\.lambda1", r".ls\1.gamma"),
            (r"mlp\.up_proj\.", "mlp.fc1."),
            (r"mlp\.down_proj\.", "mlp.fc2."),
            (r"rope_embeddings\.", "rope_embed."),
        ]
        original = {}
        for key, value in reference_state_dict.items():
            if re.search(r"attention\.[qkv]_proj", key):
                continue
            original_key = key
            for pattern, replacement in renames:
                original_key = re.sub(pattern, replacement, original_key)
            original[original_key] = value.contiguous()
        for i in range(config.num_hidden_layers):
            prefix = f"encoder.model.layer.{i}.attention"
            for suffix in ("weight", "bias"):
                qkv = [reference_state_dict[f"{prefix}.{name}_proj.{suffix}"] for name in "qkv"]
                original[f"encoder.blocks.{i}.attn.qkv.{suffix}"] = torch.cat(qkv)
            original[f"encoder.blocks.{i}.attn.qkv.bias_mask"] = torch.zeros(3 * config.hidden_size)
        original["encoder.mask_token"] = torch.zeros(1, config.hidden_size)
        # original checkpoints store bf16-rounded periods, they must be loaded rather than recomputed
        original["encoder.rope_embed.periods"] = original["encoder.rope_embed.periods"].bfloat16().float()
        original["W2"] = torch.eye(3 * config.patch_stride**2).reshape(-1, 3, config.patch_stride, config.patch_stride)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            save_file(original, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})
            model, info = CellposeDinoForInstanceSegmentation.from_pretrained(tmp_dir, output_loading_info=True)

        self.assertEqual(info["missing_keys"], set())
        self.assertEqual(info["unexpected_keys"], set())
        state_dict = model.state_dict()
        for key, value in reference_state_dict.items():
            if key != "encoder.rope_embeddings.periods":
                torch.testing.assert_close(state_dict[key], value, msg=key)
        torch.testing.assert_close(
            state_dict["encoder.rope_embeddings.periods"], original["encoder.rope_embed.periods"]
        )

    def test_reverse_loading_mapping(self):
        # `out` -> `readout` only exists on the model with head
        super().test_reverse_loading_mapping(skip_base_model=True)

    @unittest.skip(reason="Cellpose does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="The input embeddings are a patch embedding convolution, not an `nn.Embedding`")
    def test_model_get_set_embeddings(self):
        pass


@require_torch
class CellposeDinoModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        # Reference: original cellpose `cpdino` on the same tiles, max |diff| 3e-2; masks identical at flow_threshold=0.4
        repo = "ArthurZ/cellpose-dino"
        processor = CellposeImageProcessor.from_pretrained(repo)
        model = CellposeDinoForInstanceSegmentation.from_pretrained(repo).to(torch_device).eval()
        image = np.array(load_image("https://www.cellpose.org/static/images/img02.png"))
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(pixel_values=inputs.pixel_values.to(torch_device))
        self.assertEqual(outputs.logits.shape, (inputs.pixel_values.shape[0], 3, 384, 384))
        expected_slice = torch.tensor(
            [
                [-0.19111869, -0.38693511, -0.24515280],
                [1.10966420, 2.56664276, 3.98162627],
                [-1.57934940, -0.08078121, 1.53366065],
            ]
        ).to(torch_device)
        torch.testing.assert_close(outputs.logits[0, :, 100, 100:103], expected_slice, rtol=1e-3, atol=1e-3)
        masks = processor.post_process_instance_segmentation(outputs, target_sizes=inputs.original_sizes)[0]["masks"]
        self.assertEqual(int(masks.max()), 100)
