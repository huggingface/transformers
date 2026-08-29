# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import inspect
import unittest

from transformers.models.roma import RomaConfig, RomaModel
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import RomaForKeypointMatching

if is_vision_available():
    from transformers import AutoImageProcessor


class RomaModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=42,  # must be a multiple of the backbone patch size (14)
        num_samples=20,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_samples = num_samples

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, 2, 3, self.image_size, self.image_size])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return RomaConfig(
            backbone_config={
                "model_type": "dinov2",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "image_size": self.image_size,
                "patch_size": 14,
                "out_indices": [-1],
            },
            cnn_feature_dims=[4, 8, 8, 16],
            gp_dim=16,
            coarse_feature_dim=16,
            proj_out_dims=[16, 16, 16, 8, 9],
            anchor_resolution=4,
            num_decoder_layers=2,
            decoder_num_attention_heads=2,
            refiner_hidden_dims=[64, 48, 48, 24, 24],
            refiner_displacement_emb_dims=[4, 4, 4, 4, 4],
            refiner_local_corr_radius=[2, 1, 1, 0, 0],
            refiner_hidden_blocks=1,
            num_samples=self.num_samples,
        )

    def create_and_check_model(self, config, pixel_values):
        model = RomaForKeypointMatching(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # symmetric (default) => the dense warp/certainty width is doubled.
        self.parent.assertEqual(result.warp.shape, (self.batch_size, self.image_size, 2 * self.image_size, 4))
        self.parent.assertEqual(result.certainty.shape, (self.batch_size, self.image_size, 2 * self.image_size))
        self.parent.assertEqual(result.matches.shape, (self.batch_size, self.num_samples, 4))
        self.parent.assertEqual(result.matching_scores.shape, (self.batch_size, self.num_samples))

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class RomaModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (RomaForKeypointMatching, RomaModel) if is_torch_available() else ()
    test_resize_embeddings = False
    test_head_masking = False
    test_pruning = False
    has_attentions = False

    def setUp(self):
        self.model_tester = RomaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=RomaConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_upsample(self):
        # The high-resolution refinement pass should return a dense warp at the upsample resolution.
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        config.upsample_predictions = True
        model = RomaForKeypointMatching(config).to(torch_device).eval()
        # The refinement resolution is determined by the high-res input, not the config.
        pixel_values_upsampled = floats_tensor([self.model_tester.batch_size, 2, 3, 64, 64])
        with torch.no_grad():
            out = model(pixel_values, pixel_values_upsampled=pixel_values_upsampled)
        # symmetric -> the warp width is doubled.
        self.assertEqual(out.warp.shape, (self.model_tester.batch_size, 64, 128, 4))
        # without the high-res input it falls back to the coarse resolution.
        with torch.no_grad():
            out_coarse = model(pixel_values)
        self.assertEqual(out_coarse.warp.shape[1], self.model_tester.image_size)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            self.assertListEqual(arg_names[:1], ["pixel_values"])

    def test_forward_labels_should_raise(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = RomaForKeypointMatching(config).to(torch_device).eval()
        with torch.no_grad(), self.assertRaises(NotImplementedError):
            model(inputs_dict["pixel_values"], labels=torch.rand(4, 4))

    @unittest.skip(reason="RoMa does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="RoMa does not support input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="RoMa does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="RoMa does not return hidden_states or attentions in its output")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="RoMa does not return hidden_states or attentions in its output")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="RoMa samples matches with multinomial, so outputs are not deterministic")
    def test_determinism(self):
        pass

    @unittest.skip(reason="RoMa samples matches with multinomial, so outputs are not deterministic")
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="RoMa samples matches with multinomial, so outputs are not deterministic")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="RoMa samples matches with multinomial, so save/load outputs are not bit-identical")
    def test_save_load(self):
        pass

    @unittest.skip(reason="RoMa does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="RoMa does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="RoMa does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skip(reason="RoMa does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="RoMa does not output a loss")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = RomaForKeypointMatching.from_pretrained("Parskatt/roma_outdoor")
        self.assertIsNotNone(model)


@require_torch
@require_vision
@slow
class RomaModelIntegrationTest(unittest.TestCase):
    def test_inference(self):
        from transformers.image_utils import load_image

        processor = AutoImageProcessor.from_pretrained("Parskatt/roma_outdoor")
        model = RomaForKeypointMatching.from_pretrained("Parskatt/roma_outdoor").to(torch_device).eval()
        image1 = load_image(
            "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
        )
        image2 = load_image(
            "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
        )
        inputs = processor([[image1, image2]], return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs = model(**inputs)
        self.assertEqual(outputs.warp.shape[-1], 4)
        self.assertTrue((outputs.certainty >= 0).all() and (outputs.certainty <= 1).all())
