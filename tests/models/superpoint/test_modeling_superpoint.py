import inspect
import unittest
from typing import List

from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import (
        SUPERPOINT_PRETRAINED_MODEL_ARCHIVE_LIST,
        AutoModelForInterestPointDescription,
        SuperPointForInterestPointDescription,
        SuperPointModel,
    )

if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class SuperPointModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        image_width=640,
        image_height=480,
        hidden_sizes: List[int] = [64, 64, 128, 128, 256],
        descriptor_dim: int = 256,
        keypoint_threshold: float = 0.005,
        max_keypoints: int = -1,
        nms_radius: int = 4,
        border_removal_distance: int = 4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.hidden_sizes = hidden_sizes
        self.descriptor_dim = descriptor_dim
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.border_removal_distance = border_removal_distance

    def prepare_config_and_inputs(self):
        # SuperPoint expects a grayscale image as input
        pixel_values = floats_tensor([self.batch_size, 3, self.image_width, self.image_height])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return SuperPointConfig(
            hidden_sizes=self.hidden_sizes,
            descriptor_dim=self.descriptor_dim,
            keypoint_threshold=self.keypoint_threshold,
            max_keypoints=self.max_keypoints,
            nms_radius=self.nms_radius,
            border_removal_distance=self.border_removal_distance,
        )

    def create_and_check_model(self, config, pixel_values):
        model = SuperPointModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.conv_layers_sizes[-2], self.image_width // 8, self.image_height // 8),
        )

    def create_and_check_for_interest_point_description(self, config, pixel_values):
        model = SuperPointForInterestPointDescription(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.hidden_sizes[-1], self.image_size // 32, self.image_size // 32),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class SuperPointModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SuperPointModel, SuperPointForInterestPointDescription) if is_torch_available() else ()
    all_generative_model_classes = () if is_torch_available() else ()

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = SuperPointModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SuperPointConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        return

    @unittest.skip(reason="SuperPointModel does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="SuperPointModel does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="SuperPointModel does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="SuperPointModel is not trainable")
    def test_training(self):
        pass

    @unittest.skip(reason="SuperPointModel is not trainable")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SuperPoint was not designed smaller.")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="SuperPoint does not output any loss term in the forward pass")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            # SuperPoint's feature maps are of shape (batch_size, num_channels, width, height)
            for i, conv_layer_size in enumerate(self.model_tester.conv_layers_sizes[:-2]):
                self.assertListEqual(
                    list(hidden_states[i].shape[-3:]),
                    [
                        conv_layer_size,
                        self.model_tester.image_width // (2 ** (i + 1)),
                        self.model_tester.image_height // (2 ** (i + 1)),
                    ],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @slow
    def test_model_from_pretrained(self):
        for model_name in SUPERPOINT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = SuperPointModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


def prepare_imgs():
    image1 = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    image2 = Image.open("./tests/fixtures/tests_samples/COCO/000000004016.png")
    return [image1, image2]


@require_torch
@require_vision
class SuperPointModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained("stevenbucaille/superpoint") if is_vision_available() else None

    def infer_on_model(self, model):
        preprocessor = self.default_image_processor
        images = prepare_imgs()
        inputs = preprocessor(images=images, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_number_keypoints_image0 = 568
        expected_number_keypoints_image1 = 830
        expected_max_number_keypoints = max(expected_number_keypoints_image0, expected_number_keypoints_image1)

        expected_keypoints_shape = torch.Size((len(images), expected_max_number_keypoints, 2))
        expected_scores_shape = torch.Size(
            (
                len(images),
                expected_max_number_keypoints,
            )
        )
        expected_descriptors_shape = torch.Size((len(images), expected_max_number_keypoints, 256))

        # Check output shapes
        self.assertEqual(outputs.keypoints.shape, expected_keypoints_shape)
        self.assertEqual(outputs.scores.shape, expected_scores_shape)
        self.assertEqual(outputs.descriptors.shape, expected_descriptors_shape)

        expected_keypoints_image0_values = torch.tensor([[480.0, 9.0], [494.0, 9.0], [489.0, 16.0]]).to(torch_device)
        expected_scores_image0_values = torch.tensor(
            [0.0064, 0.0140, 0.0595, 0.0728, 0.5170, 0.0175, 0.1523, 0.2055, 0.0336]
        ).to(torch_device)
        expected_descriptors_image0_value = torch.tensor(-0.1096).to(torch_device)

        # Check output values
        self.assertTrue(torch.allclose(outputs.keypoints[0, :3], expected_keypoints_image0_values, atol=1e-4))
        self.assertTrue(torch.allclose(outputs.scores[0, :9], expected_scores_image0_values, atol=1e-4))
        self.assertTrue(torch.allclose(outputs.descriptors[0, 0, 0], expected_descriptors_image0_value, atol=1e-4))

        # Check mask values
        self.assertTrue(outputs.mask[0, expected_number_keypoints_image0 - 1].item() == 1)
        self.assertTrue(outputs.mask[0, expected_number_keypoints_image0].item() == 0)
        self.assertTrue(torch.all(outputs.mask[0, : expected_number_keypoints_image0 - 1]))
        self.assertTrue(torch.all(outputs.mask[1]))

    @slow
    def test_inference(self):
        model = SuperPointModel.from_pretrained("stevenbucaille/superpoint").to(torch_device)
        self.infer_on_model(model)

    @slow
    def test_auto_model_class(self):
        model = AutoModelForInterestPointDescription.from_pretrained("stevenbucaille/superpoint").to(torch_device)
        self.infer_on_model(model)
