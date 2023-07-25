""" Testing suite for the Tensorflow CvT model. """


from __future__ import annotations

import inspect
import unittest
from math import floor

import numpy as np

from transformers import CvtConfig
from transformers.testing_utils import require_tf, require_vision, slow
from transformers.utils import cached_property, is_tf_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import TFCvtForImageClassification, TFCvtModel
    from transformers.models.cvt.modeling_tf_cvt import TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class TFCvtConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "embed_dim"))
        self.parent.assertTrue(hasattr(config, "num_heads"))


class TFCvtModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=64,
        num_channels=3,
        embed_dim=[16, 32, 48],
        num_heads=[1, 2, 3],
        depth=[1, 2, 10],
        patch_sizes=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        stride_kv=[2, 2, 2],
        cls_token=[False, False, True],
        attention_drop_rate=[0.0, 0.0, 0.0],
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_training=True,
        use_labels=True,
        num_labels=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_sizes = patch_sizes
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.is_training = is_training
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.stride_kv = stride_kv
        self.depth = depth
        self.cls_token = cls_token
        self.attention_drop_rate = attention_drop_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            # create a random int32 tensor of given shape
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()
        return config, pixel_values, labels

    def get_config(self):
        return CvtConfig(
            image_size=self.image_size,
            num_labels=self.num_labels,
            num_channels=self.num_channels,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            patch_sizes=self.patch_sizes,
            patch_padding=self.patch_padding,
            patch_stride=self.patch_stride,
            stride_kv=self.stride_kv,
            depth=self.depth,
            cls_token=self.cls_token,
            attention_drop_rate=self.attention_drop_rate,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = TFCvtModel(config=config)
        result = model(pixel_values, training=False)
        image_size = (self.image_size, self.image_size)
        height, width = image_size[0], image_size[1]
        for i in range(len(self.depth)):
            height = floor(((height + 2 * self.patch_padding[i] - self.patch_sizes[i]) / self.patch_stride[i]) + 1)
            width = floor(((width + 2 * self.patch_padding[i] - self.patch_sizes[i]) / self.patch_stride[i]) + 1)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.embed_dim[-1], height, width))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = TFCvtForImageClassification(config)
        result = model(pixel_values, labels=labels, training=False)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_tf
class TFCvtModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Cvt
    does not use input_ids, inputs_embeds, attention_mask and seq_length.
    """

    all_model_classes = (TFCvtModel, TFCvtForImageClassification) if is_tf_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": TFCvtModel, "image-classification": TFCvtForImageClassification}
        if is_tf_available()
        else {}
    )
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFCvtModelTester(self)
        self.config_tester = TFCvtConfigTester(self, config_class=CvtConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    @unittest.skip(reason="Cvt does not output attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="Cvt does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Cvt does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skipIf(
        not is_tf_available() or len(tf.config.list_physical_devices("GPU")) == 0,
        reason="TF does not support backprop for grouped convolutions on CPU.",
    )
    def test_dataset_conversion(self):
        super().test_dataset_conversion()

    @unittest.skipIf(
        not is_tf_available() or len(tf.config.list_physical_devices("GPU")) == 0,
        reason="TF does not support backprop for grouped convolutions on CPU.",
    )
    @slow
    def test_keras_fit(self):
        super().test_keras_fit()

    @unittest.skip(reason="Get `Failed to determine best cudnn convolution algo.` error after using TF 2.12+cuda 11.8")
    def test_keras_fit_mixed_precision(self):
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        super().test_keras_fit()
        tf.keras.mixed_precision.set_global_policy("float32")

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            hidden_states = outputs.hidden_states

            expected_num_layers = len(self.model_tester.depth)
            self.assertEqual(len(hidden_states), expected_num_layers)

            # verify the first hidden states (first block)
            self.assertListEqual(
                list(hidden_states[0].shape[-3:]),
                [
                    self.model_tester.embed_dim[0],
                    self.model_tester.image_size // 4,
                    self.model_tester.image_size // 4,
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

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFCvtModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_tf
@require_vision
class TFCvtModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained(TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST[0])

    @slow
    def test_inference_image_classification_head(self):
        model = TFCvtForImageClassification.from_pretrained(TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST[0])

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="tf")

        # forward pass
        outputs = model(**inputs)

        # verify the logits
        expected_shape = tf.TensorShape((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = tf.constant([0.9285, 0.9015, -0.3150])
        self.assertTrue(np.allclose(outputs.logits[0, :3].numpy(), expected_slice, atol=1e-4))
