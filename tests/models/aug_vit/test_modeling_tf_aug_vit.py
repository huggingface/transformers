# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

from transformers import is_tf_available, AugViTConfig
from transformers.testing_utils import require_tf, slow, require_vision
from transformers.utils import cached_property, is_tf_available, is_vision_available
from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin

if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TFAugViTForImageClassification
    )


class TFAugViTModelTester:
    def __init__(
        self,
        image_size = 224,
        patch_size = 32,
        num_classes = 1000,
        dim = 128,
        depth = 2,
        heads = 16,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1,
        num_channels=3,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.num_channels=num_channels

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([4,  self.image_size, self.image_size,self.num_channels])

        labels = None

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return AugViTConfig(
            image_size = self.image_size,
            patch_size = self.patch_size,
            num_classes = self.num_classes,
            dim = self.dim,
            depth = self.depth,
            heads = self.heads,
            mlp_dim = self.mlp_dim,
            dropout = self.dropout,
            emb_dropout =self.emb_dropout
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_classes = self.type_sequence_label_size
        model = TFAugViTForImageClassification(config)
        result = model(pixel_values, training=False)
        self.parent.assertEqual(result.shape, (4, 1000))

        # Test with an image with different size than the one specified in config.
        image_size = self.image_size // 2
        pixel_values = pixel_values[:, :, :image_size, :image_size]
        result = model(pixel_values, training=False)
        self.parent.assertEqual(result.shape, (4,1000))


    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_tf
class TFAugViTModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_tf_common.py, as ViT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (TFAugViTForImageClassification) if is_tf_available() else ()
    pipeline_model_mapping = (
        {"image-classification": TFAugViTForImageClassification}
        if is_tf_available()
        else {}
    )

    test_resize_embeddings = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFAugViTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AugViTConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="AugViT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="AugViT does not use inputs_embeds")
    def test_graph_mode_with_inputs_embeds(self):
        pass


    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)


if is_vision_available():
    from PIL import Image

# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_tf
@require_vision
class TFViTModelIntegrationTest(unittest.TestCase):
    
    @slow
    def test_inference_image_classification_head(self):
        model = TFAugViTForImageClassification.from_pretrained("tensorgirl/TFaugvit")

        image = prepare_img()

        # forward pass
        outputs = model({'pixel_values':image})

        # verify the logits
        expected_shape = tf.TensorShape((1, 1000))
        self.assertEqual(outputs.shape, expected_shape)

        