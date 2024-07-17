# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import numpy as np
import pytest

from transformers import (
    MODEL_MAPPING,
    TF_MODEL_MAPPING,
    TOKENIZER_MAPPING,
    ImageFeatureExtractionPipeline,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    pipeline,
)
from transformers.testing_utils import is_pipeline_test, nested_simplify, require_tf, require_torch


if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf

if is_vision_available():
    from PIL import Image


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@is_pipeline_test
class ImageFeatureExtractionPipelineTests(unittest.TestCase):
    model_mapping = MODEL_MAPPING
    tf_model_mapping = TF_MODEL_MAPPING

    @require_torch
    def test_small_model_pt(self):
        feature_extractor = pipeline(
            task="image-feature-extraction", model="hf-internal-testing/tiny-random-vit", framework="pt"
        )
        img = prepare_img()
        outputs = feature_extractor(img)
        self.assertEqual(
            nested_simplify(outputs[0][0]),
            [-1.417, -0.392, -1.264, -1.196, 1.648, 0.885, 0.56, -0.606, -1.175, 0.823, 1.912, 0.081, -0.053, 1.119, -0.062, -1.757, -0.571, 0.075, 0.959, 0.118, 1.201, -0.672, -0.498, 0.364, 0.937, -1.623, 0.228, 0.19, 1.697, -1.115, 0.583, -0.981])  # fmt: skip

    @require_torch
    def test_small_model_w_pooler_pt(self):
        feature_extractor = pipeline(
            task="image-feature-extraction", model="hf-internal-testing/tiny-random-vit-w-pooler", framework="pt"
        )
        img = prepare_img()
        outputs = feature_extractor(img, pool=True)
        self.assertEqual(
            nested_simplify(outputs[0]),
            [-0.056,  0.083,  0.021,  0.038,  0.242, -0.279, -0.033, -0.003, 0.200, -0.192,  0.045, -0.095, -0.077,  0.017, -0.058, -0.063, -0.029, -0.204,  0.014,  0.042,  0.305, -0.205, -0.099,  0.146, -0.287,  0.020,  0.168, -0.052,  0.046,  0.048, -0.156,  0.093])  # fmt: skip

    @require_tf
    def test_small_model_tf(self):
        feature_extractor = pipeline(
            task="image-feature-extraction", model="hf-internal-testing/tiny-random-vit-w-pooler", framework="tf"
        )
        img = prepare_img()
        outputs = feature_extractor(img)
        self.assertEqual(
            nested_simplify(outputs[0][0]),
            [-1.417, -0.392, -1.264, -1.196, 1.648, 0.885, 0.56, -0.606, -1.175, 0.823, 1.912, 0.081, -0.053, 1.119, -0.062, -1.757, -0.571, 0.075, 0.959, 0.118, 1.201, -0.672, -0.498, 0.364, 0.937, -1.623, 0.228, 0.19, 1.697, -1.115, 0.583, -0.981])  # fmt: skip

    @require_tf
    def test_small_model_w_pooler_tf(self):
        feature_extractor = pipeline(
            task="image-feature-extraction", model="hf-internal-testing/tiny-random-vit-w-pooler", framework="tf"
        )
        img = prepare_img()
        outputs = feature_extractor(img, pool=True)
        self.assertEqual(
            nested_simplify(outputs[0]),
            [-0.056,  0.083,  0.021,  0.038,  0.242, -0.279, -0.033, -0.003, 0.200, -0.192,  0.045, -0.095, -0.077,  0.017, -0.058, -0.063, -0.029, -0.204,  0.014,  0.042,  0.305, -0.205, -0.099,  0.146, -0.287,  0.020,  0.168, -0.052,  0.046,  0.048, -0.156,  0.093])  # fmt: skip

    @require_torch
    def test_image_processing_small_model_pt(self):
        feature_extractor = pipeline(
            task="image-feature-extraction", model="hf-internal-testing/tiny-random-vit", framework="pt"
        )

        # test with image processor parameters
        image_processor_kwargs = {"size": {"height": 300, "width": 300}}
        img = prepare_img()
        with pytest.raises(ValueError):
            # Image doesn't match model input size
            feature_extractor(img, image_processor_kwargs=image_processor_kwargs)

        image_processor_kwargs = {"image_mean": [0, 0, 0], "image_std": [1, 1, 1]}
        img = prepare_img()
        outputs = feature_extractor(img, image_processor_kwargs=image_processor_kwargs)
        self.assertEqual(np.squeeze(outputs).shape, (226, 32))

        # Test pooling option
        outputs = feature_extractor(img, pool=True)
        self.assertEqual(np.squeeze(outputs).shape, (32,))

    @require_tf
    def test_image_processing_small_model_tf(self):
        feature_extractor = pipeline(
            task="image-feature-extraction", model="hf-internal-testing/tiny-random-vit", framework="tf"
        )

        # test with image processor parameters
        image_processor_kwargs = {"size": {"height": 300, "width": 300}}
        img = prepare_img()
        with pytest.raises(ValueError):
            # Image doesn't match model input size
            feature_extractor(img, image_processor_kwargs=image_processor_kwargs)

        image_processor_kwargs = {"image_mean": [0, 0, 0], "image_std": [1, 1, 1]}
        img = prepare_img()
        outputs = feature_extractor(img, image_processor_kwargs=image_processor_kwargs)
        self.assertEqual(np.squeeze(outputs).shape, (226, 32))

        # Test pooling option
        outputs = feature_extractor(img, pool=True)
        self.assertEqual(np.squeeze(outputs).shape, (32,))

    @require_torch
    def test_return_tensors_pt(self):
        feature_extractor = pipeline(
            task="image-feature-extraction", model="hf-internal-testing/tiny-random-vit", framework="pt"
        )
        img = prepare_img()
        outputs = feature_extractor(img, return_tensors=True)
        self.assertTrue(torch.is_tensor(outputs))

    @require_tf
    def test_return_tensors_tf(self):
        feature_extractor = pipeline(
            task="image-feature-extraction", model="hf-internal-testing/tiny-random-vit", framework="tf"
        )
        img = prepare_img()
        outputs = feature_extractor(img, return_tensors=True)
        self.assertTrue(tf.is_tensor(outputs))

    def get_test_pipeline(self, model, tokenizer, processor, torch_dtype="float32"):
        if processor is None:
            self.skipTest(reason="No image processor")

        elif type(model.config) in TOKENIZER_MAPPING:
            self.skipTest(
                reason="This is a bimodal model, we need to find a more consistent way to switch on those models."
            )

        elif model.config.is_encoder_decoder:
            self.skipTest(
                """encoder_decoder models are trickier for this pipeline.
                Do we want encoder + decoder inputs to get some featues?
                Do we want encoder only features ?
                For now ignore those.
                """
            )

        feature_extractor = ImageFeatureExtractionPipeline(
            model=model, image_processor=processor, torch_dtype=torch_dtype
        )
        img = prepare_img()
        return feature_extractor, [img, img]

    def run_pipeline_test(self, feature_extractor, examples):
        imgs = examples
        outputs = feature_extractor(imgs[0])

        self.assertEqual(len(outputs), 1)

        outputs = feature_extractor(imgs)
        self.assertEqual(len(outputs), 2)
