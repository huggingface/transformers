# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Dict

import numpy as np
from huggingface_hub.utils import insecure_hashlib

from transformers import (
    MODEL_FOR_MASK_GENERATION_MAPPING,
    TF_MODEL_FOR_MASK_GENERATION_MAPPING,
    is_vision_available,
    pipeline,
)
from transformers.pipelines import MaskGenerationPipeline
from transformers.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_tf,
    require_torch,
    require_vision,
    slow,
)


if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


def hashimage(image: Image) -> str:
    m = insecure_hashlib.md5(image.tobytes())
    return m.hexdigest()[:10]


def mask_to_test_readable(mask: Image) -> Dict:
    npimg = np.array(mask)
    shape = npimg.shape
    return {"hash": hashimage(mask), "shape": shape}


@is_pipeline_test
@require_vision
@require_torch
class MaskGenerationPipelineTests(unittest.TestCase):
    model_mapping = dict(
        (list(MODEL_FOR_MASK_GENERATION_MAPPING.items()) if MODEL_FOR_MASK_GENERATION_MAPPING else [])
    )
    tf_model_mapping = dict(
        (list(TF_MODEL_FOR_MASK_GENERATION_MAPPING.items()) if TF_MODEL_FOR_MASK_GENERATION_MAPPING else [])
    )

    def get_test_pipeline(self, model, tokenizer, processor, torch_dtype="float32"):
        image_segmenter = MaskGenerationPipeline(model=model, image_processor=processor, torch_dtype=torch_dtype)
        return image_segmenter, [
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
        ]

    @unittest.skip(reason="TODO @Arthur: Implement me")
    def run_pipeline_test(self, mask_generator, examples):
        pass

    @require_tf
    @unittest.skip(reason="Image segmentation not implemented in TF")
    def test_small_model_tf(self):
        pass

    @slow
    @require_torch
    def test_small_model_pt(self):
        image_segmenter = pipeline("mask-generation", model="facebook/sam-vit-huge")

        outputs = image_segmenter("http://images.cocodataset.org/val2017/000000039769.jpg", points_per_batch=256)

        # Shortening by hashing
        new_outupt = []
        for i, o in enumerate(outputs["masks"]):
            new_outupt += [{"mask": mask_to_test_readable(o), "scores": outputs["scores"][i]}]

        # fmt: off
        self.assertEqual(
            nested_simplify(new_outupt, decimals=4),
            [
                {'mask': {'hash': '115ad19f5f', 'shape': (480, 640)}, 'scores': 1.0444},
                {'mask': {'hash': '6affa964c6', 'shape': (480, 640)}, 'scores': 1.021},
                {'mask': {'hash': 'dfe28a0388', 'shape': (480, 640)}, 'scores': 1.0167},
                {'mask': {'hash': 'c0a5f4a318', 'shape': (480, 640)}, 'scores': 1.0132},
                {'mask': {'hash': 'fe8065c197', 'shape': (480, 640)}, 'scores': 1.0053},
                {'mask': {'hash': 'e2d0b7a0b7', 'shape': (480, 640)}, 'scores': 0.9967},
                {'mask': {'hash': '453c7844bd', 'shape': (480, 640)}, 'scores': 0.993},
                {'mask': {'hash': '3d44f2926d', 'shape': (480, 640)}, 'scores': 0.9909},
                {'mask': {'hash': '64033ddc3f', 'shape': (480, 640)}, 'scores': 0.9879},
                {'mask': {'hash': '801064ff79', 'shape': (480, 640)}, 'scores': 0.9834},
                {'mask': {'hash': '6172f276ef', 'shape': (480, 640)}, 'scores': 0.9716},
                {'mask': {'hash': 'b49e60e084', 'shape': (480, 640)}, 'scores': 0.9612},
                {'mask': {'hash': 'a811e775fd', 'shape': (480, 640)}, 'scores': 0.9599},
                {'mask': {'hash': 'a6a8ebcf4b', 'shape': (480, 640)}, 'scores': 0.9552},
                {'mask': {'hash': '9d8257e080', 'shape': (480, 640)}, 'scores': 0.9532},
                {'mask': {'hash': '32de6454a8', 'shape': (480, 640)}, 'scores': 0.9516},
                {'mask': {'hash': 'af3d4af2c8', 'shape': (480, 640)}, 'scores': 0.9499},
                {'mask': {'hash': '3c6db475fb', 'shape': (480, 640)}, 'scores': 0.9483},
                {'mask': {'hash': 'c290813fb9', 'shape': (480, 640)}, 'scores': 0.9464},
                {'mask': {'hash': 'b6f0b8f606', 'shape': (480, 640)}, 'scores': 0.943},
                {'mask': {'hash': '92ce16bfdf', 'shape': (480, 640)}, 'scores': 0.943},
                {'mask': {'hash': 'c749b25868', 'shape': (480, 640)}, 'scores': 0.9408},
                {'mask': {'hash': 'efb6cab859', 'shape': (480, 640)}, 'scores': 0.9335},
                {'mask': {'hash': '1ff2eafb30', 'shape': (480, 640)}, 'scores': 0.9326},
                {'mask': {'hash': '788b798e24', 'shape': (480, 640)}, 'scores': 0.9262},
                {'mask': {'hash': 'abea804f0e', 'shape': (480, 640)}, 'scores': 0.8999},
                {'mask': {'hash': '7b9e8ddb73', 'shape': (480, 640)}, 'scores': 0.8986},
                {'mask': {'hash': 'cd24047c8a', 'shape': (480, 640)}, 'scores': 0.8984},
                {'mask': {'hash': '6943e6bcbd', 'shape': (480, 640)}, 'scores': 0.8873},
                {'mask': {'hash': 'b5f47c9191', 'shape': (480, 640)}, 'scores': 0.8871}
            ],
        )
        # fmt: on

    @require_torch
    @slow
    def test_threshold(self):
        model_id = "facebook/sam-vit-huge"
        image_segmenter = pipeline("mask-generation", model=model_id)

        outputs = image_segmenter(
            "http://images.cocodataset.org/val2017/000000039769.jpg", pred_iou_thresh=1, points_per_batch=256
        )

        # Shortening by hashing
        new_outupt = []
        for i, o in enumerate(outputs["masks"]):
            new_outupt += [{"mask": mask_to_test_readable(o), "scores": outputs["scores"][i]}]

        self.assertEqual(
            nested_simplify(new_outupt, decimals=4),
            [
                {"mask": {"hash": "115ad19f5f", "shape": (480, 640)}, "scores": 1.0444},
                {"mask": {"hash": "6affa964c6", "shape": (480, 640)}, "scores": 1.0210},
                {"mask": {"hash": "dfe28a0388", "shape": (480, 640)}, "scores": 1.0167},
                {"mask": {"hash": "c0a5f4a318", "shape": (480, 640)}, "scores": 1.0132},
                {"mask": {"hash": "fe8065c197", "shape": (480, 640)}, "scores": 1.0053},
            ],
        )
