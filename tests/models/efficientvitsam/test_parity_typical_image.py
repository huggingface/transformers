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

import sys
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import requests
from huggingface_hub import hf_hub_download

from transformers import EfficientvitsamImageProcessor, EfficientvitsamModel, EfficientvitsamProcessor
from transformers.testing_utils import require_torch, require_vision, slow


IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
POINT_COORDS = np.array([[320, 240]])
POINT_LABELS = np.array([1])

efficientvit_path = Path(__file__).resolve().parents[3] / "efficientvit"
if str(efficientvit_path) not in sys.path:
    sys.path.insert(0, str(efficientvit_path))


def load_image():
    from PIL import Image

    response = requests.get(IMAGE_URL, timeout=60)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


@slow
@require_torch
@require_vision
class EfficientvitsamTypicalImageParityTest(unittest.TestCase):
    def test_huggingface_matches_upstream_on_typical_image(self):
        import torch

        pytest.importorskip("segment_anything")

        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
        from efficientvit.sam_model_zoo import create_efficientvit_sam_model

        from transformers.models.efficientvitsam.convert_efficientvitsam_to_hf import get_config, replace_keys

        checkpoint_path = hf_hub_download("mit-han-lab/efficientvit-sam", "efficientvit_sam_l0.pt")
        image = load_image()
        image_array = np.array(image)

        upstream_model = create_efficientvit_sam_model(
            "efficientvit-sam-l0", pretrained=True, weight_url=checkpoint_path
        ).eval()
        upstream_predictor = EfficientViTSamPredictor(upstream_model)
        upstream_predictor.set_image(image_array)
        _, upstream_iou, upstream_low_res = upstream_predictor.predict(
            point_coords=POINT_COORDS,
            point_labels=POINT_LABELS,
            multimask_output=True,
            return_logits=True,
        )

        state_dict = replace_keys(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        hf_model = EfficientvitsamModel(get_config("efficientvit-sam-l0"))
        hf_model.load_state_dict(state_dict)
        hf_model.eval()

        hf_processor = EfficientvitsamProcessor(EfficientvitsamImageProcessor())
        inputs = hf_processor(
            images=image,
            input_points=[POINT_COORDS.tolist()],
            input_labels=[POINT_LABELS.tolist()],
            return_tensors="pt",
        )

        with torch.no_grad():
            hf_outputs = hf_model(**inputs)

        torch.testing.assert_close(hf_outputs.iou_scores[0, 0], torch.tensor(upstream_iou), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(hf_outputs.pred_masks[0, 0], torch.tensor(upstream_low_res), rtol=5e-4, atol=5e-4)
