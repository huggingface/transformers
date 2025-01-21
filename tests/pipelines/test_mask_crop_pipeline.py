import unittest
from typing import List

import numpy as np
import torch
from PIL import Image

from transformers import pipeline, AutoImageProcessor
from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils.import_utils import is_vision_available

from .test_pipelines_common import ANY

@require_vision
@require_torch
class MaskCropPipelineTests(unittest.TestCase):
    def get_test_pipeline(self):
        pipe = pipeline("mask-generation", model="facebook/sam-vit-base")
        return pipe
    
    def get_image_processor(self):
        return AutoImageProcessor.from_pretrained("facebook/sam-vit-base")