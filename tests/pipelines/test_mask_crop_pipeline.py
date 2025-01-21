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
    
    def get_test_image(self) -> Image.Image:
        # Create a simple test image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        return Image.fromarray(image)

    def test_pipeline_attributes(self):
        pipe = self.get_test_pipeline()
        processor = self.get_image_processor()
        self.assertTrue(hasattr(processor, "generate_crop_boxes"))

    def test_crop_generation_basic(self):
        processor = self.get_image_processor()
        image = self.get_test_image()
        
        outputs = processor.generate_crop_boxes(
            image,
            target_size=64,
            crop_n_layers=1,
            points_per_crop=32
        )
        
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 4)  # crop_boxes, points_per_crop, cropped_images, input_labels
        
        crop_boxes, points_per_crop, cropped_images, input_labels = outputs
        
        # Check shapes and types - allow both numpy arrays and torch tensors
        self.assertTrue(isinstance(crop_boxes, (np.ndarray, torch.Tensor)))
        self.assertTrue(isinstance(points_per_crop, (np.ndarray, torch.Tensor)))
        self.assertIsInstance(cropped_images, List)
        self.assertTrue(isinstance(input_labels, (np.ndarray, torch.Tensor)))
        
        # Convert to numpy if tensor for consistent checking
        if isinstance(crop_boxes, torch.Tensor):
            crop_boxes = crop_boxes.numpy()
        if isinstance(points_per_crop, torch.Tensor):
            points_per_crop = points_per_crop.numpy()
        
        # Verify points grid properties
        self.assertEqual(points_per_crop.ndim, 4)  # [num_crops, batch, num_points, 2]
        self.assertEqual(points_per_crop.shape[-1], 2)  # x,y coordinates

    def test_crop_generation_parameters(self):
        processor = self.get_image_processor()
        image = self.get_test_image()
        
        # Test different crop_n_layers
        for n_layers in [0, 1, 2]:
            outputs = processor.generate_crop_boxes(
                image,
                target_size=64,
                crop_n_layers=n_layers,
                points_per_crop=32
            )
            # Number of crops should increase with layers
            crop_boxes = outputs[0]
            if isinstance(crop_boxes, torch.Tensor):
                crop_boxes = crop_boxes.numpy()
            self.assertGreaterEqual(len(crop_boxes), n_layers + 1)

    @slow
    def test_integration_with_pipeline(self):
        pipe = self.get_test_pipeline()
        processor = self.get_image_processor()
        image = self.get_test_image()
        
        # First generate crops
        crop_outputs = processor.generate_crop_boxes(
            image,
            target_size=64,
            crop_n_layers=1,
            points_per_crop=32
        )
        
        # Then use these crops for mask generation
        masks = pipe(
            image,
            points_per_batch=32,
            crops_n_layers=1
        )
        
        self.assertIsInstance(masks, dict)
        self.assertIn("masks", masks)
        self.assertIn("scores", masks)
        self.assertIsInstance(masks["masks"], list)
        self.assertIsInstance(masks["scores"], list)


if __name__ == "__main__":
    unittest.main()
