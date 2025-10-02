import torch
from PIL import Image
import numpy as np
from transformers import Deimv2ImageProcessor

def test_preprocess_postprocess():
    proc = Deimv2ImageProcessor(size=256)
    img = Image.fromarray((np.random.rand(256,256,3)*255).astype("uint8"))
    batch = proc.preprocess([img])
    assert "pixel_values" in batch
    dummy = {"logits": torch.randn(1, 300, 91), "pred_boxes": torch.rand(1, 300, 4)}
    res = proc.post_process_object_detection(dummy, threshold=0.9)
    assert isinstance(res, list)
    assert "scores" in res[0]
