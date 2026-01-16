import torch
from transformers import Deimv2Config
from transformers.models.deimv2.modeling_deimv2 import Deimv2ForObjectDetection

def test_forward_shapes():
    cfg = Deimv2Config()
    model = Deimv2ForObjectDetection(cfg)
    pixel_values = torch.randn(2, 3, 512, 512)
    out = model(pixel_values)
    assert out["logits"].shape[:2] == (2, cfg.num_queries)
    assert out["pred_boxes"].shape[-1] == 4
    
