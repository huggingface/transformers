import pytest
import torch
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoModel
from transformers.models.grounding_dino.configuration_grounding_dino import GroundingDinoConfig

def test_bbox_embed_heads_are_independent_with_custom_config():
    config = GroundingDinoConfig(
        decoder_layers=2,
        decoder_bbox_embed_share=True,
        d_model=256,
        num_queries=1,
                             )
    model = GroundingDinoModel(config)
    assert model.bbox_embed[0] is not model.bbox_embed[1]
    original_weight = model.bbox_embed[1].layers[0].weight.clone()

    with torch.no_grad():
        model.bbox_embed[0].layers[0].weight.add_(10.0)

    assert not torch.equal(model.bbox_embed[0].layers[0].weight, original_weight)
    assert torch.equal(model.bbox_embed[1].layers[0].weight, original_weight)
