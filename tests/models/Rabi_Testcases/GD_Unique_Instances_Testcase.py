import pytest
from transformers import AutoConfig
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoModel

def test_bbox_embed_instances_are_unique():
    config = AutoConfig.for_model("grounding_dino")
    config.decoder_layers = 2
    config.decoder_bbox_embed_share = True

    model = GroundingDinoModel(config)

    # Ensure each layer has a unique bbox_embed head
    assert model.bbox_embed[0] is not model.bbox_embed[1]
