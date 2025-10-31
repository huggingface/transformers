from transformers import Deimv2Config


def test_roundtrip():
    cfg = Deimv2Config()
    s = cfg.to_json_string()
    cfg2 = Deimv2Config.from_json_string(s)

    assert cfg2.model_type == "deimv2"
    assert cfg2.hidden_dim == cfg.hidden_dim
    assert cfg2.num_queries == cfg.num_queries
    assert cfg2.num_decoder_layers == cfg.num_decoder_layers
