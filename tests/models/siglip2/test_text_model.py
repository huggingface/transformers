from transformers import Siglip2Config, Siglip2TextModel
import torch

def test_siglip2_text_forward_shape():
    config = Siglip2Config(
    vocab_size=100,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    max_position_embeddings=10,
    pad_token_id=0,
    layer_norm_eps=1e-5,
    projection_size=32,
    attention_dropout=0.1,
    hidden_act="gelu",  # 👈 Add this line
)

    model = Siglip2TextModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    assert outputs.last_hidden_state.shape == (2, 8, config.hidden_size)
    