# vision_config = CogVLMVisionConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=4*32, num_channels=3, image_size=224, patch_size=16,)
# config = CogVLMConfig(num_hidden_layers=2, hidden_size=32, intermediate_size=4*32, vocab_size=99,
#                       num_attention_heads=2, max_position_embeddings=512, vision_config=vision_config.to_dict())
# model = CogVLMForCausalLM(config=config)
# model.push_to_hub("nielsr/cogvlm-tiny-random")
import torch

from transformers import CogVLMForCausalLM, ViTImageProcessor


processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
reload_model = CogVLMForCausalLM.from_pretrained("nielsr/cogvlm-tiny-random")

# dummy forward pass
with torch.no_grad():
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    token_type_ids = torch.zeros_like(input_ids)
    import requests
    from PIL import Image

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    outputs = reload_model(input_ids=input_ids, token_type_ids=token_type_ids, pixel_values=pixel_values)
    print(outputs.logits.shape)
    print(outputs.logits[0, :3, :3])

    expected_slice = torch.tensor([[-0.3000, -0.1683, 0.0200], [0.2676, 0.1951, -0.0081], [0.2811, 0.2266, -0.0005]])

    assert torch.allclose(outputs.logits[0, :3, :3], expected_slice, atol=1e-4)
    print("Looks ok!")
