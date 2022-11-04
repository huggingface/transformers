from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests
import torch

model_name = "nielsr/clipseg-rd64-refined"
processor = CLIPSegProcessor.from_pretrained(model_name)
model = CLIPSegForImageSegmentation.from_pretrained(model_name)

def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image

image = prepare_img()
texts = ["a cat", "a remote", "a blanket"]
inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

# forward pass: return dict
with torch.no_grad():
    dict_outputs = model(**inputs, output_attentions=True)

# forward pass: return tuple
with torch.no_grad():
    tuple_outputs = model(**inputs, output_attentions=True, return_dict=False)

for idx, key in enumerate(dict_outputs.keys()):
    if idx < 3:
        assert torch.allclose(dict_outputs[key], tuple_outputs[idx])
    elif key == "vision_model_output":
        for i, vision_key in enumerate(dict_outputs[key].keys()):
            # last hidden state, pooler output
            if isinstance(dict_outputs["vision_model_output"][vision_key], torch.Tensor):
                assert torch.allclose(dict_outputs["vision_model_output"][vision_key], tuple_outputs[idx][i])
            # attentions
            else:
                print("Key:", vision_key)
                for j, value in enumerate(dict_outputs["vision_model_output"][vision_key]):
                    assert torch.allclose(value, tuple_outputs[idx][i][j])
    elif key == "decoder_output":
        for j, decoder_key in enumerate(dict_outputs["decoder_output"].keys()):
            if isinstance(dict_outputs["decoder_output"][decoder_key], torch.Tensor):
                assert torch.allclose(dict_outputs["decoder_output"][decoder_key], tuple_outputs[idx][j])

# print(len(dict_outputs), len(tuple_outputs))

# print(len(dict_outputs[-1]), len(tuple_outputs[-1]))

# print(type(dict_outputs[-1]), type(tuple_outputs[-1]))

# assert torch.allclose(dict_outputs[-1][0], tuple_outputs[-1][0])

# for x, y in zip(dict_outputs[-1], tuple_outputs[-1]):
#     assert torch.allclose(x, y)