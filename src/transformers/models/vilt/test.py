from transformers import BertTokenizer, ViTFeatureExtractor, ViltConfig, ViltModel
import requests 
from PIL import Image

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
feature_extractor = ViTFeatureExtractor(size=384)

model = ViltModel(ViltConfig(image_size=384, patch_size=32))

# prepare text
text = "hello world"
input_ids = tokenizer(text, return_tensors="pt").input_ids 

# prepare image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
pixel_values = feature_extractor(image, return_tensors="pt").pixel_values

# forward pass
outputs = model(input_ids, pixel_values=pixel_values)

print(outputs.last_hidden_state.shape)

