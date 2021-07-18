from PIL import Image

from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Processor, LayoutLMv2Tokenizer


feature_extractor = LayoutLMv2FeatureExtractor()
tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
processor = LayoutLMv2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

image = Image.open("src/transformers/models/layoutlmv2/document.png").convert("RGB")

# encoding = processor(images=image, padding="max_length", return_tensors="pt")

# print(encoding.keys())

# for k, v in encoding.items():
#     print(k, v.shape)

# print(tokenizer.decode(encoding.input_ids.squeeze().tolist()))

encoding = processor(
    images=image, text=["How old is he?"], padding="max_length", max_length=20, truncation=True, return_tensors="pt"
)

print(encoding.keys())

for k, v in encoding.items():
    print(k, v.shape)

print(tokenizer.decode(encoding.input_ids.squeeze().tolist()))
print(encoding.bbox.shape)

for id, box in zip(encoding.input_ids.squeeze().tolist(), encoding.bbox.squeeze().tolist()):
    print(tokenizer.decode([id]), box)
