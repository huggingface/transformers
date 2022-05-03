from datasets import load_dataset

from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Processor,
    LayoutLMv3Tokenizer,
)


feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)

model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
image = dataset[0]["image"]
words = dataset[0]["tokens"]
boxes = dataset[0]["bboxes"]

encoding = processor(image, words, boxes=boxes, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)

outputs = model(**encoding)
logits = outputs.logits
