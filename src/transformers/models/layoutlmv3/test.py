from datasets import load_dataset

from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)


feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)

model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
image = dataset[0]["image"]
words = dataset[0]["tokens"]
boxes = dataset[0]["bboxes"]
word_labels = dataset[0]["ner_tags"]

encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)

outputs = model(**encoding)
logits = outputs.logits
