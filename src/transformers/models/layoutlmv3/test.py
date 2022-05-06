from transformers import LayoutLMv3Tokenizer


# feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
# tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
# processor = LayoutLMv3Processor(feature_extractor, tokenizer)

# model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

# dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
# image = dataset[0]["image"]
# words = dataset[0]["tokens"]
# boxes = dataset[0]["bboxes"]
# word_labels = dataset[0]["ner_tags"]

# encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")

# for k, v in encoding.items():
#     print(k, v.shape)

# outputs = model(**encoding)
# logits = outputs.logits

tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

question = "what's his name?"
words = ["hello", "world", "my", "name", "is", "niels"]
boxes = [[1, 2, 3, 4], [5, 6, 7, 8], [7, 1, 9, 2], [6, 1, 9, 2], [1, 2, 7, 2], [8, 2, 8, 3]]

encoding = tokenizer(
    question,
    words,
    boxes=boxes,
)

for id, box in zip(encoding.input_ids, encoding.bbox):
    print(tokenizer.decode([id]), box)
