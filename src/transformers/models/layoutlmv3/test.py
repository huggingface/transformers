from transformers import LayoutLMv3Tokenizer


# pair input
question = "what's his name?"
words = ["hello", "niels"]
boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]

tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

encoding = tokenizer(
    question, words, boxes=boxes, truncation="only_second", return_overflowing_tokens=True, max_length=5
)

for id, box in zip(encoding.input_ids, encoding.bbox):
    print(tokenizer.decode([id]), box)
