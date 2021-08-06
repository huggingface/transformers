from transformers import LayoutLMv2TokenizerFast


tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")

# test non batched
text = ["hello", "my", "name", "is", "Niels"]
boxes = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]

encoding = tokenizer(text, boxes=boxes)

print(encoding)
print("Word ids:", encoding.word_ids())

print(tokenizer.decode(encoding.input_ids))

# test batched
text = [["hello", "my", "name", "is", "Niels"], ["gent"]]
boxes = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]], [[1, 2, 3, 4]]]

encoding = tokenizer(text, boxes=boxes)

print(encoding)
print("Word ids:", encoding.word_ids())
