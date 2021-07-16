from transformers import LayoutLMv2Tokenizer


tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")

encoding = tokenizer(
    ["a", "weirdly", "test"],
    boxes=[[423, 237, 440, 251], [427, 272, 441, 287], [419, 115, 437, 129], [961, 885, 992, 912]],
    is_split_into_words=True,
    padding="max_length",
    max_length=20,
)

print(encoding)

print(tokenizer.decode(encoding.input_ids))
