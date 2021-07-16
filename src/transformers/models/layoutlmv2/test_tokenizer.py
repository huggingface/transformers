from transformers import LayoutLMv2Tokenizer


tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")

# # FAILURE CASE

# encoding = tokenizer(
#     ["a", "weirdly", "test"],
#     boxes=[[423, 237, 440, 251], [427, 272, 441, 287], [419, 115, 437, 129]],
#     is_split_into_words=True,
#     word_labels=[1, 2, 3],
#     padding="max_length",
#     max_length=20,
# )

print("-----------------------------------------------------------------")
print("CASE 1: single example (training, inference) + CASE 2 (inference)")
encoding = tokenizer(
    ["a", "weirdly", "test"],
    boxes=[[423, 237, 440, 251], [427, 272, 441, 287], [419, 115, 437, 129]],
    padding="max_length",
    max_length=20,
)

print(encoding)

print(tokenizer.decode(encoding.input_ids))

print("--------------------------------------------------------------------")
print("CASE 1: batch of examples (training, inference) + CASE 2 (inference)")

encoding = tokenizer(
    [["a", "weirdly", "test"], ["hello", "my", "name", "is", "niels"]],
    boxes=[
        [[423, 237, 440, 251], [427, 272, 441, 287], [419, 115, 437, 129]],
        [[961, 885, 992, 912], [256, 38, 330, 58], [256, 38, 330, 58], [336, 42, 353, 57], [34, 42, 66, 69]],
    ],
    padding="max_length",
    max_length=20,
)

print(encoding)

print(tokenizer.decode(encoding.input_ids[0]))
print(tokenizer.decode(encoding.input_ids[1]))

print("--------------------------------------------------------------------")
print("CASE 2: single example (training)")
encoding = tokenizer(
    ["a", "weirdly", "test"],
    boxes=[[423, 237, 440, 251], [427, 272, 441, 287], [419, 115, 437, 129]],
    word_labels=[1, 2, 3],
    padding="max_length",
    max_length=20,
)

print(encoding)

print(tokenizer.decode(encoding.input_ids))

# CASE 3: single example (inference)
encoding = tokenizer(
    "what's his name?",
    [[423, 237, 440, 251], [427, 272, 441, 287], [419, 115, 437, 129]],
    ["a", "weirdly", "test"],
    padding="max_length",
    max_length=20,
)

print(encoding)

print(tokenizer.decode(encoding.input_ids))

print("----------------------------------")
print("CASE 3: batched example (inference)")
encoding = tokenizer(
    ["what's his name?", "how is he called?"],
    [
        [[423, 237, 440, 251], [427, 272, 441, 287], [419, 115, 437, 129]],
        [[256, 38, 330, 58], [256, 38, 330, 58], [336, 42, 353, 57], [34, 42, 66, 69]],
    ],
    [["a", "weirdly", "test"], ["what", "a", "laif", "gastn"]],
    padding="max_length",
    max_length=20,
)

print(encoding)

print(tokenizer.decode(encoding.input_ids[0]))
