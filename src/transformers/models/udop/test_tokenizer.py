import torch

from transformers import UdopTokenizer, UdopTokenizerFast


tokenizer = UdopTokenizer.from_pretrained("nielsr/udop-large")
fast_tokenizer = UdopTokenizerFast.from_pretrained("nielsr/udop-large")


# CASE 1: only words + boxes (non-batched)

text = ["hello", "niels"]
boxes = [[1, 8, 7, 9], [1, 2, 3, 4]]

encoding = tokenizer(text, boxes=boxes, return_tensors="pt")
fast_encoding = fast_tokenizer(text, boxes=boxes, return_tensors="pt")

for k, v in encoding.items():
    assert torch.allclose(v, fast_encoding[k])

print("SLOW TOKENIZER OUTPUTS:")

for id, box in zip(encoding.input_ids.squeeze(), encoding.bbox.squeeze()):
    print(tokenizer.decode(id), box)

# CASE 1: only words + boxes (batched)

text = [["hello", "niels", "test"], ["hello", "weirdly", "world"]]
boxes = [[[1, 8, 7, 9], [1, 2, 3, 4], [6, 12, 5, 2]], [[1, 8, 7, 9], [1, 2, 3, 4], [7, 2, 9, 2]]]

encoding = tokenizer(text, boxes=boxes, padding=True, return_tensors="pt")
fast_encoding = fast_tokenizer(text, boxes=boxes, padding=True, return_tensors="pt")

for k, v in encoding.items():
    assert torch.allclose(v, fast_encoding[k])

print("SLOW TOKENIZER OUTPUTS:")
for id, box in zip(encoding.input_ids[0], encoding.bbox[0]):
    print(tokenizer.decode(id), box)
for id, box in zip(encoding.input_ids[1], encoding.bbox[1]):
    print(tokenizer.decode(id), box)


# CASE 2: words + boxes + word labels (non-batched)
print("------- CASE 2 ---------")
text = ["hello", "niels"]
boxes = [[1, 8, 7, 9], [1, 2, 3, 4]]
word_labels = [1, 2]

encoding = tokenizer(text, boxes=boxes, word_labels=word_labels, return_tensors="pt")

print("SLOW TOKENIZER OUTPUTS:")
for id, box, label in zip(encoding.input_ids.squeeze(), encoding.bbox.squeeze(), encoding.labels.squeeze()):
    print(tokenizer.decode(id), box, label.item())

# CASE 2: words + boxes + word labels (batched)
print("------- CASE 2 (batched) ---------")
text = [["hello", "niels", "test"], ["hello", "weirdly", "world"]]
boxes = [[[1, 8, 7, 9], [1, 2, 3, 4], [6, 12, 5, 2]], [[1, 8, 7, 9], [1, 2, 3, 4], [7, 2, 9, 2]]]
word_labels = [[1, 2, 3], [1, 2, 3]]

encoding = tokenizer(text, boxes=boxes, word_labels=word_labels, padding=True, return_tensors="pt")

print("SLOW TOKENIZER OUTPUTS:")
for id, box, label in zip(encoding.input_ids[1], encoding.bbox[1], encoding.labels[1]):
    print(tokenizer.decode(id), box, label.item())

# # CASE 3: question + words + boxes (non-batched)
# question = "what is the meaning of life?"
# text = ["hello", "niels"]
# boxes = [[1, 8, 7, 9], [1, 2, 3, 4]]

# encoding = tokenizer(question, text, boxes=boxes, return_tensors="pt")

# print("SLOW TOKENIZER OUTPUTS:")
# for id, box in zip(encoding.input_ids.squeeze(), encoding.bbox.squeeze()):
#     print(tokenizer.decode(id), box)
