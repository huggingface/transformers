from transformers import UdopTokenizer, UdopTokenizerFast


tokenizer = UdopTokenizer.from_pretrained("t5-base")
fast_tokenizer = UdopTokenizerFast.from_pretrained("t5-base")

text = ["hello", "niels"]
boxes = [[1, 8, 7, 9], [1, 2, 3, 4]]

encoding = tokenizer(text, boxes=boxes, return_tensors="pt")
fast_encoding = fast_tokenizer(text, boxes=boxes, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)

print("SLOW TOKENIZER OUTPUTS:")

for id, box in zip(encoding.input_ids.squeeze(), encoding.bbox.squeeze()):
    print(tokenizer.decode(id), box)

print("FAST TOKENIZER OUTPUTS:")

for id, box in zip(fast_encoding.input_ids.squeeze(), fast_encoding.bbox.squeeze()):
    print(tokenizer.decode(id), box)
