from transformers import UdopTokenizer

tokenizer = UdopTokenizer.from_pretrained("t5-base")

text = ["hello", "niels"]
boxes = [[1, 8, 7, 9], [1,2,3,4]]

encoding = tokenizer(text, boxes=boxes, return_tensors="pt")

for k,v in encoding.items():
    print(k,v.shape)

for id, box in zip(encoding.input_ids.squeeze(), encoding.bbox.squeeze()):
    print(tokenizer.decode(id), box)

