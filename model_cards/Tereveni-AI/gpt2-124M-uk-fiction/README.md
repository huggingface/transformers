Note: **default code snippet above won't work** because we are using `AlbertTokenizer` with `GPT2LMHeadModel`, see [issue](https://github.com/huggingface/transformers/issues/4285).

## GPT2 124M Trained on Ukranian Fiction

Example usage:
```python
from transformers import AlbertTokenizer, GPT2LMHeadModel

tokenizer = AlbertTokenizer.from_pretrained("Tereveni-AI/gpt2-124M-uk-fiction")
model = GPT2LMHeadModel.from_pretrained("Tereveni-AI/gpt2-124M-uk-fiction")

input_ids = tokenizer.encode('Но зла Юнона, суча дочка,', add_special_tokens=False, return_tensors='pt')

outputs = model.generate(
    input_ids,
    do_sample=True,
    num_return_sequences=3,
    max_length=50
)

for i, out in enumerate(outputs):
    print('{}: {}'.format(i, tokenizer.decode(out)))
```
