---
language: ukrainian
---

Note: **default code snippet above won't work** because we are using `AlbertTokenizer` with `GPT2LMHeadModel`, see [issue](https://github.com/huggingface/transformers/issues/4285).

## GPT2 124M Trained on Ukranian Fiction

### Training details

Model was trained on corpus of 4040 fiction books, 2.77 GiB in total.
Evaluation on [brown-uk](https://github.com/brown-uk/corpus) gives perplexity of 50.16. 

### Example usage:
```python
from transformers import AlbertTokenizer, GPT2LMHeadModel

tokenizer = AlbertTokenizer.from_pretrained("Tereveni-AI/gpt2-124M-uk-fiction")
model = GPT2LMHeadModel.from_pretrained("Tereveni-AI/gpt2-124M-uk-fiction")

input_ids = tokenizer.encode("Но зла Юнона, суча дочка,", add_special_tokens=False, return_tensors='pt')

outputs = model.generate(
    input_ids,
    do_sample=True,
    num_return_sequences=3,
    max_length=50
)

for i, out in enumerate(outputs):
    print("{}: {}".format(i, tokenizer.decode(out)))
```

Prints something like this:
```bash
0: Но зла Юнона, суча дочка, яка затьмарила всі її таємниці: І хто з'їсть її душу, той помре». І, не дочекавшись гніву богів, посунула в пітьму, щоб не бачити перед собою. Але, за
1: Но зла Юнона, суча дочка, і довела мене до божевілля. Але він не знав нічого. Після того як я його побачив, мені стало зле. Я втратив рівновагу. Але в мене не було часу на роздуми. Я вже втратив надію
2: Но зла Юнона, суча дочка, не нарікала нам! — раптом вигукнула Юнона. — Це ти, старий йолопе! — мовила вона, не перестаючи сміятись. — Хіба ти не знаєш, що мені подобається ходити з тобою?
```