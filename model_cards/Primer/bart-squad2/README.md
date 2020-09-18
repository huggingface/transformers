---
language: "en"
---

# BART-Squad2

## Model description

BART for extractive (span-based) question answering, trained on Squad 2.0.

F1 score of 87.4.

## Intended uses & limitations

Unfortunately, the Huggingface auto-inference API won't run this model, so if you're attempting to try it through the input box above and it complains, don't be discouraged!

#### How to use

Here's a quick way to get question answering running locally:

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("Primer/bart-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("Primer/bart-squad2")
model.to('cuda'); model.eval()

def answer(question, text):
    seq = '<s>' +  question + ' </s> </s> ' + text + ' </s>'
    tokens = tokenizer.encode_plus(seq, return_tensors='pt', padding='max_length', max_length=1024)
    input_ids = tokens['input_ids'].to('cuda')
    attention_mask = tokens['attention_mask'].to('cuda')
    start, end, _ = model(input_ids, attention_mask=attention_mask)
    start_idx = int(start.argmax().int())
    end_idx =  int(end.argmax().int())
    print(tokenizer.decode(input_ids[0, start_idx:end_idx]).strip())
    # ^^ it will be an empty string if the model decided "unanswerable"

>>> question = "Where does Tom live?"
>>> context = "Tom is an engineer in San Francisco."
>>> answer(question, context)
San Francisco
```

(Just drop the `.to('cuda')` stuff if running on CPU).

#### Limitations and bias

Unknown, no further evaluation has been performed. In a technical sense one big limitation is that it's 1.6G 😬

## Training procedure

`run_squad.py` with:

|param|value|
|---|---|
|batch size|8|
|max_seq_length|1024|
|learning rate|1e-5|
|epochs|2|

Modified to freeze shared parameters and encoder embeddings.

