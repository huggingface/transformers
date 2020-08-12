## Reformer Language model on character level and trained on enwik8. 

*enwik8* is a dataset based on Wikipedia and is often used to measure the model's ability to *compress* data, *e.g.* in 
the scope of the *Hutter prize*: https://en.wikipedia.org/wiki/Hutter_Prize.

`reformer-enwik8` was pretrained on the first 90M chars of *enwik8* whereas the text was chunked into batches of size 65536 chars (=2^16).
The model's weights were taken from https://console.cloud.google.com/storage/browser/trax-ml/reformer/enwik8 and converted 
to Hugging Face's PyTorch ReformerLM model `ReformerModelWithLMHead`.

The model is a language model that operates on characters. 
Therefore, this model does not need a tokenizer. The following function can instead be used for **encoding** and **decoding**:

```python
import torch

# Encoding
def encode(list_of_strings, pad_token_id=0):
    max_length = max([len(string) for string in list_of_strings])

    # create emtpy tensors
    attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
    input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

    for idx, string in enumerate(list_of_strings):
        # make sure string is in byte format
        if not isinstance(string, bytes):
            string = str.encode(string)

        input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
        attention_masks[idx, :len(string)] = 1

    return input_ids, attention_masks
    
# Decoding
def decode(outputs_ids):
    decoded_outputs = []
    for output_ids in outputs_ids.tolist():
        # transform id back to char IDs < 2 are simply transformed to ""
        decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
    return decoded_outputs
```

Text can be generated as follows:

```python
from transformers import ReformerModelWithLMHead

model = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8")
encoded, attention_masks = encode(["In 1965, Brooks left IBM to found the Department of"])
decode(model.generate(encoded, do_sample=True, max_length=150))

# gives:
# In 1965, Brooks left IBM to found the Department of Journalism in 1968. IBM had jurisdiction himself in 1980, while Brooks resolved, nevertheless thro

```

***Note***: Language generation using `ReformerModelWithLMHead` is not optimized yet and is rather slow.
