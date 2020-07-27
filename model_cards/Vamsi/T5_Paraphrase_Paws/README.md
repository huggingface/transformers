---
language: "en"
tags:
- paraphrase-generation
- text-generation
- Conditional Generation
inference: false
---
​
# Paraphrase-Generation
​
## Model description
​
T5 Model for generating paraphrases of english sentences. Trained on the [Google PAWS](https://github.com/google-research-datasets/paws) dataset.
​
## How to use
​
PyTorch and TF models available
​
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
​
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
​
sentence = "This is something which i cannot understand at all"

text =  "paraphrase: " + sentence + " </s>"

encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")


outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    do_sample=True,
    top_k=120,
    top_p=0.95,
    early_stopping=True,
    num_return_sequences=5
)

for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print(line)
​

```

For more reference on training your own T5 model or using this model, do check out [Paraphrase Generation](https://github.com/Vamsi995/Paraphrase-Generator).
