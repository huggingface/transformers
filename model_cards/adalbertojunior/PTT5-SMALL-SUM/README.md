---
language: pt
---

# PTT5-SMALL-SUM

## Model description

This model was trained to summarize texts in portuguese


based on ```unicamp-dl/ptt5-small-portuguese-vocab```

#### How to use

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('adalbertojunior/PTT5-SMALL-SUM')

t5 = T5ForConditionalGeneration.from_pretrained('adalbertojunior/PTT5-SMALL-SUM')

text="Esse é um exemplo de sumarização."

input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

generated_ids = t5.generate(
        input_ids=input_ids,
        num_beams=1,
        max_length=40,
        #repetition_penalty=2.5
    ).squeeze()
    
predicted_span = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


```
