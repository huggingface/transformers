# Telugu Question-Answering model trained on Tydiqa dataset from Google

#### How to use

```python
from transformers.pipelines import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("kuppuluri/telugu_bertu_tydiqa",
                                          clean_text=False,
                                          handle_chinese_chars=False,
                                          strip_accents=False,
                                          wordpieces_prefix='##')
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
result = nlp({'question': question, 'context': context})
```

## Training data
I used Tydiqa Telugu data from Google https://github.com/google-research-datasets/tydiqa
