---
datasets:
- squad
tags:
- question-generation
widget:
- text: "<hl> 42 <hl> is the answer to life, the universe and everything. </s>"
- text: "Python is a programming language. It is developed by <hl> Guido Van Rossum <hl>. </s>"
- text: "Although <hl> practicality <hl> beats purity </s>"
license: "MIT"
---

## T5 for question-generation
This is [t5-base](https://arxiv.org/abs/1910.10683) model trained for answer aware question generation task. The answer spans are highlighted within the text with special highlight tokens. 

You can play with the model using the inference API, just highlight the answer spans with `<hl>` tokens and end the text with `</s>`. For example

`<hl> 42 <hl> is the answer to life, the universe and everything. </s>`

For more deatils see [this](https://github.com/patil-suraj/question_generation) repo.

### Model in action 🚀

You'll need to clone the [repo](https://github.com/patil-suraj/question_generation).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patil-suraj/question_generation/blob/master/question_generation.ipynb)

```python3
from pipelines import pipeline
nlp = pipeline("question-generation", model="valhalla/t5-base-qg-hl")
nlp("42 is the answer to life, universe and everything.")
=> [{'answer': '42', 'question': 'What is the answer to life, universe and everything?'}]
```