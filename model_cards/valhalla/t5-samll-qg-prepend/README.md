---
datasets:
- squad
tags:
- question-generation
widget:
- text: "answer: 42  context: 42 is the answer to life, the universe and everything. </s>"
- text: "answer: Guido Van Rossum context: Python is a programming language. It is developed by Guido Van Rossum. </s>"
- text: "answer: Explicit context: Explicit is better than implicit </s>"
license: "MIT"
---

## T5 for question-generation
This is [t5-small](https://arxiv.org/abs/1910.10683) model trained for answer aware question generation task. The answer text is prepended before the context text. 

You can play with the model using the inference API, just get the input text in this format and see the results!
`answer: answer_text context: context_text </s>`

For example

`answer: 42  context: 42 is the answer to life, the universe and everything. </s>`

For more deatils see [this](https://github.com/patil-suraj/question_generation) repo.

### Model in action 🚀

You'll need to clone the [repo](https://github.com/patil-suraj/question_generation).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patil-suraj/question_generation/blob/master/question_generation.ipynb)

```python3
from pipelines import pipeline
nlp = pipeline("question-generation", qg_format="prepend")
nlp("42 is the answer to life, universe and everything.")
=> [{'answer': '42', 'question': 'What is the answer to life, universe and everything?'}]
```