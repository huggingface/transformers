---
datasets:
- squad
tags:
- question-generation
widget:
- text: "Python is developed by Guido Van Rossum and released in 1991. </s>"
license: "MIT"
---

## T5 for question-generation
This is [t5-small](https://arxiv.org/abs/1910.10683) model trained for end-to-end question generation task. Simply input the text and the model will generate multile questions. 

You can play with the model using the inference API, just put the text and see the results!

For more deatils see [this](https://github.com/patil-suraj/question_generation) repo.

### Model in action 🚀

You'll need to clone the [repo](https://github.com/patil-suraj/question_generation).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patil-suraj/question_generation/blob/master/question_generation.ipynb)

```python3
from pipelines import pipeline

text = "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum \
and first released in 1991, Python's design philosophy emphasizes code \
readability with its notable use of significant whitespace."

nlp = pipeline("e2e-qg")
nlp(text)
=> [
 'Who created Python?',
 'When was Python first released?',
 "What is Python's design philosophy?"
]
```