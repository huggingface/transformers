# T5 for question-answering
This is T5-base model fine-tuned on SQuAD1.1 for QA using text-to-text approach

## Model training
This model was trained on colab TPU with 35GB RAM for 4 epochs

## Results:
| Metric      | #Value  |
|-------------|---------|
| Exact Match | 81.5610 |
| F1          | 89.9601 |

## Model in Action 🚀
```
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-squad")
model = AutoModelWithLMHead.from_pretrained("valhalla/t5-base-squad")

def get_answer(question, context):
  input_text = "question: %s  context: %s </s>" % (question, context)
  features = tokenizer([input_text], return_tensors='pt')

  out = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'])
  
  return tokenizer.decode(out[0])

context = "In Norse mythology, Valhalla is a majestic, enormous hall located in Asgard, ruled over by the god Odin."
question = "What is Valhalla ?"

get_answer(question, context)
# output: 'a majestic, enormous hall located in Asgard, ruled over by the god Odin'
```
Play with this model [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a5xpJiUjZybfU9Mi-aDkOp116PZ9-wni?usp=sharing)

> Created by Suraj Patil [![Github icon](https://cdn0.iconfinder.com/data/icons/octicons/1024/mark-github-32.png)](https://github.com/patil-suraj/)
[![Twitter icon](https://cdn0.iconfinder.com/data/icons/shift-logotypes/32/Twitter-32.png)](https://twitter.com/psuraj28)
