# T5 for abstractive question-answering
This is T5-base model fine-tuned for abstractive QA using text-to-text approach

## Model training
This model was trained on colab TPU with 35GB RAM for 2 epochs

## Model in Action 🚀
```
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tuner007/t5_abs_qa")
model = AutoModelWithLMHead.from_pretrained("tuner007/t5_abs_qa")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_answer(question, context):
input_text = "context: %s <question for context: %s </s>" % (context,question)
features = tokenizer([input_text], return_tensors='pt')
out = model.generate(input_ids=features['input_ids'].to(device), attention_mask=features['attention_mask'].to(device))
return tokenizer.decode(out[0])
```
#### Example 1: Answer available
```
context = "In Norse mythology, Valhalla is a majestic, enormous hall located in Asgard, ruled over by the god Odin."
question = "What is Valhalla?"
get_answer(question, context)
# output: 'It is a hall of worship ruled by Odin.'
```
#### Example 2: Answer not available 
```
context = "In Norse mythology, Valhalla is a majestic, enormous hall located in Asgard, ruled over by the god Odin."
question = "What is Asgard?"
get_answer(question, context)
# output: 'No answer available in context.'
```


> Created by Arpit Rajauria
[![Twitter icon](https://cdn0.iconfinder.com/data/icons/shift-logotypes/32/Twitter-32.png)](https://twitter.com/arpit_rajauria)
