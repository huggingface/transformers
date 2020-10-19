# Pegasus for question-answering
Pegasus model fine-tuned for QA using text-to-text approach

## Model in Action 🚀
```
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_qa'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_answer(question, context):
  input_text = "question: %s text: %s" % (question,context)
  batch = tokenizer.prepare_seq2seq_batch([input_text], truncation=True, padding='longest').to(torch_device)
  translated = model.generate(**batch)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text[0]
```
#### Example:
```
context = "PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
question = "How many customers were affected by the shutoffs?"
get_answer(question, context)
# output: '800 thousand'
```


> Created by Arpit Rajauria
[![Twitter icon](https://cdn0.iconfinder.com/data/icons/shift-logotypes/32/Twitter-32.png)](https://twitter.com/arpit_rajauria)
