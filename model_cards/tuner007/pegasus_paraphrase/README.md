# Pegasus for Paraphrasing
Pegasus model fine-tuned for paraphrasing

## Model in Action 🚀
```
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60).to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text
```
#### Example 1: 
```
context = "The ultimate test of your knowledge is your capacity to convey it to another."
get_response(context,10)
# output:
['The test of your knowledge is your ability to convey it.',
 'The ability to convey your knowledge is the ultimate test of your knowledge.',
 'The ability to convey your knowledge is the most important test of your knowledge.',
 'Your capacity to convey your knowledge is the ultimate test of it.',
 'The test of your knowledge is your ability to communicate it.',
 'Your capacity to convey your knowledge is the ultimate test of your knowledge.',
 'Your capacity to convey your knowledge to another is the ultimate test of your knowledge.',
 'Your capacity to convey your knowledge is the most important test of your knowledge.',
 'The test of your knowledge is how well you can convey it.',
 'Your capacity to convey your knowledge is the ultimate test.']
```
#### Example 2: Question paraphrasing (was not trained on quora dataset)
```
context = "Which course should I take to get started in data science?"
get_response(context,10)
# output: 
['Which data science course should I take?',
 'Which data science course should I take first?',
 'Should I take a data science course?',
 'Which data science class should I take?',
 'Which data science course should I attend?',
 'I want to get started in data science.',
 'Which data science course should I enroll in?',
 'Which data science course is right for me?',
 'Which data science course is best for me?',
 'Which course should I take to get started?']
```

> Created by Arpit Rajauria
[![Twitter icon](https://cdn0.iconfinder.com/data/icons/shift-logotypes/32/Twitter-32.png)](https://twitter.com/arpit_rajauria)
