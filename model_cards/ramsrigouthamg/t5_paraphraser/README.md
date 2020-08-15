## Model in Action 🚀

```python
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)

sentence = "Which course should I take to get started in data science?"
# sentence = "What are the ingredients required to bake a perfect cake?"
# sentence = "What is the best possible approach to learn aeronautical engineering?"
# sentence = "Do apples taste better than oranges in general?"


text =  "paraphrase: " + sentence + " </s>"


max_len = 256

encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
beam_outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True,
    max_length=256,
    top_k=120,
    top_p=0.98,
    early_stopping=True,
    num_return_sequences=10
)


print ("\nOriginal Question ::")
print (sentence)
print ("\n")
print ("Paraphrased Questions :: ")
final_outputs =[]
for beam_output in beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    if sent.lower() != sentence.lower() and sent not in final_outputs:
        final_outputs.append(sent)

for i, final_output in enumerate(final_outputs):
    print("{}: {}".format(i, final_output))

```
## Output
```
Original Question ::
Which course should I take to get started in data science?


Paraphrased Questions :: 
0: What should I learn to become a data scientist?
1: How do I get started with data science?
2: How would you start a data science career?
3: How can I start learning data science?
4: How do you get started in data science?
5: What's the best course for data science?
6: Which course should I start with for data science?
7: What courses should I follow to get started in data science?
8: What degree should be taken by a data scientist?
9: Which course should I follow to become a Data Scientist?
```

## Detailed blog post available here :
https://towardsdatascience.com/paraphrase-any-question-with-t5-text-to-text-transfer-transformer-pretrained-model-and-cbb9e35f1555

