# LONGFORMER-BASE-4096 fine-tuned on SQuAD v1
This is longformer-base-4096 model fine-tuned on SQuAD v1 dataset for question answering task. 

[Longformer](https://arxiv.org/abs/2004.05150) model  created by Iz Beltagy, Matthew E. Peters, Arman Coha from AllenAI.  As the paper explains it 

> `Longformer` is a BERT-like model for long documents. 

The pre-trained model can handle sequences with upto 4096 tokens. 


## Model Training
This model was trained on google colab v100 GPU. You can find the fine-tuning colab here [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zEl5D-DdkBKva-DdreVOmN0hrAfzKG1o?usp=sharing).

Few things to keep in mind while training longformer for QA task,
by default longformer uses sliding-window local attention on all tokens. But For QA, all question tokens should  have global attention. For more details on this please refer the paper. The `LongformerForQuestionAnswering` model automatically does that for you. To allow it to do that 
1. The input sequence must have three sep tokens, i.e the sequence should be encoded like this
   ` <s> question</s></s> context</s>`.  If you encode the question and answer as a input  pair, then the tokenizer already takes care of that, you shouldn't worry about it.
2. `input_ids` should always be a batch of examples. 

## Results
|Metric       | # Value |
|-------------|---------|
| Exact Match | 85.1466 |
| F1          | 91.5415 |

## Model in Action  🚀
```python
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering,

tokenizer = AutoTokenizer.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
model = AutoModelForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

text = "Huggingface has democratized NLP. Huge thanks to Huggingface for this."
question = "What has Huggingface done ?"
encoding = tokenizer.encode_plus(question, text, return_tensors="pt")
input_ids = encoding["input_ids"]

# default is local attention everywhere
# the forward method will automatically set global attention on question tokens
attention_mask = encoding["attention_mask"]

start_scores, end_scores = model(input_ids, attention_mask=attention_mask)
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
# output => democratized NLP
```

The `LongformerForQuestionAnswering` isn't yet supported in `pipeline` . I'll update this card once the support has been added.

> Created with ❤️ by Suraj Patil [![Github icon](https://cdn0.iconfinder.com/data/icons/octicons/1024/mark-github-32.png)](https://github.com/patil-suraj/)
[![Twitter icon](https://cdn0.iconfinder.com/data/icons/shift-logotypes/32/Twitter-32.png)](https://twitter.com/psuraj28)
