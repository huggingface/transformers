---
tags:
- finance
---
# Roberta Masked Language Model Trained On Financial Phrasebank Corpus 


This is a Masked Language Model trained with [Roberta](https://huggingface.co/transformers/model_doc/roberta.html) on a Financial Phrasebank Corpus.
The model is built using Huggingface transformers.
The model can be found at :[Financial_Roberta](https://huggingface.co/abhilash1910/financial_roberta)


## Specifications


The corpus for training is taken from the Financial Phrasebank (Malo et al)[https://www.researchgate.net/publication/251231107_Good_Debt_or_Bad_Debt_Detecting_Semantic_Orientations_in_Economic_Texts]. 


## Model Specification


The model chosen for training is [Roberta](https://arxiv.org/abs/1907.11692) with the following specifications:
 1. vocab_size=56000
 2. max_position_embeddings=514
 3. num_attention_heads=12
 4. num_hidden_layers=6
 5. type_vocab_size=1


This is trained by using  RobertaConfig from transformers package.
The model is trained for 10 epochs with a gpu batch size of 64 units. 



## Usage Specifications


For using this model, we have to first import AutoTokenizer and AutoModelWithLMHead Modules from transformers
After that we have to specify, the pre-trained model,which in this case is 'abhilash1910/financial_roberta' for the tokenizers and the model.


```python
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("abhilash1910/financial_roberta")

model = AutoModelWithLMHead.from_pretrained("abhilash1910/financial_roberta")
```


After this the model will be downloaded, it will take some time to download all the model files.
For testing the model, we have to import  pipeline module from transformers and create a masked output model for inference as follows:


```python
from transformers import pipeline
model_mask = pipeline('fill-mask', model='abhilash1910/inancial_roberta')
model_mask("The  company had a <mask> of 20% in 2020.")
```


Some of the examples are also provided with generic financial statements:

Example 1:


```python
model_mask("The  company had a <mask> of 20% in 2020.")
```


Output:


```bash
[{'sequence': '<s>The  company had a profit of 20% in 2020.</s>',
  'score': 0.023112965747714043,
  'token': 421,
  'token_str': 'Ġprofit'},
 {'sequence': '<s>The  company had a loss of 20% in 2020.</s>',
  'score': 0.021379893645644188,
  'token': 616,
  'token_str': 'Ġloss'},
 {'sequence': '<s>The  company had a year of 20% in 2020.</s>',
  'score': 0.0185744296759367,
  'token': 443,
  'token_str': 'Ġyear'},
 {'sequence': '<s>The  company had a sales of 20% in 2020.</s>',
  'score': 0.018143286928534508,
  'token': 428,
  'token_str': 'Ġsales'},
 {'sequence': '<s>The  company had a value of 20% in 2020.</s>',
  'score': 0.015319528989493847,
  'token': 776,
  'token_str': 'Ġvalue'}]
  ```
 
 Example 2:
 
```python
 model_mask("The <mask>  is listed under NYSE")
```

Output:

```bash
[{'sequence': '<s>The company  is listed under NYSE</s>',
  'score': 0.1566661298274994,
  'token': 359,
  'token_str': 'Ġcompany'},
 {'sequence': '<s>The total  is listed under NYSE</s>',
  'score': 0.05542507395148277,
  'token': 522,
  'token_str': 'Ġtotal'},
 {'sequence': '<s>The value  is listed under NYSE</s>',
  'score': 0.04729423299431801,
  'token': 776,
  'token_str': 'Ġvalue'},
 {'sequence': '<s>The order  is listed under NYSE</s>',
  'score': 0.02533523552119732,
  'token': 798,
  'token_str': 'Ġorder'},
 {'sequence': '<s>The contract  is listed under NYSE</s>',
  'score': 0.02087237872183323,
  'token': 635,
  'token_str': 'Ġcontract'}]
  ```
  

## Resources

For all resources , please look into the [HuggingFace](https://huggingface.co/) Site and the [Repositories](https://github.com/huggingface).
