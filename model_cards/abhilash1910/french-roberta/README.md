# Roberta Trained Model For Masked Language Model On French Corpus :robot:


This is a Masked Language Model trained with [Roberta](https://huggingface.co/transformers/model_doc/roberta.html) on a small French News Corpus(Leipzig corpora).
The model is built using Huggingface transformers.
The model can be found at :[French-Roberta](https://huggingface.co/abhilash1910/french-roberta)


## Specifications


The corpus for training is taken from Leipzig Corpora (French News) , and is trained on a small set of the corpus (300K). 


## Model Specification


The model chosen for training is [Roberta](https://arxiv.org/abs/1907.11692) with the following specifications:
 1. vocab_size=32000
 2. max_position_embeddings=514
 3. num_attention_heads=12
 4. num_hidden_layers=6
 5. type_vocab_size=1


This is trained by using  RobertaConfig from transformers package.The total training parameters :68124416
The model is trained for 100 epochs with a gpu batch size of 64 units. 
More details for building custom models can be found at the [HuggingFace Blog](https://huggingface.co/blog/how-to-train)



## Usage Specifications


For using this model, we have to first import AutoTokenizer and AutoModelWithLMHead Modules from transformers
After that we have to specify, the pre-trained model,which in this case is 'abhilash1910/french-roberta' for the tokenizers and the model.


```python
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("abhilash1910/french-roberta")

model = AutoModelWithLMHead.from_pretrained("abhilash1910/french-roberta")
```


After this the model will be downloaded, it will take some time to download all the model files.
For testing the model, we have to import  pipeline module from transformers and create a masked output model for inference as follows:


```python
from transformers import pipeline
model_mask = pipeline('fill-mask', model='abhilash1910/french-roberta')
model_mask("Le tweet <mask>.")
```


Some of the examples are also provided with generic French sentences:

Example 1:


```python
model_mask("À ce jour, <mask> projet a entraîné")
```


Output:


```bash
[{'sequence': '<s>À ce jour, belles projet a entraîné</s>',
  'score': 0.18685665726661682,
  'token': 6504,
  'token_str': 'Ġbelles'},
 {'sequence': '<s>À ce jour,- projet a entraîné</s>',
  'score': 0.0005200508167035878,
  'token': 17,
  'token_str': '-'},
 {'sequence': '<s>À ce jour, de projet a entraîné</s>',
  'score': 0.00045729897101409733,
  'token': 268,
  'token_str': 'Ġde'},
 {'sequence': '<s>À ce jour, du projet a entraîné</s>',
  'score': 0.0004307595663703978,
  'token': 326,
  'token_str': 'Ġdu'},
 {'sequence': '<s>À ce jour," projet a entraîné</s>',
  'score': 0.0004219160182401538,
  'token': 6,
  'token_str': '"'}]
  ```
 
 Example 2:
 
```python
 model_mask("C'est un <mask>")
```

Output:

```bash
[{'sequence': "<s>C'est un belles</s>",
  'score': 0.16440927982330322,
  'token': 6504,
  'token_str': 'Ġbelles'},
 {'sequence': "<s>C'est un de</s>",
  'score': 0.0005495127406902611,
  'token': 268,
  'token_str': 'Ġde'},
 {'sequence': "<s>C'est un du</s>",
  'score': 0.00044988933950662613,
  'token': 326,
  'token_str': 'Ġdu'},
 {'sequence': "<s>C'est un-</s>",
  'score': 0.00044542422983795404,
  'token': 17,
  'token_str': '-'},
 {'sequence': "<s>C'est un\t</s>",
  'score': 0.00037563967634923756,
  'token': 202,
  'token_str': 'ĉ'}]
  ```
  

## Resources

For all resources , please look into the [HuggingFace](https://huggingface.co/) Site and the [Repositories](https://github.com/huggingface).


