# German Sentiment Classification with Bert

This model was trained for sentiment classification of German language texts. To achieve the best results all model inputs needs to be preprocessed with the same procedure, that was applied during the training. To simplify the usage of the model, 
we provide a Python package that bundles the code need for the preprocessing and inferencing. 

The model uses the Googles Bert architecture and was trained on 1.834 million German-language samples. The training data contains texts from various domains like Twitter, Facebook and movie, app and hotel reviews. 
You can find more information about the dataset and the training process in the [paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.201.pdf).

## Using the Python package

To get started install the package from [pypi](https://pypi.org/project/germansentiment/):

```bash
pip install germansentiment
```

```python
from germansentiment import SentimentModel

model = SentimentModel()

texts = [
    "Mit keinem guten Ergebniss","Das ist gar nicht mal so gut",
    "Total awesome!","nicht so schlecht wie erwartet",
    "Der Test verlief positiv.","Sie fährt ein grünes Auto."]
       
result = model.predict_sentiment(texts)
print(result)
```

The code above will output following list:

```python
["negative","negative","positive","positive","neutral", "neutral"]
```

## A minimal working Sample


```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import torch
import re

class SentimentModel():
    def __init__(self, model_name: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)
        self.clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
        self.clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)

    def predict_sentiment(self, texts: List[str])-> List[str]:
        texts = [self.clean_text(text) for text in texts]
        # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        input_ids = self.tokenizer(texts, padding=True, truncation=True, add_special_tokens=True)
        input_ids = torch.tensor(input_ids["input_ids"])

        with torch.no_grad():
            logits = self.model(input_ids)    

        label_ids = torch.argmax(logits[0], axis=1)

        labels = [self.model.config.id2label[label_id] for label_id in label_ids.tolist()]
        return labels

    def replace_numbers(self,text: str) -> str:
            return text.replace("0"," null").replace("1"," eins").replace("2"," zwei").replace("3"," drei").replace("4"," vier").replace("5"," fünf").replace("6"," sechs").replace("7"," sieben").replace("8"," acht").replace("9"," neun")         

    def clean_text(self,text: str)-> str:    
            text = text.replace("\n", " ")        
            text = self.clean_http_urls.sub('',text)
            text = self.clean_at_mentions.sub('',text)        
            text = self.replace_numbers(text)                
            text = self.clean_chars.sub('', text) # use only text chars                          
            text = ' '.join(text.split()) # substitute multiple whitespace with single whitespace   
            text = text.strip().lower()
            return text

texts = ["Mit keinem guten Ergebniss","Das war unfair", "Das ist gar nicht mal so gut",
        "Total awesome!","nicht so schlecht wie erwartet", "Das ist gar nicht mal so schlecht",
        "Der Test verlief positiv.","Sie fährt ein grünes Auto.", "Der Fall wurde an die Polzei übergeben."]

model = SentimentModel(model_name = "oliverguhr/german-sentiment-bert")

print(model.predict_sentiment(texts))
```

## Model and Data

If you are interested in code and data that was used to train this model please have a look at [this repository](https://github.com/oliverguhr/german-sentiment) and our [paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.202.pdf). Here is a table of the F1 scores that his model achieves on following datasets. Since we trained this model on a newer version of the transformer library, the results are slightly better than reported in the paper.

| Dataset                                                      | F1 micro Score |
| :----------------------------------------------------------- | -------------: |
| [holidaycheck](https://github.com/oliverguhr/german-sentiment) |         0.9568 |
| [scare](https://www.romanklinger.de/scare/)                  |         0.9418 |
| [filmstarts](https://github.com/oliverguhr/german-sentiment) |         0.9021 |
| [germeval](https://sites.google.com/view/germeval2017-absa/home) |         0.7536 |
| [PotTS](https://www.aclweb.org/anthology/L16-1181/)          |         0.6780 |
| [emotions](https://github.com/oliverguhr/german-sentiment)  |         0.9649 |
| [sb10k](https://www.spinningbytes.com/resources/germansentiment/) |         0.7376 |
| [Leipzig Wikipedia Corpus 2016](https://wortschatz.uni-leipzig.de/de/download/german) |         0.9967 |
| all                                                          |         0.9639 |

## Cite

For feedback and questions contact me view mail or Twitter [@oliverguhr](https://twitter.com/oliverguhr). Please cite us if you found this useful:

```
@InProceedings{guhr-EtAl:2020:LREC,
  author    = {Guhr, Oliver  and  Schumann, Anne-Kathrin  and  Bahrmann, Frank  and  Böhme, Hans Joachim},
  title     = {Training a Broad-Coverage German Sentiment Classification Model for Dialog Systems},
  booktitle      = {Proceedings of The 12th Language Resources and Evaluation Conference},
  month          = {May},
  year           = {2020},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {1620--1625},
  url       = {https://www.aclweb.org/anthology/2020.lrec-1.201}
}
```


