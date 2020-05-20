# Bert-base Turkish Sentiment Model

https://huggingface.co/savasy/bert-base-turkish-sentiment-cased

This model is used for Sentiment Analysis, which is based on BERTurk for Turkish Language https://huggingface.co/dbmdz/bert-base-turkish-cased


## Dataset

The dataset is taken from the studies [[2]](#paper-2) and [[3]](#paper-3), and merged.

* The study [2] gathered movie and product reviews. The products are book, DVD, electronics, and kitchen.
The movie dataset is taken from a cinema Web page ([Beyazperde](www.beyazperde.com)) with
5331 positive and 5331 negative sentences. Reviews in the Web page are marked in
scale from 0 to 5 by the users who made the reviews. The study considered a review
sentiment positive if the rating is equal to or bigger than 4, and negative if it is less
or equal to 2. They also built Turkish product review dataset from an online retailer
Web page. They constructed benchmark dataset consisting of reviews regarding some
products (book, DVD, etc.). Likewise, reviews are marked in the range from 1 to 5,
and majority class of reviews are 5. Each category has 700 positive and 700 negative
reviews in which average rating of negative reviews is 2.27 and of positive reviews
is 4.5. This dataset is also used by the study [[1]](#paper-1).

* The study [[3]](#paper-3) collected tweet dataset. They proposed a new approach for automatically classifying the sentiment of microblog messages. The proposed approach is based on utilizing robust feature representation and fusion. 

*Merged Dataset* 

| *size*   | *data* |
|--------|----|
|   8000 |dev.tsv|
|   8262 |test.tsv|
|  32000 |train.tsv|
|  *48290* |*total*|

### The dataset is used by following papers

<a id="paper-1">[1]</a> Yildirim, Savaş. (2020). Comparing Deep Neural Networks to Traditional Models for Sentiment Analysis in Turkish Language. 10.1007/978-981-15-1216-2_12. 

<a id="paper-2">[2]</a> Demirtas, Erkin and Mykola Pechenizkiy. 2013. Cross-lingual polarity detection with machine translation. In Proceedings of the Second International Workshop on Issues of Sentiment
Discovery and Opinion Mining (WISDOM ’13)

<a id="paper-3">[3]</a> Hayran, A.,   Sert, M. (2017), "Sentiment Analysis on Microblog Data based on Word Embedding and Fusion Techniques", IEEE 25th Signal Processing and Communications Applications Conference (SIU 2017), Belek, Turkey


## Training

```shell
export GLUE_DIR="./sst-2-newall"
export TASK_NAME=SST-2

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path dbmdz/bert-base-turkish-uncased\
  --task_name "SST-2" \
  --do_train \
  --do_eval \
  --data_dir "./sst-2-newall" \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir "./model"
```


## Results

> 05/10/2020 17:00:43 - INFO - transformers.trainer -   \*\*\*\*\* Running Evaluation \*\*\*\*\*  
> 05/10/2020 17:00:43 - INFO - transformers.trainer -     Num examples = 7999  
> 05/10/2020 17:00:43 - INFO - transformers.trainer -     Batch size = 8  
> Evaluation: 100% 1000/1000 [00:34<00:00, 29.04it/s]  
> 05/10/2020 17:01:17 - INFO - \_\_main__ -   \*\*\*\*\* Eval results sst-2 \*\*\*\*\*  
> 05/10/2020 17:01:17 - INFO - \_\_main__ -     acc = 0.9539942492811602  
> 05/10/2020 17:01:17 - INFO - \_\_main__ -     loss = 0.16348013816401363

Accuracy is about **95.4%**


## Code Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
sa= pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)

p = sa("bu telefon modelleri çok kaliteli , her parçası çok özel bence")
print(p)
# [{'label': 'LABEL_1', 'score': 0.9871089}]
print(p[0]['label'] == 'LABEL_1')
# True

p = sa("Film çok kötü ve çok sahteydi")
print(p)
# [{'label': 'LABEL_0', 'score': 0.9975505}]
print(p[0]['label'] == 'LABEL_1')
# False
```


## Test
### Data

Suppose your file has lots of lines of comment and label (1 or 0) at the end  (tab seperated)

> comment1 ... \t label  
> comment2 ... \t label  
> ...

### Code

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
sa = pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)

input_file = "/path/to/your/file/yourfile.tsv"

i, crr = 0, 0
for line in open(input_file):
    lines = line.strip().split("\t")
    if len(lines) == 2:
        
        i = i + 1
        if i%100 == 0:
            print(i)
        
        pred = sa(lines[0])
        pred = pred[0]["label"].split("_")[1]
        
        if pred == lines[1]:
        crr = crr + 1

print(crr, i, crr/i)
```
