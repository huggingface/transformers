# Details

https://huggingface.co/savasy/bert-base-turkish-sentiment-cased

This model is used for Sentiment Analysis, which is based on BERTurk for Turkish Language https://huggingface.co/dbmdz/bert-base-turkish-cased


# Dataset

We used product and movie dataset provided by the study [2] . This dataset includes
movie and product reviews. The products are book, DVD, electronics, and kitchen.
The movie dataset is taken from a cinema Web page (www.beyazperde.com) with
5331 positive and 5331 negative sentences. Reviews in the Web page are marked in
scale from 0 to 5 by the users who made the reviews. The study considered a review
sentiment positive if the rating is equal to or bigger than 4, and negative if it is less
or equal to 2. They also built Turkish product review dataset from an online retailer
Web page. They constructed benchmark dataset consisting of reviews regarding some
products (book, DVD, etc.). Likewise, reviews are marked in the range from 1 to 5,
and majority class of reviews are 5. Each category has 700 positive and 700 negative
reviews in which average rating of negative reviews is 2.27 and of positive reviews
is 4.5.


The dataset is used by following papers

 
* 1 Yildirim, Savaş. (2020). Comparing Deep Neural Networks to Traditional Models for Sentiment Analysis in Turkish Language. 10.1007/978-981-15-1216-2_12. 
* 2 Demirtas, Erkin and Mykola Pechenizkiy. 2013. Cross-lingual polarity detection with machine translation. In Proceedings of the Second International Workshop on Issues of Sentiment
Discovery and Opinion Mining (WISDOM ’13)

# Code Usage

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
sa= pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)

p= sa("bu telefon modelleri çok kaliteli , her parçası çok özel bence")
print(p)
#[{'label': 'LABEL_1', 'score': 0.9871089}]
print (p[0]['label']=='LABEL_1')
#True


p= sa("Film çok kötü ve çok sahteydi")
print(p)
#[{'label': 'LABEL_0', 'score': 0.9975505}]
print (p[0]['label']=='LABEL_1')
#False
```
