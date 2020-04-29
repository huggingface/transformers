# DistilBERT Yelp Review Sentiment
This model is used for sentiment analysis on english yelp reviews.  
It is a DistilBERT model trained on 1 million reviews from the yelp open dataset.  
It is a regression model, with outputs in the range of ~-2 to ~2. With -2 being 1 star and 2 being 5 stars.  
It was trained using the [ktrain](https://github.com/amaiya/ktrain) because of it's ease of use.

Example use:

```
tokenizer = AutoTokenizer.from_pretrained(
    'distilbert-base-uncased', use_fast=True)
model = TFAutoModelForSequenceClassification.from_pretrained(
    "spentaur/yelp")
    
review = "This place is great!"
input_ids = tokenizer.encode(review, return_tensors='tf')
pred = model(input_ids)[0][0][0].numpy()
# pred should === 1.9562385
```
