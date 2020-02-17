---
language:
- english
- dutch
- german
- french
- italian
- spanish
---

# bert-base-multilingual-uncased-sentiment

This a bert-base-multilingual-uncased model finetuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish and Italian. It predicts the sentiment of the review as a number of stars (between 1 and 5).

This model is intended for direct use as a sentiment analysis model for product reviews in any of the six languages above, or for further finetuning on related sentiment analysis tasks.

## Training data

Here is the number of product reviews we used for finetuning the model: 

| Language | Number of reviews |
| -------- | ----------------- |
| English  | 150k           |
| Dutch    | 80k            |
| German   | 137k           |
| French   | 140k           |
| Italian  | 72k            |
| Spanish  | 50k            |

## Accuracy

The finetuned model obtained the following accuracy on 5,000 held-out product reviews in each of the languages:

- Accuracy (exact) is the exact match on the number of stars.
- Accuracy (off-by-1) is the percentage of reviews where the number of stars the model predicts differs by a maximum of 1 from the number given by the human reviewer. 


| Language | Accuracy (exact) | Accuracy (off-by-1) |
| -------- | ---------------------- | ------------------- |
| English  | 67%                 | 95%
| Dutch    | 57%                 | 93%
| German   | 61%                 | 94%
| French   | 59%                 | 94%
| Italian  | 59%                 | 95%
| Spanish  | 58%                 | 95%

## Contact 

Contact [NLP Town](https://www.nlp.town) for questions, feedback and/or requests for similar models.
