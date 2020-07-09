---
language: english
thumbnail:
---

# T5-base fine-tuned fo News Summarization 📖✏️🧾

All credits to [Abhishek Kumar Mishra](https://github.com/abhimishra91)

[Google's T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) base fine-tuned on [News Summary](https://www.kaggle.com/sunnysai12345/news-summary) dataset for **summarization** downstream task.

## Details of T5

The **T5** model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu* in Here the abstract:

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.

![model image](https://i.imgur.com/jVFMMWR.png)

## Details of the downstream task (Summarization) - Dataset 📚

[News Summary](https://www.kaggle.com/sunnysai12345/news-summary)

The dataset consists of **4515 examples** and contains Author_name, Headlines, Url of Article, Short text, Complete Article. I gathered the summarized news from Inshorts and only scraped the news articles from Hindu, Indian times and Guardian. Time period ranges from febrauary to august 2017.




## Model fine-tuning 🏋️‍

The training script is a slightly modified version of [this Colab Notebook](https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb) created by [Abhishek Kumar Mishra](https://github.com/abhimishra91), so all credits to him!
I also trained the model for more epochs (6).


## Model in Action 🚀

```python
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")

def summarize(text, max_length=150):
  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

  generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)

  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

  return preds[0]
```
Given the following article from **NYT** (2020/06/09) with title *George Floyd’s death energized a movement. He will be buried in Houston today*:

After the sound and the fury, weeks of demonstrations and anguished calls for racial justice, the man whose death gave rise to an international movement, and whose last words — “I can’t breathe” — have been a rallying cry, will be laid to rest on Tuesday at a private funeral in Houston.George Floyd, who was 46, will then be buried in a grave next to his mother’s.The service, scheduled to begin at 11 a.m. at the Fountain of Praise church, comes after five days of public memorials in Minneapolis, North Carolina and Houston and two weeks after a Minneapolis police officer was caught on video pressing his knee into Mr. Floyd’s neck for nearly nine minutes before Mr. Floyd died. That officer, Derek Chauvin, has been charged with second-degree murder and second-degree manslaughter. His bail was set at $1.25 million in a court appearance on Monday. The outpouring of anger and outrage after Mr. Floyd’s death — and the speed at which protests spread from tense, chaotic demonstrations in the city where he died to an international movement from Rome to Rio de Janeiro — has reflected the depth of frustration borne of years of watching black people die at the hands of the police or vigilantes while calls for change went unmet.
```
summarize('After the sound and the fury, weeks of demonstrations and anguished calls for racial justice, the man whose death gave rise to an international movement, and whose last words — “I can’t breathe” — have been a rallying cry, will be laid to rest on Tuesday at a private funeral in Houston.George Floyd, who was 46, will then be buried in a grave next to his mother’s.The service, scheduled to begin at 11 a.m. at the Fountain of Praise church, comes after five days of public memorials in Minneapolis, North Carolina and Houston and two weeks after a Minneapolis police officer was caught on video pressing his knee into Mr. Floyd’s neck for nearly nine minutes before Mr. Floyd died. That officer, Derek Chauvin, has been charged with second-degree murder and second-degree manslaughter. His bail was set at $1.25 million in a court appearance on Monday. The outpouring of anger and outrage after Mr. Floyd’s death — and the speed at which protests spread from tense, chaotic demonstrations in the city where he died to an international movement from Rome to Rio de Janeiro — has reflected the depth of frustration borne of years of watching black people die at the hands of the police or vigilantes while calls for change went unmet.', 80)
```
We would obtain:

At a private funeral in Houston. Floyd, who was 46 years old when his death occurred, will be buried next to the grave of his mother. A Minnesota police officer was caught on video pressing his knee into Mr's neck for nearly nine minutes before his death. The officer has been charged with second-degree manslaughter and $1.2 million bail is set at

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488) | [LinkedIn](https://www.linkedin.com/in/manuel-romero-cs/)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
