---
language: ga
tags:
- irish
---

## BERTreach

([beirtreach](https://www.teanglann.ie/en/fgb/beirtreach) means 'oyster bed')

**Model size:** 84M

**Training data:** 
* [PARSEME 1.2](https://gitlab.com/parseme/parseme_corpus_ga/-/blob/master/README.md) 
* Newscrawl 300k portion of the [Leipzig Corpora](https://wortschatz.uni-leipzig.de/en/download/irish)
* Private news corpus crawled with [Corpus Crawler](https://github.com/google/corpuscrawler)

(2125804 sentences, 47419062 tokens, as reckoned by wc)

```
from transformers import pipeline
fill_mask = pipeline("fill-mask", model="jimregan/BERTreach", tokenizer="jimregan/BERTreach")
```
