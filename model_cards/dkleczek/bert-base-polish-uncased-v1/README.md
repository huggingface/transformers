---
language: polish
thumbnail: https://raw.githubusercontent.com/kldarek/polbert/master/img/polbert.png
---

# Polbert - Polish BERT
Polish version of BERT language model is here! It is now available in two variants: cased and uncased, both can be downloaded and used via HuggingFace transformers library. I recommend using the cased model, more info on the differences and benchmark results below. 

![PolBERT image](https://raw.githubusercontent.com/kldarek/polbert/master/img/polbert.png)

## Cased and uncased variants

* I initially trained the uncased model, the corpus and training details are referenced below. Here are some issues I found after I published the uncased model:
    * Some Polish characters and accents are not tokenized correctly through the BERT tokenizer when applying lowercase. This doesn't impact sequence classification much, but may influence token classfication tasks significantly.
    * I noticed a lot of duplicates in the Open Subtitles dataset, which dominates the training corpus.
    * I didn't use Whole Word Masking. 
* The cased model improves on the uncased model in the following ways:
    * All Polish characters and accents should now be tokenized correctly. 
    * I removed duplicates from Open Subtitles dataset. The corpus is smaller, but more balanced now. 
    * The model is trained with Whole Word Masking. 

## Pre-training corpora

Below is the list of corpora used along with the output of `wc` command (counting lines, words and characters). These corpora were divided into sentences with srxsegmenter (see references), concatenated and tokenized with HuggingFace BERT Tokenizer. 

### Uncased

| Tables        | Lines           | Words  | Characters  |
| ------------- |--------------:| -----:| -----:|
| [Polish subset of Open Subtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php)      | 236635408| 1431199601 | 7628097730 |
| [Polish subset of ParaCrawl](http://opus.nlpl.eu/ParaCrawl.php)     | 8470950      |   176670885 | 1163505275 |
| [Polish Parliamentary Corpus](http://clip.ipipan.waw.pl/PPC) | 9799859      |    121154785 | 938896963 |
| [Polish Wikipedia - Feb 2020](https://dumps.wikimedia.org/plwiki/latest/plwiki-latest-pages-articles.xml.bz2) | 8014206      |    132067986 | 1015849191 |
| Total | 262920423      |    1861093257 | 10746349159 |

### Cased

| Tables        | Lines           | Words  | Characters  |
| ------------- |--------------:| -----:| -----:|
| [Polish subset of Open Subtitles (Deduplicated) ](http://opus.nlpl.eu/OpenSubtitles-v2018.php)      | 41998942| 213590656 | 1424873235 |
| [Polish subset of ParaCrawl](http://opus.nlpl.eu/ParaCrawl.php)     | 8470950      |   176670885 | 1163505275 |
| [Polish Parliamentary Corpus](http://clip.ipipan.waw.pl/PPC) | 9799859      |    121154785 | 938896963 |
| [Polish Wikipedia - Feb 2020](https://dumps.wikimedia.org/plwiki/latest/plwiki-latest-pages-articles.xml.bz2) | 8014206      |    132067986 | 1015849191 |
| Total | 68283960      |    646479197 | 4543124667 |


## Pre-training details

### Uncased 

* Polbert was trained with code provided in Google BERT's github repository (https://github.com/google-research/bert)
* Currently released model follows bert-base-uncased model architecture (12-layer, 768-hidden, 12-heads, 110M parameters)
* Training set-up: in total 1 million training steps: 
    * 100.000 steps - 128 sequence length, batch size 512, learning rate 1e-4 (10.000 steps warmup)
    * 800.000 steps - 128 sequence length, batch size 512, learning rate 5e-5
    * 100.000 steps - 512 sequence length, batch size 256, learning rate 2e-5
* The model was trained on a single Google Cloud TPU v3-8 

### Cased

* Same approach as uncased model, with the following differences:
    * Whole Word Masking
* Training set-up:
    * 100.000 steps - 128 sequence length, batch size 2048, learning rate 1e-4 (10.000 steps warmup)
    * 100.000 steps - 128 sequence length, batch size 2048, learning rate 5e-5
    * 100.000 steps - 512 sequence length, batch size 256, learning rate 2e-5


## Usage
Polbert is released via [HuggingFace Transformers library](https://huggingface.co/transformers/).

For an example use as language model, see [this notebook](/LM_testing.ipynb) file. 

### Uncased

```python
from transformers import *
model = BertForMaskedLM.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer)
for pred in nlp(f"Adam Mickiewicz wielkim polskim {nlp.tokenizer.mask_token} był."):
  print(pred)
# Output:
# {'sequence': '[CLS] adam mickiewicz wielkim polskim poeta był. [SEP]', 'score': 0.47196975350379944, 'token': 26596}
# {'sequence': '[CLS] adam mickiewicz wielkim polskim bohaterem był. [SEP]', 'score': 0.09127858281135559, 'token': 10953}
# {'sequence': '[CLS] adam mickiewicz wielkim polskim człowiekiem był. [SEP]', 'score': 0.0647173821926117, 'token': 5182}
# {'sequence': '[CLS] adam mickiewicz wielkim polskim pisarzem był. [SEP]', 'score': 0.05232388526201248, 'token': 24293}
# {'sequence': '[CLS] adam mickiewicz wielkim polskim politykiem był. [SEP]', 'score': 0.04554257541894913, 'token': 44095}
```

### Cased

```python
model = BertForMaskedLM.from_pretrained("dkleczek/bert-base-polish-cased-v1")
tokenizer = BertTokenizer.from_pretrained("dkleczek/bert-base-polish-cased-v1")
nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer)
for pred in nlp(f"Adam Mickiewicz wielkim polskim {nlp.tokenizer.mask_token} był."):
  print(pred)
# Output:
# {'sequence': '[CLS] Adam Mickiewicz wielkim polskim pisarzem był. [SEP]', 'score': 0.5391148328781128, 'token': 37120}
# {'sequence': '[CLS] Adam Mickiewicz wielkim polskim człowiekiem był. [SEP]', 'score': 0.11683262139558792, 'token': 6810}
# {'sequence': '[CLS] Adam Mickiewicz wielkim polskim bohaterem był. [SEP]', 'score': 0.06021466106176376, 'token': 17709}
# {'sequence': '[CLS] Adam Mickiewicz wielkim polskim mistrzem był. [SEP]', 'score': 0.051870670169591904, 'token': 14652}
# {'sequence': '[CLS] Adam Mickiewicz wielkim polskim artystą był. [SEP]', 'score': 0.031787533313035965, 'token': 35680}
```

See the next section for an example usage of Polbert in downstream tasks. 

## Evaluation
Thanks to Allegro, we now have the [KLEJ benchmark](https://klejbenchmark.com/leaderboard/), a set of nine evaluation tasks for the Polish language understanding. The following results are achieved by running standard set of evaluation scripts (no tricks!) utilizing both cased and uncased variants of Polbert.

| Model	| Average |	NKJP-NER | CDSC-E |	CDSC-R |	CBD	| PolEmo2.0-IN |	PolEmo2.0-OUT |	DYK	| PSC |	AR |
| ------------- |--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| Polbert cased | 81.7 |	93.6 |	93.4 |	93.8 |	52.7 |	87.4 |	71.1 |	59.1 |	98.6 |	85.2 |
| Polbert uncased | 81.4 |	90.1 |	93.9 |	93.5 |	55.0 |	88.1 |	68.8 |	59.4 |	98.8 |	85.4 |

Note how the uncased model performs better than cased on some tasks? My guess this is because of the oversampling of Open Subtitles dataset and its similarity to data in some of these tasks. All these benchmark tasks are sequence classification, so the relative strength of the cased model is not so visible here. 

## Bias
The data used to train the model is biased. It may reflect stereotypes related to gender, ethnicity etc. Please be careful when using the model for downstream task to consider these biases and mitigate them.  

## Acknowledgements
* I'd like to express my gratitude to Google [TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) for providing the free TPU credits - thank you!
* Also appreciate the help from Timo Möller from [deepset](https://deepset.ai) for sharing tips and scripts based on their experience training German BERT model.
* Big thanks to Allegro for releasing KLEJ Benchmark and specifically to Piotr Rybak for help with the evaluation and pointing out some issues with the tokenization. 
* Finally, thanks to Rachel Thomas, Jeremy Howard and Sylvain Gugger from [fastai](https://www.fast.ai) for their NLP and Deep Learning courses! 

## Author
Darek Kłeczek - contact me on Twitter [@dk21](https://twitter.com/dk21)

## References
* https://github.com/google-research/bert
* https://github.com/narusemotoki/srx_segmenter
* SRX rules file for sentence splitting in Polish, written by Marcin Miłkowski: https://raw.githubusercontent.com/languagetool-org/languagetool/master/languagetool-core/src/main/resources/org/languagetool/resource/segment.srx
* [KLEJ benchmark](https://klejbenchmark.com/leaderboard/)