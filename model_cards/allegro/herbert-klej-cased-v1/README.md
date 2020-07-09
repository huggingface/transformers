---
language: polish
---

# HerBERT 
**[HerBERT](https://en.wikipedia.org/wiki/Zbigniew_Herbert)** is a BERT-based Language Model trained on Polish Corpora
using only MLM objective with dynamic masking of whole words. For more details, please refer to: 
[KLEJ: Comprehensive Benchmark for Polish Language Understanding](https://arxiv.org/abs/2005.00630).

## Dataset
**HerBERT** training dataset is a combination of several publicly available corpora for Polish language:

| Corpus | Tokens | Texts |
| :------ | ------: | ------: |
| [OSCAR](https://traces1.inria.fr/oscar/)| 6710M  | 145M |
| [Open Subtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php) | 1084M  | 1.1M |
| [Wikipedia](https://dumps.wikimedia.org/) | 260M  | 1.5M |
| [Wolne Lektury](https://wolnelektury.pl/) | 41M  | 5.5k |
| [Allegro Articles](https://allegro.pl/artykuly) | 18M  | 33k |

## Tokenizer
The training dataset was tokenized into subwords using [HerBERT Tokenizer](https://huggingface.co/allegro/herbert-klej-cased-tokenizer-v1); a character level byte-pair encoding with
a vocabulary size of 50k tokens. The tokenizer itself was trained on [Wolne Lektury](https://wolnelektury.pl/) and a publicly available subset of 
[National Corpus of Polish](http://nkjp.pl/index.php?page=14&lang=0) with a [fastBPE](https://github.com/glample/fastBPE) library.

Tokenizer utilizes `XLMTokenizer` implementation for that reason, one should load it as `allegro/herbert-klej-cased-tokenizer-v1`.

## HerBERT models summary
| Model | WWM | Cased | Tokenizer | Vocab Size  | Batch Size | Train Steps |
| :------ | ------: | ------: | ------: | ------: | ------: | ------: |
| herbert-klej-cased-v1 | YES | YES | BPE | 50K | 570 | 180k | 

## Model evaluation
HerBERT was evaluated on the [KLEJ](https://klejbenchmark.com/) benchmark, publicly available set of nine evaluation tasks for the Polish language understanding.
It had the best average performance and obtained the best results for three of them.

| Model | Average | NKJP-NER | CDSC-E | CDSC-R | CBD | PolEmo2.0-IN	|PolEmo2.0-OUT | DYK | PSC | AR	|
| :------ | ------: | ------: | ------: | ------: | ------: | ------: | ------: |  ------: | ------: | ------: |
| herbert-klej-cased-v1 | **80.5** | 92.7 | 92.5 | 91.9 | **50.3** | **89.2** |**76.3** |52.1 |95.3 | 84.5 |

Full leaderboard is available [online](https://klejbenchmark.com/leaderboard). 


## HerBERT usage
Model training and experiments were conducted with [transformers](https://github.com/huggingface/transformers) in version 2.0.

Example code:
```python
from transformers import XLMTokenizer, RobertaModel

tokenizer = XLMTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1")

encoded_input = tokenizer.encode("Kto ma lepszą sztukę, ma lepszy rząd – to jasne.", return_tensors='pt')
outputs = model(encoded_input)
```

HerBERT can also be loaded using `AutoTokenizer` and `AutoModel`:

```python
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
model = AutoModel.from_pretrained("allegro/herbert-klej-cased-v1")
```

## License
CC BY-SA 4.0

## Citation
If you use this model, please cite the following paper:
```
@misc{rybak2020klej,
    title={KLEJ: Comprehensive Benchmark for Polish Language Understanding},
    author={Piotr Rybak and Robert Mroczkowski and Janusz Tracz and Ireneusz Gawlik},
    year={2020},
    eprint={2005.00630},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
Paper is accepted at ACL 2020, as soon as proceedings appear, we will update the BibTeX.

## Authors
Model was trained by **Allegro Machine Learning Research** team.

You can contact us at: <a href="mailto:klejbenchmark@allegro.pl">klejbenchmark@allegro.pl</a>
