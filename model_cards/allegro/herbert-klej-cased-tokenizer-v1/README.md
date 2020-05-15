---
language: polish
---

# HerBERT tokenizer

**[HerBERT](https://en.wikipedia.org/wiki/Zbigniew_Herbert)** tokenizer is a character level byte-pair encoding with
vocabulary size of 50k tokens. The tokenizer was trained on [Wolne Lektury](https://wolnelektury.pl/) and a publicly available subset of
[National Corpus of Polish](http://nkjp.pl/index.php?page=14&lang=0) with [fastBPE](https://github.com/glample/fastBPE) library.
Tokenizer utilize `XLMTokenizer` implementation from [transformers](https://github.com/huggingface/transformers).

## Tokenizer usage
Herbert tokenizer should be used together with [HerBERT model](https://huggingface.co/allegro/herbert-klej-cased-v1):
```python
from transformers import XLMTokenizer, RobertaModel

tokenizer = XLMTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1")

encoded_input = tokenizer.encode("Kto ma lepszą sztukę, ma lepszy rząd – to jasne.", return_tensors='pt')
outputs = model(encoded_input)
```

## License
CC BY-SA 4.0

## Citation
If you use this tokenizer, please cite the following paper:
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
Tokenizer was created by **Allegro Machine Learning Research** team.

You can contact us at: <a href="mailto:klejbenchmark@allegro.pl">klejbenchmark@allegro.pl</a>
