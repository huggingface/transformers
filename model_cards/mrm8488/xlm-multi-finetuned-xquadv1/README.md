---
language: multilingual
thumbnail:
---

# [XLM](https://github.com/facebookresearch/XLM/) (multilingual version) fine-tuned for multilingual Q&A

Released from `Facebook` together with the paper [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample and Alexis Conneau and fine-tuned on [XQuAD](https://github.com/deepmind/xquad) for multilingual (`11 different languages`) **Q&A** downstream task.

## Details of the language model('xlm-mlm-100-1280')

[Language model](https://github.com/facebookresearch/XLM/#ii-cross-lingual-language-model-pretraining-xlm)

| Languages
| --------- |
| 100 |

It includes the following languages:

<details>
en-es-fr-de-zh-ru-pt-it-ar-ja-id-tr-nl-pl-simple-fa-vi-sv-ko-he-ro-no-hi-uk-cs-fi-hu-th-da-ca-el-bg-sr-ms-bn-hr-sl-zh_yue-az-sk-eo-ta-sh-lt-et-ml-la-bs-sq-arz-af-ka-mr-eu-tl-ang-gl-nn-ur-kk-be-hy-te-lv-mk-zh_classical-als-is-wuu-my-sco-mn-ceb-ast-cy-kn-br-an-gu-bar-uz-lb-ne-si-war-jv-ga-zh_min_nan-oc-ku-sw-nds-ckb-ia-yi-fy-scn-gan-tt-am
</details>

## Details of the downstream task (multilingual Q&A) - Dataset

Deepmind [XQuAD](https://github.com/deepmind/xquad)

Languages covered:

- Arabic: `ar`
- German: `de`
- Greek: `el`
- English: `en`
- Spanish: `es`
- Hindi: `hi`
- Russian: `ru`
- Thai: `th`
- Turkish: `tr`
- Vietnamese: `vi`
- Chinese: `zh`

As the dataset is based on SQuAD v1.1, there are no unanswerable questions in the data. We chose this
setting so that models can focus on cross-lingual transfer.

We show the average number of tokens per paragraph, question, and answer for each language in the
table below. The statistics were obtained using [Jieba](https://github.com/fxsjy/jieba) for Chinese
and the [Moses tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl)
for the other languages.

|           |  en   |  es   |  de   |  el   |  ru   |  tr   |  ar   |  vi   |  th   |  zh   |  hi   |
| --------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Paragraph | 142.4 | 160.7 | 139.5 | 149.6 | 133.9 | 126.5 | 128.2 | 191.2 | 158.7 | 147.6 | 232.4 |
| Question  | 11.5  | 13.4  | 11.0  | 11.7  | 10.0  |  9.8  | 10.7  | 14.8  | 11.5  | 10.5  | 18.7  |
| Answer    |  3.1  |  3.6  |  3.0  |  3.3  |  3.1  |  3.1  |  3.1  |  4.5  |  4.1  |  3.5  |  5.6  |

Citation:

<details>

```bibtex
@article{Artetxe:etal:2019,
      author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
      title     = {On the cross-lingual transferability of monolingual representations},
      journal   = {CoRR},
      volume    = {abs/1910.11856},
      year      = {2019},
      archivePrefix = {arXiv},
      eprint    = {1910.11856}
}
```

</details>

As XQuAD is just an evaluation dataset, I used Data augmentation techniques (scraping, neural machine translation, etc) to obtain more samples and splited the dataset in order to have a train and test set. The test set was created in a way that contains the same number of samples for each language. Finally, I got:

| Dataset     | # samples |
| ----------- | --------- |
| XQUAD train | 50 K      |
| XQUAD test  | 8 K       |

## Model training

The model was trained on a Tesla P100 GPU and 25GB of RAM.
The script for fine tuning can be found [here](https://github.com/huggingface/transformers/blob/master/examples/distillation/run_squad_w_distillation.py)


## Model in action

Fast usage with **pipelines**:

```python
from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/xlm-multi-finetuned-xquadv1",
    tokenizer="mrm8488/xlm-multi-finetuned-xquadv1"
)

# English
qa_pipeline({
    'context': "Manuel Romero has been working hardly in the repository hugginface/transformers lately",
    'question': "Who has been working hard for hugginface/transformers lately?"
})

#Output: {'answer': 'Manuel', 'end': 6, 'score': 8.531880747878265e-05, 'start': 0}

# Russian
qa_pipeline({
    'context': "Мануэль Ромеро в последнее время почти не работал в репозитории hugginface / transformers",
    'question': "Кто в последнее время усердно работал над обнимашками / трансформерами?"
    
})

#Output: {'answer': 'работал в репозитории hugginface /','end': 76, 'score': 0.00012340750456964894, 'start': 42}
```
Try it on a Colab (*Do not forget to change the model and tokenizer path in the Colab if necessary*):

<a href="https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/Try_mrm8488_xquad_finetuned_uncased_model.ipynb" target="_parent"><img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a>

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
