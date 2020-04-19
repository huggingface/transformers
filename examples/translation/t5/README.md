***This script evaluates the multitask pre-trained checkpoint for ``t5-base`` (see paper [here](https://arxiv.org/pdf/1910.10683.pdf)) on the English to German WMT dataset. Please note that the results in the paper were attained using a model fine-tuned on translation, so that results will be worse here by approx. 1.5 BLEU points***

### Intro

This example shows how T5 (here the official [paper](https://arxiv.org/abs/1910.10683)) can be
evaluated on the WMT English-German dataset.

### Get the WMT Data

To be able to reproduce the authors' results on WMT English to German, you first need to download 
the WMT14 en-de news datasets.
Go on Stanford's official NLP [website](https://nlp.stanford.edu/projects/nmt/) and find "newstest2014.en" and "newstest2014.de" under WMT'14 English-German data or download the dataset directly via:

```bash
curl https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en > newstest2014.en
curl https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de > newstest2014.de
```

You should have 2737 sentences in each file. You can verify this by running:

```bash
wc -l newstest2014.en  # should give 2737
```

### Usage

Let's check the longest and shortest sentence in our file to find reasonable decoding hyperparameters: 

Get the longest and shortest sentence:

```bash 
awk '{print NF}' newstest2014.en | sort -n | head -1 # shortest sentence has 2 word
awk '{print NF}' newstest2014.en | sort -n | tail -1 # longest sentence has 91 words
```

We will set our `max_length` to ~3 times the longest sentence and leave `min_length` to its default value of 0.
We decode with beam search `num_beams=4` as proposed in the paper. Also as is common in beam search we set `early_stopping=True` and `length_penalty=2.0`.

To create translation for each in dataset and get a final BLEU score, run:
```bash
python evaluate_wmt.py <path_to_newstest2014.en> newstest2014_de_translations.txt <path_to_newstest2014.de> newsstest2014_en_de_bleu.txt
```
the default batch size, 16, fits in 16GB GPU memory, but may need to be adjusted to fit your system.

### Where is the code?
The core model is in `src/transformers/modeling_t5.py`. This directory only contains examples.

### BLEU Scores

The BLEU score is calculated using [sacrebleu](https://github.com/mjpost/sacreBLEU) by mjpost.
To get the BLEU score we used 
