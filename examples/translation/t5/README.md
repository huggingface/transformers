### Intro

This example shows how T5 (here the official [paper](https://arxiv.org/abs/1910.10683)) can be
evaluated on the WMT English-German dataset.

### Get the WMT Data

To be able to reproduce the authors' results on WMT English to German, you first need to download 
the WMT14 en-de news dataset. 
Go on Stanford's official NLP [website](https://nlp.stanford.edu/projects/nmt/) and find "newstest2013.en" under WMT'14 English-German data or download the dataset directly via:

```bash
curl https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en > newstest2013.en
```

You should have 3000 sentence in your file. You can verify this by running:

```bash
wc -l newstest2013.en
```

### Usage

Let's check the longest and shortest sentence in our file to find reasonable decoding hyperparameters: 

Get the longest and shortest sentence:

```bash 
awk '{print NF}' newstest2013.en | sort -n | head -1 # shortest sentence has 1 word
awk '{print NF}' newstest2013.en | sort -n | tail -1 # longest sentence has 106 words
```

We will set our `max_length` to ~3 times the longest sentence and leave `min_length` to its default value of 0.
We decode with beam search `num_beams=4` as proposed in the paper. Also as is common in beam search we set `early_stopping=True` and `length_penalty=2.0`.

To create translation for each in dataset, run:
```bash
python evaluate_wmt.py <path_to_newstest2013> newstest2013_de_translations.txt
```
the default batch size, 16, fits in 16GB GPU memory, but may need to be adjusted to fit your system.

### Where is the code?
The core model is in `src/transformers/modeling_t5.py`. This directory only contains examples.

### (WIP) BLEU Scores
