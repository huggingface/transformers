---
language: english
thumbnail:
---

# GPT2-IMDB-neutral (LM + RL) 🎞😐✍

## What is it?
A small GPT2 (`lvwerra/gpt2-imdb`) language model fine-tuned to produce **neutral**-ish movie reviews based on the [IMDB dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The model is trained with rewards from a BERT sentiment classifier (`lvwerra/gpt2-imdb`) via **PPO**.

## Why?
After reproducing the experiment [lvwerra/gpt2-imdb-pos](https://huggingface.co/lvwerra/gpt2-imdb-pos) but for generating **negative** movie reviews ([mrm8488/gpt2-imdb-neg](https://huggingface.co/mrm8488/gpt2-imdb-neg)) I wanted to check if I could generate neutral-ish movie reviews. So, based on the classifier output (logit), I saw that clearly negative reviews gives around *-4* values and clearly positive reviews around *4*. Then, it was esay to establish an interval ```[-1.75,1.75]``` that it could be considered as **neutral**. So if the classifier output was in that interval I gave it a positive reward while values out of the interval got a negative reward.

## Training setting
The model was trained for `100` optimisation steps with a batch size of `128` which corresponds to `30000` training samples. The full experiment setup (for positive samples) in [trl repo](https://lvwerra.github.io/trl/04-gpt2-sentiment-ppo-training/).

## Examples
A few examples of the model response to a query before and after optimisation:

| query | response (before) | response (after) | rewards (before) | rewards (after) |
|-------|-------------------|------------------|------------------|-----------------|
|Okay, my title is|partly over, but this drama still makes me proud to read its first 40...|weird. The title is "mana were, ahunter". "Man...|4.200727 |-1.891443|
|Where is it written that|there is a monster in this movie anyway? How is it that the entire|[ of the women in the recent women of jungle business between Gender and husband| -3.113942| -1.944993|
|As a lesbian, I|cannot believe I was in the Sixties! Subtle yet witty, with original| found it hard to get responsive. In fact I found myself with the long|	3.906178|	0.769166|
|The Derek's have over|three times as many acting hours than Jack Nicholson? You think bitches?|30 dueling characters and kill of, they retreat themselves to their base.|-2.503655| -1.898380|


> All credits to [@lvwerra](https://twitter.com/lvwerra)

> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
