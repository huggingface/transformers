# GPT2-IMDB-neg (LM + RL) 🎞😡✍

All credits to [@lvwerra](https://twitter.com/lvwerra)

## What is it?
A small GPT2 (`lvwerra/gpt2-imdb`) language model fine-tuned to produce **negative** movie reviews based the [IMDB dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The model is trained with rewards from a BERT sentiment classifier (`lvwerra/gpt2-imdb`) via **PPO**.

## Why?
I wanted to reproduce the experiment [lvwerra/gpt2-imdb-pos](https://huggingface.co/lvwerra/gpt2-imdb-pos) but for generating **negative** movie reviews.

## Training setting
The model was trained for `100` optimisation steps with a batch size of `256` which corresponds to `25600` training samples. The full experiment setup (for positive samples) in [trl repo](https://lvwerra.github.io/trl/04-gpt2-sentiment-ppo-training/).

## Examples
A few examples of the model response to a query before and after optimisation:

| query | response (before) | response (after) | rewards (before) | rewards (after) |
|-------|-------------------|------------------|------------------|-----------------|
|This movie is a fine |	attempt as far as live action is concerned, n...|example of how bad Hollywood in theatrics pla...|	2.118391 |	-3.31625|
|I have watched 3 episodes |with this guy and he is such a talented actor...|	but the show is just plain awful and there ne...|	2.681171|	-4.512792|
|We know that firefighters and|	police officers are forced to become populari...|	other chains have going to get this disaster ...|	1.367811|	-3.34017|



> Created by [Manuel Romero/@mrm8488](https://twitter.com/mrm8488)

> Made with <span style="color: #e25555;">&hearts;</span> in Spain
