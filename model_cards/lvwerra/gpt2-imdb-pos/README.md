# GPT2-IMDB-pos

## What is it?
A small GPT2 (`lvwerra/gpt2-imdb`) language model fine-tuned to produce positive movie reviews based the [IMDB dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The model is trained with rewards from a BERT sentiment classifier (`lvwerra/gpt2-imdb`) via PPO.

## Training setting
The model was trained for `100` optimisation steps with a batch size of `256` which corresponds to `25600` training samples. The full experiment setup can be found in the Jupyter notebook in the [trl repo](https://lvwerra.github.io/trl/04-gpt2-sentiment-ppo-training/).

## Examples
A few examples of the model response to a query before and after optimisation:

| query | response (before) | response (after) | rewards (before) | rewards (after) |
|-------|-------------------|------------------|------------------|-----------------|
|I'd never seen a |heavier, woodier example of Victorian archite... |film of this caliber, and I think it's wonder... |3.297736 |4.158653|
|I love John's work	|but I actually have to write language as in w... |and I hereby recommend this film. I am really... |-1.904006 |4.159198 |
|I's a big struggle |to see anyone who acts in that way. by Jim Th... |, but overall I'm happy with the changes even ... |-1.595925 |2.651260|


