# GPT2-IMDB-ctrl

## What is it?
A small GPT2 (`lvwerra/gpt2-imdb`) language model fine-tuned to produce controlled movie reviews based the [IMDB dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The model is trained with rewards from a BERT sentiment classifier (`lvwerra/bert-imdb`) via PPO.

## Training setting
The model was trained for `200` optimisation steps with a batch size of `256` which corresponds to `51200` training samples. The full experiment setup can be found in the Jupyter notebook in the [trl repo](https://lvwerra.github.io/trl/05-gpt2-sentiment-ppo-training/). The strings `"[negative]"`, `"[neutral]"`, and `"[positive]"` are added at the beginning of the query to control the sentiment.

## Examples
A few examples of the model response to a query before and after optimisation:

| query | response [negative] | rewards [negative] | response [neutral] | rewards [neutral] | response [positive] | rewards [positive] |
|-------|---------------------|--------------------|--------------------|-------------------|---------------------|--------------------|
|I watched this movie when|it was released and was awful. Little bit of ...|3.130034|it was released and it was the first movie I ...|-1.351991|I was younger it was wonderful. The new play ...|4.232218|
|I can remember seeing this|movie in 2008, and I was so disappointed...yo...|3.428725|in support groups, which I think was not as i...|0.213288|movie, and it is one of my favorite movies ev...|4.168838|
|This 1970 hit film has|little resonance. This movie is bad, not only...|4.241872|a bit of Rocket power.783287. It can be easil...|0.849278|the best formula for comedy and is't just jus...|4.208804|


