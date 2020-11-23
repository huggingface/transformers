### How to use

You can use this model directly with a pipeline for text generation. Since the generation relies on some randomness, we
set a seed for reproducibility:

```python
>>> from transformers import pipeline, set_seed
>>> generator = pipeline('text-generation', model='e-tony/gpt2-rnm')
>>> set_seed(42)
>>> generator("Rick: I turned myself into a pickle, Morty!\nMorty: ", max_length=50, num_return_sequences=5)

[{'generated_text': "Rick: I turned myself into a pickle, Morty!\nMorty:  I didn't want to have children. It was my fate! I'll pay my mom and dad.\nSnuffles:  Well, at least we"}, 
 {'generated_text': "Rick: I turned myself into a pickle, Morty!\nMorty:  you know what happened?\n(Steven begins dragging people down the toilet with his hand. As Steven falls) The whole thing starts.\nA man approaches Steven"}, 
 {'generated_text': "Rick: I turned myself into a pickle, Morty!\nMorty:  Oh wait! And do you remember what I did to you?\nJerry:  Uh, it didn't hurt. It should have hurt a lot since I"}, 
 {'generated_text': "Rick: I turned myself into a pickle, Morty!\nMorty:  Rick!\nKraven:  Wait! [wary gasp] What the hell are you doing this time?!\nJerry:  Hey, are you"}, 
 {'generated_text': "Rick: I turned myself into a pickle, Morty!\nMorty:  Uh.\nJerry:  You don't have to put your finger on me today, do you?\nRick:  It's just, what do you"}]
```

### Training data
We used the original `gpt2` model and fine-tuned it on [Rick and Morty transcripts](https://rickandmorty.fandom.com/wiki/Category:Transcripts).
