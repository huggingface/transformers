
# GPT2 Genre Based Story Generator

## Model description

GPT2 fine-tuned on genre-based story generation.

## Intended uses

Used to generate stories based on user inputted genre and starting prompts.

## How to use

#### Supported Genres
superhero, action, drama, horror, thriller, sci_fi
#### Input text format
\<BOS> \<genre> Some optional text...

**Example**: \<BOS> \<sci_fi> After discovering time travel,

```python
# Example of usage
from transformers import pipeline

story_gen = pipeline("text-generation", "pranavpsv/gpt2-genre-story-generator")
print(story_gen("<BOS> <superhero> Batman"))

```

## Training data

Initialized with pre-trained weights of "gpt2" checkpoint. Fine-tuned the model on stories of various genres.
