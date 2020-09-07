---
widget:
- text: "Even the Dwarves"
- text: "The secrets of"
---

# Model name
Magic The Generating

## Model description

This is a fine tuned GPT-2 model trained on a corpus of all available English language Magic the Gathering card flavour texts.

## Intended uses & limitations

This is intended only for use in generating new, novel, and sometimes surprising, MtG like flavour texts.

#### How to use

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("rjbownes/Magic-The-Generating")

model = GPT2LMHeadModel.from_pretrained("rjbownes/Magic-The-Generating")

```

#### Limitations and bias

The training corpus was surprisingly small, only ~29000 cards, I had suspected there were more. This might mean there is a real limit to the number of entirely original strings this will generate.
This is also only based on the 117M parameter GPT2, it's a pretty obvious upgrade to retrain with medium, large or XL models. However, despite this, the outputs I tested were very convincing!

## Training data

The data was 29222 MtG card flavour texts. The model was based on the "gpt2" pretrained transformer: https://huggingface.co/gpt2.

## Training procedure

Only English language MtG flavour texts were scraped from the [Scryfall](https://scryfall.com/) API. Empty strings and any non-UTF-8 encoded tokens were removed leaving 29222 entries.
This was trained using google Colab with a T4 instance. 4 epochs, adamW optimizer with default parameters and a batch size of 32. Token embedding lengths were capped at 98 tokens as this was the longest string and an attention mask was added to the training model to ignore all padding tokens.

## Eval results

Average Training Loss: 0.44866578806635815.
Validation loss: 0.5606984243444775.

Sample model outputs:

1. "Every branch a crossroads, every vine a swift steed."
	—Gwendlyn Di Corci

2. "The secrets of this world will tell their masters where to strike if need be."
	—Noyan Dar, Tazeem roilmage

3. "The secrets of nature are expensive. You'd be better off just to have more freedom."

4. "Even the Dwarves knew to leave some stones unturned."

5. "The wise always keep an ear open to the whispers of power."

### BibTeX entry and citation info

```bibtex
@article{BownesLM,
  title={Fine Tuning GPT-2 for Magic the Gathering flavour text generation.},
  author={Richard J. Bownes},
  journal={Medium},
  year={2020}
}

```
