---
language: "en"
thumbnail: "https://raw.githubusercontent.com/stevhliu/satsuma/master/images/astroGPT-thumbnail.png"
widget:
- text: "Jan 18, 2020"
- text: "Feb 14, 2020"
- text: "Jul 04, 2020"
---

# astroGPT 🪐

## Model description

This is a GPT-2 model fine-tuned on Western zodiac signs. For more information about GPT-2, take a look at 🤗 Hugging Face's GPT-2 [model card](https://huggingface.co/gpt2). You can use astroGPT to generate a daily horoscope by entering the current date.

## How to use

To use this model, simply enter the current date like so `Mon DD, YEAR`:

```python
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("stevhliu/astroGPT")
model = AutoModelWithLMHead.from_pretrained("stevhliu/astroGPT")

input_ids = tokenizer.encode('Sep 03, 2020', return_tensors='pt').to('cuda')

sample_output = model.generate(input_ids,
                               do_sample=True, 
                               max_length=75,
                               top_k=20, 
                               top_p=0.97)
                                
print(sample_output)
```

## Limitations and bias

astroGPT inherits the same biases that affect GPT-2 as a result of training on a lot of non-neutral content on the internet. The model does not currently support zodiac sign-specific generation and only returns a general horoscope. While the generated text may occasionally mention a specific zodiac sign, this is  due to how the horoscopes were originally written by it's human authors.

## Data

The data was scraped from [Horoscope.com](https://www.horoscope.com/us/index.aspx) and trained on 4.7MB of text. The text was collected from four categories (daily, love, wellness, career) and span from 09/01/19 to 08/01/2020. The archives only store horoscopes dating a year back from the current date.

## Training and results

The text was tokenized using the fast GPT-2 BPE [tokenizer](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2tokenizerfast). It has a vocabulary size of 50,257 and sequence length of 1024 tokens. The model was trained with on one of Google Colaboratory's GPU's for approximately 2.5 hrs with [fastai's](https://docs.fast.ai/) learning rate finder, discriminative learning rates and 1cycle policy. See table below for a quick summary of the training procedure and results.

| dataset size  | epochs | lr                | training time | train_loss | valid_loss | perplexity | 
|:-------------:|:------:|:-----------------:|:-------------:|:----------:|:----------:|:----------:|
| 5.9MB         |32      | slice(1e-7,1e-5)  | 2.5 hrs       | 2.657170   | 2.642387   | 14.046692	|
