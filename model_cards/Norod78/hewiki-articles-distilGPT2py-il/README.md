---
language: he

thumbnail: https://avatars1.githubusercontent.com/u/3617152?norod.jpg
widget:
- text: "<|startoftext|>החוק השני של מועדון קרב הוא"
- text: "<|startoftext|>ראש הממשלה בן גוריון"
- text: "<|startoftext|>למידת מכונה (סרט)"
- text: "<|startoftext|>מנשה פומפרניקל"
- text: "<|startoftext|>אי שוויון "

license: mit
---


# hewiki-articles-distilGPT2py-il

## A tiny GPT2 model for generating Hebrew text

A distilGPT2 sized model. <br>
Training data was hewiki-20200701-pages-articles-multistream.xml.bz2 from https://dumps.wikimedia.org/hewiki/20200701/  <br>
XML has been converted to plain text using Wikipedia Extractor http://medialab.di.unipi.it/wiki/Wikipedia_Extractor  <br>
I then added <|startoftext|> and <|endoftext|> markers and deleted empty lines.  <br>

#### How to use

```python
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("Norod78/hewiki-articles-distilGPT2py-il")
model = GPT2LMHeadModel.from_pretrained("Norod78/hewiki-articles-distilGPT2py-il").eval()

bos_token = tokenizer.bos_token #Beginning of sentace 
eos_token = tokenizer.eos_token #End of sentence 

def generate_word(model, tokens_tensor, temperature=1.0):
  """ 
  Sample a word given a tensor of tokens of previous words from a model. Given 
  the words we have, sample a plausible word. Temperature is used for 
  controlling randomness. If using temperature==0 we simply use a greedy arg max. 
  Else, we sample from a multinomial distribution using a lower inverse 
  temperature to allow for more randomness to escape repetitions. 
  """
  with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]
    if temperature>0:
      # Make the distribution more or less skewed based on the temperature
      predictions = outputs[0]/temperature
      # Sample from the distribution
      softmax = nn.Softmax(dim=0)
      predicted_index = torch.multinomial(softmax(predictions[0,-1,:]),1).item()
    # Simply take the arg-max of the distribution
    else:
      predicted_index = torch.argmax(predictions[0, -1, :]).item()
    # Decode the encoding to the corresponding word
    predicted_text = tokenizer.decode([predicted_index])
  return predicted_text

def generate_sentence(model, tokenizer, initial_text, temperature=1.0):
  """ Generate a sentence given some initial text using a model and a tokenizer.
  Returns the new sentence. """
        
  # Encode a text inputs
  text = ""
  sentence = text

  # We avoid an infinite loop by setting a maximum range
  for i in range(0,84):
    indexed_tokens = tokenizer.encode(initial_text + text)
      
    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    
    new_word = generate_word(model, tokens_tensor, temperature=temperature)

    # Here the temperature is slowly decreased with each generated word,
    # this ensures that the sentence (ending) makes more sense.
    # We don't decrease to a temperature of 0.0 to leave some randomness in.
    if temperature<(1-0.008):
      temperature += 0.008
    else:
      temperature = 0.996

    text = text+new_word

    # Stop generating new words when we have reached the end of the line or the poem
    if eos_token in new_word:
      # returns new sentence and whether poem is done
      return (text.replace(eos_token,"").strip(), True)
    elif '/' in new_word:
      return (text.strip(), False)
    elif bos_token in new_word:
        return (text.replace(bos_token,"").strip(), False)
      
  return (text, True)

for output_num in range(1,5):
  init_text = "בוקר טוב"
  text = bos_token + init_text
  for i in range(0,84):
    sentence = generate_sentence(model, tokenizer, text, temperature=0.9)    
    text = init_text + sentence[0]
    print(text)
    if (sentence[1] == True):
      break   
```
