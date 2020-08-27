---
language: fr
tags:
- conversational
widget:
- text: "bonjour."
- text: "mais encore"
- text: "est ce que l'argent achete le bonheur?"
---

## a dialoggpt model trained on french opensubtitles with custom tokenizer
trained with this notebook
https://colab.research.google.com/drive/1pfCV3bngAmISNZVfDvBMyEhQKuYw37Rl#scrollTo=AyImj9qZYLRi&uniqifier=3

config from microsoft/DialoGPT-medium
### How to use

Now we are ready to try out how the model works as a chatting partner!

```python
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("cedpsam/chatbot_fr")

model = AutoModelWithLMHead.from_pretrained("cedpsam/chatbot_fr")

for step in range(6):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(
        bot_input_ids, max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.92, top_k = 50
    )
    
    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
