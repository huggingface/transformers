---
thumbnail: https://huggingface.co/front/thumbnails/dialogpt.png
tags:
- conversational
license: mit
---

## A State-of-the-Art Large-scale Pretrained Response generation model (DialoGPT)

DialoGPT is a SOTA large-scale pretrained dialogue response generation model for multiturn conversations. 
The [human evaluation results](https://github.com/dreasysnail/Dialogpt_dev#human-evaluation) indicate that the response generated from DialoGPT is comparable to human response quality under a single-turn conversation Turing test.
The model is trained on 147M multi-turn dialogue from Reddit discussion thread. 

* Multi-turn generation examples from an interactive environment:

|Role | Response |
|---------|--------|
|User | Does money buy happiness? |
| Bot |  Depends how much money you spend on it .|
|User | What is the best way to buy happiness ? |
| Bot | You just have to be a millionaire by your early 20s, then you can be happy . |
|User |This is so difficult ! |
| Bot | You have no idea how hard it is to be a millionaire and happy . There is a reason the rich have a lot of money |

Please find the information about preprocessing, training and full details of the DialoGPT in the [original DialoGPT repository](https://github.com/microsoft/DialoGPT)

ArXiv paper: [https://arxiv.org/abs/1911.00536](https://arxiv.org/abs/1911.00536)

### How to use

Now we are ready to try out how the model works as a chatting partner!

```python
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")

# Let's chat for 5 lines
for step in range(5):
	# encode the new user input, add the eos_token and return a tensor in Pytorch
	new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

	# append the new user input tokens to the chat history
	bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

	# generated a response while limiting the total chat history to 1000 tokens, 
	chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

	# pretty print last ouput tokens from bot
	print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
```
