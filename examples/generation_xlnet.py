import torch
from torch.nn import functional as F
from pytorch_pretrained_bert import XLNetModel, XLNetLMHeadModel, XLNetTokenizer

import logging
logging.basicConfig(level=logging.INFO)

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased', attn_type='uni')

tokens = tokenizer.encode('I am very happy')
for i in range(len(tokens), 20):
    mask = torch.tensor([[[0.0] * i + [1.0]]])
    logits, _ = model(torch.tensor([tokens + [0]]),
                    #   perm_mask=mask.expand(-1, i+1, -1),
                      target_mapping=mask,
                      inp_q=mask.squeeze(1))
    output = torch.multinomial(F.softmax(logits[0, 0, :]), 1)
    tokens.append(output.item())
    print(tokenizer.decode(tokens))
