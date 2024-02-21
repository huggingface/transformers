import sys
import torch
sys.path.append('./transformers_outer/src')
from transformers2 import BertTokenizer

class SecondaryTokenizer:
  def __init__ (self, vocab_size=0, hidden_size=768):
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size

  def tokenize(self, string):
    pass



class WordPieceToCharTokenizer(SecondaryTokenizer):
  def __init__(self, hidden_size=768, tokenizer=None):
    if tokenizer == None:
      self.tokenizer = BertTokenizer.from_pretrained("cabrooks/LOGION-50k_wordpiece")
    else: 
      self.tokenizer = tokenizer
    super().__init__(hidden_size=hidden_size, vocab_size=self.tokenizer.vocab_size)

  def encode(self, token): # might need to change
    return self.tokenizer.encode(token)

  def tokenize(self, string): # not old man
      word_tok = self.tokenizer
      return_ids = ([], [])
      cls_wp = word_tok.encode('[CLS]')[1]
      return_ids[0].append(cls_wp)
      delim_words = string.split(" ")
      unk = '[UNK]'
      unk_id = word_tok.encode(unk)[1]
      pad = '[PAD]'
      pad_id = word_tok.encode(pad)[1]
      sep = '[SEP]'
      sep_id = word_tok.encode(sep)[1]
      for index, w in enumerate(delim_words):
          # dealing with a single word a time, here
          wp_toks_ids = word_tok.encode(w)[1:-1]
          wps = word_tok.convert_ids_to_tokens(wp_toks_ids)
          # have to avoid the CLS and SEP characters
          for i, wp in enumerate(wps):
              if wp == '[MASK]':
                  return_ids[0].append(unk_id)
              elif wp == '[UNK]':
                  return_ids[0].append(unk_id)
              elif wp == '[PAD]':
                  return_ids[0].append(pad_id)
              elif wp == '[SEP]':
                  return_ids[0].append(sep_id)
              else:
                  clean_wp = wp.replace("##", "")
                  for c in clean_wp:
                      if '[MASK]' in wps:
                          return_ids[0].append(unk_id)
                      else:
                          return_ids[0].append(wp_toks_ids[i])

      
      cls_wp = word_tok.encode('[SEP]')[1]
      return_ids[0].append(cls_wp)
      return torch.tensor(return_ids[0])