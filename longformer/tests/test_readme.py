import os
import tarfile
import unittest
import urllib

import torch
from transformers import RobertaTokenizer

from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size


class TestReadme(unittest.TestCase):
  def setUp(self) -> None:
    self.model_dir = "/tmp/longformer-base-4096"
    if os.path.exists(self.model_dir):
      # already have the model. nothing to do
      return

    # download zip
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz",
        "/tmp/longformer-base-4096.tar.gz")

    # unpack
    print("unpacking....")
    with tarfile.open("/tmp/longformer-base-4096.tar.gz") as tar:
      tar.extractall("/tmp")

  def test_something(self):
    config = LongformerConfig.from_pretrained(self.model_dir)
    # choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
    # 'n2': for regular n2 attantion
    # 'tvm': a custom CUDA kernel implementation of our sliding window attention
    # 'sliding_chunks': a PyTorch implementation of our sliding window attention
    config.attention_mode = 'sliding_chunks'

    model = Longformer.from_pretrained(self.model_dir, config=config)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.model_max_length = model.config.max_position_embeddings

    SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document

    input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(
        0)  # batch of size 1

    # TVM code doesn't work on CPU. Uncomment this if `config.attention_mode = 'tvm'`
    # model = model.cuda(); input_ids = input_ids.cuda()

    # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long,
                                device=input_ids.device)  # initialize to local attention
    attention_mask[:,
    [1, 4, 21, ]] = 2  # Set global attention based on the task. For example,
    # classification: the <s> token
    # QA: question tokens

    # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
    input_ids, attention_mask = pad_to_window_size(
        input_ids, attention_mask, config.attention_window[0],
        tokenizer.pad_token_id)

    output = model(input_ids, attention_mask=attention_mask)[0]

    # could have done more here....
    self.assertIsNotNone(output)

if __name__ == '__main__':
  unittest.main()
