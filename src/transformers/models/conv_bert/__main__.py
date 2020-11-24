import torch
from .modeling_tf_conv_bert import TFConvBertModel, ConvBertConfig
from .tokenization_conv_bert import ConvBertTokenizer

import tensorflow as tf


if __name__ == "__main__":
    conf = ConvBertConfig.from_pretrained("/home/abhishek/huggingface/models/convbert/config.json")
    print(conf)
    # ei = ElectraModel(conf)
    # x = torch.randint(0, 100, (4, 64))
    # print(ei(input_ids=x))
    tokenizer = ConvBertTokenizer.from_pretrained("/home/abhishek/huggingface/models/convbert/")
    model = TFConvBertModel(conf)

    inputs = tokenizer("Hello, my dog is a very cute dog to be honest", return_tensors="tf")
    print(inputs)
    outputs = model(inputs)

    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states)
