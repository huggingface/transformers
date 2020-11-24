import torch

# from .modeling_tf_conv_bert import TFConvBertModel, ConvBertConfig
from .modeling_conv_bert import ConvBertModel, ConvBertConfig
from .tokenization_conv_bert import ConvBertTokenizer

import tensorflow as tf


if __name__ == "__main__":
    conf = ConvBertConfig.from_json_file("/home/abhishek/huggingface/models/convbert/config.json")
    print(conf)
    tokenizer = ConvBertTokenizer.from_pretrained("/home/abhishek/huggingface/models/convbert/")
    # model = TFConvBertModel(conf)
    model = ConvBertModel(conf)

    # inputs = tokenizer("Hello, my dog is a very cute dog to be honest", return_tensors="tf")
    inputs = tokenizer("Hello, my dog is a very cute dog to be honest", return_tensors="pt")
    print(inputs)
    outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states)
