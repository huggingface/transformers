import torch
from .modeling_tf_conv_bert import TFConvBertModel, ConvBertConfig
from .tokenization_conv_bert import ConvBertTokenizer

import tensorflow as tf


if __name__ == "__main__":
    conf = ConvBertConfig.from_json_file("/home/abhishek/huggingface/models/convbert/config.json")
    #
    # ei = ElectraModel(conf)
    # x = torch.randint(0, 100, (4, 64))
    # print(ei(input_ids=x))
    tokenizer = ConvBertTokenizer.from_pretrained("google/electra-small-discriminator")
    model = TFConvBertModel(conf)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    outputs = model(inputs)

    last_hidden_states = outputs.last_hidden_states