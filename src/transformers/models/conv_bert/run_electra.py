import torch

from ..electra.modeling_tf_electra import TFElectraModel, ElectraConfig
from ..electra.tokenization_electra import ElectraTokenizer

import tensorflow as tf


if __name__ == "__main__":
    conf = ElectraConfig.from_json_file("/home/abhishek/huggingface/models/convbert/config.json")
    print(conf)
    tokenizer = ElectraTokenizer.from_pretrained("/home/abhishek/huggingface/models/convbert/")
    model = TFElectraModel(conf)

    inputs = tokenizer("Hello, my dog is a very cute dog to be honest", return_tensors="tf")
    print(inputs)
    outputs = model(inputs)

    last_hidden_states = outputs.last_hidden_state
