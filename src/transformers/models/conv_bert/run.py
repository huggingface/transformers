import os

import tensorflow as tf
import torch

from .modeling_conv_bert import ConvBertConfig, ConvBertModel, load_tf_weights_in_conv_bert
from .modeling_tf_conv_bert import TFConvBertModel
from .tokenization_conv_bert import ConvBertTokenizer


if __name__ == "__main__":
    model_path = "YituTech/conv-bert-medium-small"
    model = ConvBertModel.from_pretrained(model_path)
    model.eval()
    # model = load_tf_weights_in_conv_bert(model, conf, tf_checkpoint_path)
    # model.save_pretrained(model_path)

    for p in model.named_parameters():
        print(p)
        break

    tokenizer = ConvBertTokenizer.from_pretrained(model_path)
    s = "hi how are you"
    inputs_pt = tokenizer(s, return_tensors="pt")
    inputs_tf = tokenizer(s, return_tensors="tf")

    tf_model = TFConvBertModel.from_pretrained(model_path)
    tf_model.trainable = False
    # tf_model.save_pretrained(model_path)

    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        print(sw_name)
        print(symbolic_weight)
        break

    print(tf_model(inputs_tf).last_hidden_state)
    print(model(**inputs_pt).last_hidden_state)