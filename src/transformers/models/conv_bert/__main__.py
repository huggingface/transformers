import os
import torch

from .modeling_tf_conv_bert import TFConvBertModel
from .modeling_conv_bert import ConvBertModel, ConvBertConfig, load_tf_weights_in_conv_bert
from .tokenization_conv_bert import ConvBertTokenizer

import tensorflow as tf

model_path = "/home/abhishek/huggingface/models/convbert_models/convbert_small"

if __name__ == "__main__":
    conf = ConvBertConfig.from_json_file(os.path.join(model_path, "config.json"))
    tokenizer = ConvBertTokenizer.from_pretrained(model_path)
    tf_checkpoint_path = os.path.join(model_path, "model.ckpt")
    model = ConvBertModel(conf)

    model = load_tf_weights_in_conv_bert(model, conf, tf_checkpoint_path)
    model.save_pretrained(model_path)

    tf_model = TFConvBertModel.from_pretrained(model_path, from_pt=True)
    tf_model.save_pretrained(model_path)

    print(model.dummy_inputs)
    print(model(**model.dummy_inputs).last_hidden_state)

    # tf_model = TFConvBertModel(conf)

    print(tf_model.dummy_inputs)
    print(tf_model(tf_model.dummy_inputs).last_hidden_state)

    # last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states)
