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
    # model.save_pretrained(model_path)

    tf_model = TFConvBertModel.from_pretrained(model_path, from_pt=True)
    tf_model.trainable = False
    # tf_model.save_pretrained(model_path)

    # symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    # for symbolic_weight in symbolic_weights:
    #     sw_name = symbolic_weight.name
    #     print(symbolic_weight)

    print(tf_model.dummy_inputs)
    print(tf_model(tf_model.dummy_inputs).last_hidden_state)
