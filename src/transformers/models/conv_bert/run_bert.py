import os

import tensorflow as tf
import torch

from ..bert.modeling_bert import BertConfig, BertModel, load_tf_weights_in_bert
from ..bert.modeling_tf_bert import TFBertModel
from ..bert.tokenization_bert import BertTokenizer


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tf_checkpoint_path = os.path.join(model_path, "model.ckpt")
    model = BertModel.from_pretrained("bert-base-uncased")

    # model = load_tf_weights_in_conv_bert(model, conf, tf_checkpoint_path)
    # model.save_pretrained(model_path)

    tf_model = TFBertModel.from_pretrained("bert-base-uncased")
    # tf_model.save_pretrained(model_path)

    print(model.dummy_inputs)
    print(model(**model.dummy_inputs))

    # tf_model = TFConvBertModel(conf)

    print(tf_model.dummy_inputs)
    print(tf_model(tf_model.dummy_inputs))

    # last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states)
