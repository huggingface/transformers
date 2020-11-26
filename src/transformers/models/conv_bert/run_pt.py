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
    model.eval()
    # for p in model.named_parameters():
    #     print(p)

    print(model.dummy_inputs)
    print(model(**model.dummy_inputs).last_hidden_state)
