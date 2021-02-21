import os

import tensorflow as tf
import torch

from .modeling_convbert import ConvBertConfig, ConvBertModel, load_tf_weights_in_convbert
from .modeling_tf_convbert import TFConvBertModel
from .tokenization_convbert import ConvBertTokenizer


model_path = "/home/abhishek/convbert_models/convbert_base"

if __name__ == "__main__":
    conf = ConvBertConfig.from_json_file(os.path.join(model_path, "config.json"))
    tokenizer = ConvBertTokenizer.from_pretrained(model_path)
    tf_checkpoint_path = os.path.join(model_path, "model.ckpt")
    model = ConvBertModel(conf)

    model = load_tf_weights_in_convbert(model, conf, tf_checkpoint_path)
    model.eval()
    model.save_pretrained(model_path)

    tf_model = TFConvBertModel.from_pretrained(model_path, from_pt=True)
    tf_model.trainable = False
    tf_model.save_pretrained(model_path)

    tokenizer = ConvBertTokenizer.from_pretrained(model_path)
    s = "hi how are you"
    inputs_pt = tokenizer(s, return_tensors="pt")
    inputs_tf = tokenizer(s, return_tensors="tf")

    print(tf_model(inputs_tf))
    print(model(**inputs_pt))