import tensorflow as tf
import torch

from .modeling_tf_conv_bert import ConvBertConfig, TFConvBertModel
from .tokenization_conv_bert import ConvBertTokenizer


tf_checkpoint_path = "/home/abhishek/convbert_models/convbert_medium_small/model.ckpt"

if __name__ == "__main__":
    conf = ConvBertConfig.from_json_file("/home/abhishek/convbert_models/convbert_medium_small/config.json")
    tokenizer = ConvBertTokenizer.from_pretrained("/home/abhishek/convbert_models/convbert_medium_small/")
    init_vars = tf.train.list_variables(tf_checkpoint_path)
    model = TFConvBertModel(conf)
    tf_inputs = model.dummy_inputs
    model(tf_inputs, training=False)
    print(model.base_model_prefix)

    weight_dict = {}
    for name, shape in init_vars:
        array = tf.train.load_variable(tf_checkpoint_path, name)
        weight_dict[name] = array

    print(model.summary())