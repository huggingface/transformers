import torch
import tensorflow as tf

from .modeling_tf_conv_bert import TFConvBertModel, ConvBertConfig
from .tokenization_conv_bert import ConvBertTokenizer

tf_checkpoint_path = "/home/abhishek/huggingface/models/convbert/model.ckpt"
conf = ConvBertConfig.from_json_file("/home/abhishek/huggingface/models/convbert/config.json")
tokenizer = ConvBertTokenizer.from_pretrained("/home/abhishek/huggingface/models/convbert/")


if __name__ == "__main__":
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
