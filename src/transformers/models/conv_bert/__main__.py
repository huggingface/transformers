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

    from transformers import convert_tf_weight_name_to_pt_weight_name
    import numpy as np

    pt_state_dict = model.state_dict()

    old_keys = []
    new_keys = []
    for key in pt_state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        pt_state_dict[new_key] = pt_state_dict.pop(old_key)

    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights

    start_prefix_to_remove = ""
    if not any(s.startswith(tf_model.base_model_prefix) for s in pt_state_dict.keys()):
        start_prefix_to_remove = tf_model.base_model_prefix + "."

    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        name, transpose = convert_tf_weight_name_to_pt_weight_name(
            sw_name, start_prefix_to_remove=start_prefix_to_remove
        )

        # Find associated numpy array in pytorch model state dict
        if name not in pt_state_dict:
            continue
            # raise AttributeError("{} not found in PyTorch model".format(name))
        array = pt_state_dict[name].numpy()
        # if np.array_equal(array, symbolic_weight):
        print("**********************************")
        print(name, sw_name)
        print(array.shape, symbolic_weight.shape)
        print(array)
        print(symbolic_weight)
        print("**********************************")
        break
