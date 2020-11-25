import os
import torch

from .modeling_tf_conv_bert import TFConvBertForSequenceClassification
from .modeling_conv_bert import ConvBertForSequenceClassification, ConvBertConfig, load_tf_weights_in_conv_bert
from .tokenization_conv_bert import ConvBertTokenizer

import tensorflow as tf

model_path = "/home/abhishek/huggingface/models/convbert_models/convbert_medium_small"

if __name__ == "__main__":
    tokenizer = ConvBertTokenizer.from_pretrained(model_path)
    model = ConvBertForSequenceClassification.from_pretrained(model_path)
    tf_model = TFConvBertForSequenceClassification.from_pretrained(model_path)

    print(model.dummy_inputs)
    print(model(**model.dummy_inputs))

    # tf_model = TFConvBertModel(conf)

    print(tf_model.dummy_inputs)
    print(tf_model(tf_model.dummy_inputs))

    # last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states)

    from transformers import BertTokenizer, BertForSequenceClassification
    import torch

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", return_dict=True)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    print(logits)