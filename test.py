from transformers import LongformerTokenizer, TFLongformerForSequenceClassification, PretrainedConfig
from transformers import TFLongformerForMultipleChoice, TFLongformerForTokenClassification
import tensorflow as tf


config = PretrainedConfig.from_dict(
    config_dict={
        "attention_mode": "longformer",
        "attention_probs_dropout_prob": 0.1,
        "attention_window": [
            512,
            512
        ],
        "bos_token_id": 0,
        "eos_token_id": 2,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 128,
        "ignore_attention_mask": False,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "layer_norm_eps": 0.00001,
        "max_position_embeddings": 4098,
        "model_type": "longformer",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "pad_token_id": 1,
        "sep_token_id": 2,
        "type_vocab_size": 1,
        "vocab_size": 100
    }
)   

# test sequenceclassification
# tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
# model = TFLongformerForSequenceClassification(config=config)

# inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
# inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1)) # Batch size 1

# outputs = model(inputs, return_dict=True)
# print("ok")


# test multiplechoice
# tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
# model = TFLongformerForMultipleChoice(config)
# prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
# choice0 = "It is eaten with a fork and a knife."
# choice1 = "It is eaten while held in the hand."

# encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='tf', padding=True)
# inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
# outputs = model(inputs, return_dict=True)  # batch size is 1


# test TFLongformerForTokenClassification
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = TFLongformerForTokenClassification(config=config)
inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
input_ids = inputs["input_ids"]
inputs["labels"] = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))) # Batch size 1
outputs = model(inputs, return_dict=True)
print("ok")