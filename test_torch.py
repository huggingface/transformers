from transformers import LongformerTokenizer, LongformerForSequenceClassification, PretrainedConfig, LongformerForMultipleChoice, LongformerForTokenClassification
import torch

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
        "vocab_size": 1000
    }
)   

# tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
# model = LongformerForSequenceClassification(config=config)
# inputs = tokenizer("a", return_tensors="pt")
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# logits = outputs.logits


# tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
# model = LongformerForMultipleChoice(config=config)
# prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
# choice0 = "It is eaten with a fork and a knife."
# choice1 = "It is eaten while held in the hand."
# labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
# encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
# outputs = model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels)  # batch size is 1
# # the linear classifier still needs to be trained
# loss = outputs.loss
# logits = outputs.logits



tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerForTokenClassification(config=config)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels, return_dict=True)
loss = outputs.loss
logits = outputs.logits