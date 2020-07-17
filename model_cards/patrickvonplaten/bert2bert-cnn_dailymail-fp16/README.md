# Bert2Bert Summarization with 🤗 EncoderDecoder Framework

This model is a Bert2Bert model meaning that both the encoder and the decoder are `bert-base-uncased` 
BERT models. Leveraging the [EncoderDecoderFramework](https://huggingface.co/transformers/model_doc/encoderdecoder.html#encoder-decoder-models), the 
two pretrained models can simply be loaded into the framework via:

```python
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
```

The decoder of an `EncoderDecoder` model needs cross-attention layers and usually makes use of causal 
masking for auto-regressiv generation.
Therefore, such a `bert2bert` model has to be fine-tuned.

Here, `bert2bert` is fined-tuned on `CNN/Daily Mail`, which can easily be loaded via 🤗nlp:

```python 
nlp.load_dataset("cnn_dailymail", "3.0.0")
```

After fine-tuning, `bert2bert` is fine-tuned on summarization.

## Training script:

**IMPORTANT** In order for this code to work, make sure you checkout to the branch 
[more_general_trainer_metric](https://github.com/huggingface/transformers/tree/more_general_trainer_metric), which slightly adapts 
the `Trainer` for `EncoderDecoderModels` according to this PR: https://github.com/huggingface/transformers/pull/5840. 

The following code shows the complete training script that was used to fine-tune `bert2bert-cnn_dailymail-fp16
` for reproducability. The training last ~9h on a standard GPU.

```python
#!/usr/bin/env python3
import nlp
import logging
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO)

model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# CLS token will work as BOS token
tokenizer.bos_token = tokenizer.cls_token

# SEP token will work as EOS token
tokenizer.eos_token = tokenizer.sep_token

# load train and validation data
train_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="train")
val_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="validation[:10%]")

# load rouge for validation
rouge = nlp.load_metric("rouge")


# set decoding params
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4


# map data correctly
def map_to_encoder_decoder_inputs(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512)
    # force summarization <= 128
    outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=128)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    # mask loss for padding
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]
    batch["decoder_attention_mask"] = outputs.attention_mask

    assert all([len(x) == 512 for x in inputs.input_ids])
    assert all([len(x) == 128 for x in outputs.input_ids])
    
    return batch


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, clean_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, clean_special_tokens=True)

    pred_str = [pred.split("[CLS]")[-1].split("[SEP]")[0] for pred in pred_str]
    label_str = [label.split("[CLS]")[-1].split("[SEP]")[0] for label in label_str]

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    
    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# set batch size here
batch_size = 16

# make train dataset ready
train_dataset = train_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
)
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# same for validation dataset
val_dataset = val_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
)
val_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_from_generate=True,
    evaluate_during_training=True,
    do_train=True,
    do_eval=True,
    logging_steps=1000,
    save_steps=1000,
    eval_steps=1000,
    overwrite_output_dir=True,
    warmup_steps=2000,
    save_total_limit=10,
)

# instantiate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# start training
trainer.train()
```

