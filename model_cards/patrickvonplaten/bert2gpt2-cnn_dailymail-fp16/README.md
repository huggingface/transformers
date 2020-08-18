# Bert2GPT2 Summarization with 🤗 EncoderDecoder Framework

This model is a Bert2Bert model fine-tuned on summarization.

Bert2GPT2 is a `EncoderDecoderModel`, meaning that the encoder is a `bert-base-uncased` 
BERT model and the decoder is a `gpt2` GPT2 model. Leveraging the [EncoderDecoderFramework](https://huggingface.co/transformers/model_doc/encoderdecoder.html#encoder-decoder-models), the 
two pretrained models can simply be loaded into the framework via:

```python
bert2gpt2 = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "gpt2")
```

The decoder of an `EncoderDecoder` model needs cross-attention layers and usually makes use of causal 
masking for auto-regressiv generation.
Thus, ``bert2gpt2`` is consequently fined-tuned on the `CNN/Daily Mail`dataset and the resulting model
`bert2gpt2-cnn_dailymail-fp16` is uploaded here.

## Example

The model is by no means a state-of-the-art model, but nevertheless 
produces reasonable summarization results. It was mainly fine-tuned 
as a proof-of-concept for the 🤗 EncoderDecoder Framework.

The model can be used as follows:

```python
from transformers import BertTokenizer, GPT2Tokenizer, EncoderDecoderModel

model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2gpt2-cnn_dailymail-fp16")
# reuse tokenizer from bert2bert encoder-decoder model
bert_tokenizer = BertTokenizer.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

article = """(CNN)Sigma Alpha Epsilon is under fire for a video showing party-bound fraternity members singing a racist chant. SAE's national chapter suspended the students, but University of Oklahoma President David B
oren took it a step further, saying the university's affiliation with the fraternity is permanently done. The news is shocking, but it's not the first time SAE has faced controversy. SAE was founded March 9, 185
6, at the University of Alabama, five years before the American Civil War, according to the fraternity website. When the war began, the group had fewer than 400 members, of which "369 went to war for the Confede
rate States and seven for the Union Army," the website says. The fraternity now boasts more than 200,000 living alumni, along with about 15,000 undergraduates populating 219 chapters and 20 "colonies" seeking fu
ll membership at universities. SAE has had to work hard to change recently after a string of member deaths, many blamed on the hazing of new recruits, SAE national President Bradley Cohen wrote in a message on t
he fraternity's website. The fraternity's website lists more than 130 chapters cited or suspended for "health and safety incidents" since 2010. At least 30 of the incidents involved hazing, and dozens more invol
ved alcohol. However, the list is missing numerous incidents from recent months. Among them, according to various media outlets: Yale University banned the SAEs from campus activities last month after members al
legedly tried to interfere with a sexual misconduct investigation connected to an initiation rite. Stanford University in December suspended SAE housing privileges after finding sorority members attending a frat
ernity function were subjected to graphic sexual content. And Johns Hopkins University in November suspended the fraternity for underage drinking. "The media has labeled us as the 'nation's deadliest fraternity,
' " Cohen said. In 2011, for example, a student died while being coerced into excessive alcohol consumption, according to a lawsuit. SAE's previous insurer dumped the fraternity. "As a result, we are paying Lloy
d's of London the highest insurance rates in the Greek-letter world," Cohen said. Universities have turned down SAE's attempts to open new chapters, and the fraternity had to close 12 in 18 months over hazing in
cidents."""

input_ids = bert_tokenizer(article, return_tensors="pt").input_ids
output_ids = model.generate(input_ids)

# we need a gpt2 tokenizer for the output word embeddings
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

print(gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True))
# should produce
# SAE's national chapter suspended the students, but university president says it's permanent.
# The fraternity has had to deal with a string of incidents since 2010.
# SAE has more than 200,000 members, many of whom are students.
# A student died while being coerced into drinking alcohol. 
```

## Training script:

**IMPORTANT**: In order for this code to work, make sure you checkout to the branch 
[more_general_trainer_metric](https://github.com/huggingface/transformers/tree/more_general_trainer_metric), which slightly adapts 
the `Trainer` for `EncoderDecoderModels` according to this PR: https://github.com/huggingface/transformers/pull/5840. 

The following code shows the complete training script that was used to fine-tune `bert2gpt2-cnn_dailymail-fp16
` for reproducability. The training last ~11h on a standard GPU.

```python
#!/usr/bin/env python3
import nlp
import logging
from transformers import BertTokenizer, GPT2Tokenizer, EncoderDecoderModel, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO)

model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "gpt2")
# cache is currently not supported by EncoderDecoder framework
model.decoder.config.use_cache = False
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# CLS token will work as BOS token
bert_tokenizer.bos_token = bert_tokenizer.cls_token

# SEP token will work as EOS token
bert_tokenizer.eos_token = bert_tokenizer.sep_token


# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token


# set decoding params
model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
model.config.eos_token_id = gpt2_tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4

# load train and validation data
train_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="train")
val_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="validation[:5%]")

# load rouge for validation
rouge = nlp.load_metric("rouge", experiment_id=1)

encoder_length = 512
decoder_length = 128
batch_size = 16


# map data correctly
def map_to_encoder_decoder_inputs(batch):    # Tokenizer will automatically set [BOS] <text> [EOS] 
    # use bert tokenizer here for encoder
    inputs = bert_tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_length)
    # force summarization <= 128
    outputs = gpt2_tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    batch["decoder_attention_mask"] = outputs.attention_mask

    # complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
    batch["labels"] = [
        [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch["decoder_attention_mask"], batch["labels"])]
    ]

    assert all([len(x) == encoder_length for x in inputs.input_ids])
    assert all([len(x) == decoder_length for x in outputs.input_ids])

    return batch


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = gpt2_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = gpt2_tokenizer.eos_token_id
    label_str = gpt2_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


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
    fp16=True,
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

## Evaluation

The following script evaluates the model on the test set of 
CNN/Daily Mail.

```python
#!/usr/bin/env python3
import nlp
from transformers import BertTokenizer, GPT2Tokenizer, EncoderDecoderModel

model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2gpt2-cnn_dailymail-fp16")
model.to("cuda")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# CLS token will work as BOS token
bert_tokenizer.bos_token = bert_tokenizer.cls_token

# SEP token will work as EOS token
bert_tokenizer.eos_token = bert_tokenizer.sep_token


# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token


# set decoding params
model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
model.config.eos_token_id = gpt2_tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4

test_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="test")
batch_size = 64


# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = bert_tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = gpt2_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch


results = test_dataset.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

# load rouge for validation
rouge = nlp.load_metric("rouge")

pred_str = results["pred"]
label_str = results["highlights"]

rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

print(rouge_output)
```

The obtained results should be:

| -   |      Rouge2 - mid -precision      |  Rouge2 - mid - recall | Rouge2 - mid - fmeasure |
|----------|:-------------:|:------:|:------:|
| **CNN/Daily Mail** | 14.42 | 16.99 | **15.16** |
