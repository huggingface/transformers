from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments


raw_datasets = load_dataset("kde4", lang1="en", lang2="fr", trust_remote_code=True)
split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)

split_datasets["validation"] = split_datasets.pop("test")
split_datasets["train"][1]["translation"]

split_datasets["validation"] = split_datasets["validation"].select(range(256))


model_checkpoint = "google-t5/t5-base" # 11135332352  124439808  783150080 222903552
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

print(f"Param count: {sum(p.numel() for p in model.parameters())}")


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

en_sentence = split_datasets["train"][1]["translation"]["en"]
fr_sentence = split_datasets["train"][1]["translation"]["fr"]

inputs = tokenizer(en_sentence, text_target=fr_sentence)


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=50, truncation=True
    )
    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)


from transformers import DataCollatorForSeq2Seq
import evaluate

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
batch.keys()

metric = evaluate.load("sacrebleu")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


args = Seq2SeqTrainingArguments(
    f"marian-finetuned-kde4-en-to-fr",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
    generation_max_length=250,
)

from transformers import Seq2SeqTrainer

model.generation_config.max_new_tokens=200

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.evaluate()
