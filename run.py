from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig,TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig
import torch
import time
from torchmetrics.text import SQuAD
from random import randrange
from transformers.utils import logging
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# timdettmers/openassistant-guanaco
# Stanford/web_questions
eval_dataset = load_dataset("timdettmers/openassistant-guanaco", split="test")


# train_dataset=dataset["train"]
# eval_dataset=dataset["test"]

# eval_dataset = eval_dataset.select(range(256))

quant_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_type=torch.bfloat16
)

#model_id = "meta-llama/Llama-2-7b-chat-hf"
model_id="openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    #quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    #attn_implementation="flash_attention_2",
)

print(f"Param count: {sum(p.numel() for p in model.parameters())}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.add_special_tokens({"pad_token":"</s>"})
pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)


gen_config = model.generation_config
gen_config.max_new_tokens = 200
gen_config.use_cache = True


peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model.add_adapter(peft_config)
model.enable_adapters()


class DataCollatorForGenerationSFT:
    def __init__(self, tokenizer, eval_mode=False):
        self.tokenizer = tokenizer
        self.eval_mode = eval_mode

    def __call__(self, examples):
        texts, texts_eval = [], []
        for example in examples:
            question = example["text"]
            text_eval = question.split("### Assistant")[0]
            texts.append(question.strip())
            texts_eval.append(f"{text_eval.strip()}### Assistant:")

        # Make sure we have right padding in train and left padding for eval parts
        tokenizer.padding_side = "right"
        batch = tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        if self.eval_mode:
            tokenizer.padding_side = "left"
            batch_eval = tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            batch['generation_input_ids'] = batch_eval['input_ids']
            batch['generation_attention_mask'] = batch_eval['attention_mask']
        labels = batch["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100 # Ignore index for CE-loss
        batch["labels"] = labels
        return batch


def custom_metrics(prediction_dict):
    # unmask for correct detokenization, because preds are padded to max length with -100
    preds = prediction_dict.predictions
    preds[preds == -100] = pad_token_id
    lbls = prediction_dict.label_ids
    lbls[lbls == -100] = pad_token_id

    # Decode and do magic for metrics
    preds = tokenizer.batch_decode(preds,skip_special_tokens=True)
    lbls = tokenizer.batch_decode(lbls,skip_special_tokens=True)
    return {"exact_match" : 0, "f1_score": 0}


training_args = TrainingArguments(
    per_device_train_batch_size=8,
    #per_device_eval_batch_size=128,
    num_train_epochs=20,
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500000,
    bf16=True,
    output_dir="test_predict",
    overwrite_output_dir=True,
    optim="adafactor",
    report_to="wandb",
    logging_steps=100000,
    remove_unused_columns=False,
    predict_with_generate=True,
    generation_config=gen_config
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForGenerationSFT(tokenizer),
    eval_data_collator=DataCollatorForGenerationSFT(tokenizer, eval_mode=True),
    train_dataset=eval_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=custom_metrics,
)

trainer.evaluate()
