import random

import numpy as np
import torch
from datasets import load_dataset
from Levenshtein import distance as levenshtein_distance
from peft import LoraConfig

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Idefics2ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


DEVICE = "cuda:0"
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)
pad_token_id = processor.tokenizer.pad_token_id

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
    use_dora=False,
    init_lora_weights="gaussian",
)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)

model.add_adapter(lora_config)
model.enable_adapters()

train_dataset = load_dataset("nielsr/docvqa_1200_examples", split="train")
train_dataset = train_dataset.remove_columns(["id", "words", "bounding_boxes", "answer"])

eval_dataset = load_dataset("nielsr/docvqa_1200_examples", split="test")
eval_dataset = eval_dataset.remove_columns(["id", "words", "bounding_boxes", "answer"])


class DataCollatorForGeneration:
    def __init__(self, processor, eval_mode=False):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        self.eval_mode = eval_mode

    def __call__(self, examples):
        texts, texts_eval = [], []
        images = []
        for example in examples:
            image = example["image"]
            question = example["query"]["en"]
            answer = random.choice(example["answers"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            text_eval = processor.apply_chat_template([messages[0]], add_generation_prompt=True)
            texts.append(text.strip())
            texts_eval.append(text_eval.strip())
            images.append([image])

        # Make sure we have right padding in train and left padding for eval parts
        processor.tokenizer.padding_side = "right"
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        if self.eval_mode:
            processor.tokenizer.padding_side = "left"
            batch_eval = processor(text=texts_eval, images=images, return_tensors="pt", padding=True)
            batch["generation_input_ids"] = batch_eval["input_ids"]
            batch["generation_attention_mask"] = batch_eval["attention_mask"]

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch


def calculate_levenstein(prediction_dict):
    # unmask for correct detokenization, because preds are padded to max length with -100
    preds = prediction_dict.predictions
    preds[preds == -100] = pad_token_id
    lbls = prediction_dict.label_ids
    lbls[lbls == -100] = pad_token_id

    # Decode and do magic for metrics
    preds = processor.batch_decode(preds)
    lbls = processor.batch_decode(lbls)
    levenstein_avg = np.mean([levenshtein_distance(pred, lbl) for pred, lbl in zip(preds, lbls)])
    return {"eval_levenstein": levenstein_avg}


generation_config = model.generation_config
generation_config.max_length = 200  # generate no more than 200 tokens (it includes image tokens also)

training_args = TrainingArguments(
    max_steps=1000,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    output_dir="/raid/raushan/idefics-train",
    eval_strategy="steps",
    fp16=True,
    eval_steps=10,
    save_steps=10,
    remove_unused_columns=False,
    report_to="wandb",
    predict_with_generate=True,  # will generate in eval step so we can compute text-based metrics
    generation_config=generation_config,
    metric_for_best_model="levenstein",  # will save model with lowest levenstein
    greater_is_better=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForGeneration(processor),
    eval_data_collator=DataCollatorForGeneration(processor, eval_mode=True),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=calculate_levenstein,
)

trainer.train()  # will run train and eval on the model
