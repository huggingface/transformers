"""
Fine-tune Idefics2 using the `Seq2SeqTrainer` API.

One can run the script using `CUDA_VISIBLE_DEVICES=3 python src/transformers/models/idefics2/fine_tune_idefics2.py`.
"""

import json
import random
from typing import Any, List

import Levenshtein
import numpy as np

import torch
from torch.utils.data import Dataset
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from datasets import load_dataset

DEVICE = "cuda:0"
USE_LORA = False
USE_QLORA = True

## Load model

# Three options for training, from the lowest precision training to the highest precision training:
# - QLora
# - Standard Lora
# - Full fine-tuning
if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        quantization_config=bnb_config if USE_QLORA else None,
    )
    model.add_adapter(lora_config)
    model.enable_adapters()
else:
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2", # Only available on A100 or H100
    ).to(DEVICE)


## Load dataset
dataset = load_dataset("naver-clova-ix/cord-v2")

## Create PyTorch dataset
added_tokens = []
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)

class CustomDataset(Dataset):
  def __init__(self, hf_dataset, split, sort_json_key: bool = True,):
    self.dataset = hf_dataset[split]
    self.split = split
    self.sort_json_key = sort_json_key

    ground_truth_token_sequences = []
    for sample in self.dataset:
        ground_truth = json.loads(sample["ground_truth"])
        if "gt_parses" in ground_truth:  # some datasets have multiple ground truths available, e.g. DocVQA
            assert isinstance(ground_truth["gt_parses"], list)
            ground_truth_jsons = ground_truth["gt_parses"]
        else:
            assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
            ground_truth_jsons = [ground_truth["gt_parse"]]

        ground_truth_token_sequences.append(
            [
                self.json2token(
                    ground_truth_json,
                    update_special_tokens_for_json_key=self.split == "train",
                    sort_json_key=self.sort_json_key,
                )
                for ground_truth_json in ground_truth_jsons  # load json from list of json
            ]
        )

    self.ground_truth_token_sequences = ground_truth_token_sequences

  def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj
    
  def add_tokens(self, list_of_tokens: List[str]):
      """
      Add special tokens to tokenizer and resize the token embeddings of the decoder
      """
      newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
      if newly_added_num > 0:
          model.resize_token_embeddings(len(processor.tokenizer))
          added_tokens.extend(list_of_tokens)
  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):
    example = self.dataset[idx]
    image = example["image"]
    target_sequence = random.choice(self.ground_truth_token_sequences[idx])  # can be more than one, e.g., DocVQA

    return image, target_sequence
  

train_dataset = CustomDataset(hf_dataset=dataset, split="train")
eval_dataset = CustomDataset(hf_dataset=dataset, split="validation")

## Define data collator
class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image, ground_truth = example
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract JSON."},
                        {"type": "image"},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ground_truth}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch

data_collator = MyDataCollator(processor)


## Define Training Arguments and Trainer

def normalized_levenshtein(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    distance = Levenshtein.distance(s1, s2)
    return distance / max(len_s1, len_s2)

def similarity_score(a_ij, o_q_i, tau=0.5):
    nl = normalized_levenshtein(a_ij, o_q_i)
    return 1 - nl if nl < tau else 0

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    # Get the predicted and ground truth token sequences
    # These are lists as they have different shapes for each batch
    # We explicitly pass `eval_do_concat_batches=False` to the trainer
    # TODO we could also just pad the input_ids/labels in the data collator
    preds, labels = eval_preds

    final_preds = []
    final_labels = []
    for batch_pred, batch_label in zip(preds, labels):
        if isinstance(batch_pred, tuple):
            batch_pred = batch_pred[0]
    
        # Decode the generated ids and labels
        decoded_preds = processor.batch_decode(batch_pred, skip_special_tokens=True)
        decoded_labels = processor.batch_decode(batch_label, skip_special_tokens=True)

        print("Decoded predictions:", decoded_preds)
        print("Decoded labels:", decoded_labels)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        final_preds.extend(decoded_preds)
        final_labels.extend(decoded_labels)

    N = len(final_labels)
    total_score = 0

    for i in range(N):
        a_i = final_labels[i]
        o_q_i = final_preds[i]
        if o_q_i == "":
            print("Warning: Skipped an empty prediction.")
            max_score = 0
        else:
            max_score = max(similarity_score(a_ij, o_q_i) for a_ij in a_i)

        total_score += max_score

    return {"levenshtein": total_score / N}


training_args = Seq2SeqTrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    output_dir="idefics2_ft_tutorial",
    eval_strategy="steps",
    eval_steps=1,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    # evaluation_strategy="epoch",
    fp16=True,
    # push_to_hub_model_id="idefics2-8b-docvqa-finetuned-tutorial",
    remove_unused_columns=False,
    report_to="none",
    eval_do_concat_batches=False,
    predict_with_generate=True,
)

# important: we need to disable caching during training
# otherwise the model generates past_key_values which is of type DynamicCache
model.config.use_cache = False

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, # You can also evaluate (loss) on the eval set, note that it will incur some additional GPU memory,
    compute_metrics=compute_metrics,
)

trainer.train()