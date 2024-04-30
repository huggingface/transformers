"""
Fine-tune Idefics2 by tweaking the `Seq2SeqTrainer` class.

One can run the script using `CUDA_VISIBLE_DEVICES=3 python src/transformers/models/idefics2/fine_tune_idefics2.py`.
"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import Levenshtein
import numpy as np
import requests
import torch
from datasets import load_dataset
from peft import LoraConfig
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    Idefics2ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


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
        target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian",
    )
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
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
        _attn_implementation="flash_attention_2",  # Only available on A100 or H100
    ).to(DEVICE)


## Load dataset
dataset = load_dataset("naver-clova-ix/cord-v2")

## Create PyTorch dataset
added_tokens = []
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)


class CustomDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        split,
        sort_json_key: bool = True,
    ):
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
                        self.add_tokens([rf"<s_{k}>", rf"</s_{k}>"])
                    output += (
                        rf"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + rf"</s_{k}>"
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
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": ground_truth}]},
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(
            text=texts, images=images, return_tensors="pt", truncation=True, padding="max_length", max_length=200
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch


data_collator = MyDataCollator(processor)


## Define metrics


def normalized_levenshtein(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    distance = Levenshtein.distance(s1, s2)
    return distance / max(len_s1, len_s2)


def similarity_score(a_ij, o_q_i, tau=0.5):
    nl = normalized_levenshtein(a_ij, o_q_i)
    return 1 - nl if nl < tau else 0


def average_normalized_levenshtein_similarity(ground_truth, predicted_answers):
    assert len(ground_truth) == len(predicted_answers), "Length of ground_truth and predicted_answers must match."

    N = len(ground_truth)
    total_score = 0

    for i in range(N):
        a_i = ground_truth[i]
        o_q_i = predicted_answers[i]
        if o_q_i == "":
            print("Warning: Skipped an empty prediction.")
            max_score = 0
        else:
            max_score = max(similarity_score(a_ij, o_q_i) for a_ij in a_i)

        total_score += max_score

    return total_score / N


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, processor.tokenizer.pad_token_id)
    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

    print("Decoded predictions:", decoded_preds)
    print("Decoded labels:", decoded_labels)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    score = average_normalized_levenshtein_similarity(decoded_labels, decoded_preds)
    result = {"levenshtein": score}

    prediction_lens = [np.count_nonzero(pred != processor.tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# def compute_metrics(eval_preds):
#     # Get the predicted and ground truth token sequences
#     # These are lists as they have different shapes for each batch
#     # We explicitly pass `eval_do_concat_batches=False` to the trainer
#     # TODO we could also just pad the input_ids/labels in the data collator
#     preds, labels = eval_preds

#     final_preds = []
#     final_labels = []
#     for batch_pred, batch_label in zip(preds, labels):
#         if isinstance(batch_pred, tuple):
#             batch_pred = batch_pred[0]

#         # Decode the generated ids and labels
#         decoded_preds = processor.batch_decode(batch_pred, skip_special_tokens=True)
#         decoded_labels = processor.batch_decode(batch_label, skip_special_tokens=True)

#         # Some simple post-processing
#         decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

#         final_preds.extend(decoded_preds)
#         final_labels.extend(decoded_labels)

#     N = len(final_labels)
#     total_score = 0

#     for i in range(N):
#         a_i = final_labels[i]
#         o_q_i = final_preds[i]
#         if o_q_i == "":
#             print("Warning: Skipped an empty prediction.")
#             max_score = 0
#         else:
#             max_score = max(similarity_score(a_ij, o_q_i) for a_ij in a_i)

#         total_score += max_score

#     return {"levenshtein": total_score / N}


## Define Training Arguments and Trainer

generation_config = GenerationConfig.from_pretrained("HuggingFaceM4/idefics2-8b", max_new_tokens=500)

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
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    # evaluation_strategy="epoch",
    fp16=True,
    # push_to_hub_model_id="idefics2-8b-docvqa-finetuned-tutorial",
    remove_unused_columns=False,
    report_to="none",
    # eval_do_concat_batches=False,
    predict_with_generate=True,
    generation_config=generation_config,
)

# important: we need to disable caching during training
# otherwise the model generates past_key_values which is of type DynamicCache
model.config.use_cache = False


class Idefics2Trainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        # here we need to overwrite the input_ids to only include the prompt
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)

        # use dummy image
        # we can do this since each image is always turned into 64 image tokens
        url = "https://upload.wikimedia.org/wikipedia/commons/f/f3/Zinedine_Zidane_by_Tasnim_03.jpg"
        test_image = Image.open(requests.get(url, stream=True).raw)

        # prepare prompt for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract JSON."},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        processor_inputs = processor(text=prompt, images=[test_image], return_tensors="pt")
        custom_inputs = {}
        batch_size = generation_inputs["pixel_values"].shape[0]
        device = generation_inputs["pixel_values"].device
        custom_inputs["input_ids"] = processor_inputs.input_ids.repeat(batch_size, 1).to(
            device
        )  # repeat along batch dimension
        custom_inputs["attention_mask"] = processor_inputs.attention_mask.repeat(batch_size, 1).to(
            device
        )  # repeat along batch dimension
        custom_inputs["pixel_values"] = generation_inputs["pixel_values"]
        custom_inputs["pixel_attention_mask"] = generation_inputs["pixel_attention_mask"]

        # print("Custom inputs:")
        # for k,v in custom_inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k,v.shape)

        generated_tokens = self.model.generate(**custom_inputs, **gen_kwargs)

        # Strip the prompt from the generated_tokens
        generated_tokens = generated_tokens[:, custom_inputs["input_ids"].size(1) :]

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def _pad_tensors_to_max_len(self, tensor, max_length):
        pad_token_id = processor.tokenizer.pad_token_id

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


trainer = Idefics2Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
