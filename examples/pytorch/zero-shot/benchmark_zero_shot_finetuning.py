#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Small-scale fine-tuning benchmark for the zero-shot object detectors that now support a training loss
(Grounding DINO, OWL-ViT, OWLv2, OmDet-Turbo).

For a single model it:
  * loads a (sub-sampled, <500 images) object detection dataset,
  * fine-tunes for a few epochs with a plain PyTorch loop,
  * logs train loss, validation loss and validation mAP per epoch,
  * renders sample inference images for the vanilla (pretrained) vs fine-tuned model,
  * writes `<output_dir>/metrics.json` + sample images, and optionally pushes everything to the Hub.

A manual loop (rather than `Trainer`) is used on purpose: the four architectures have very different
input signatures, and computing mAP directly from the raw outputs keeps the comparison uniform.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from PIL import ImageDraw
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


MODEL_DEFAULTS = {
    "grounding_dino": "IDEA-Research/grounding-dino-tiny",
    "owlvit": "google/owlvit-base-patch32",
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "omdet_turbo": "omlab/omdet-turbo-swin-tiny-hf",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def coco_to_cxcywh_norm(bboxes, width, height):
    boxes = torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
    x, y, w, h = boxes.unbind(-1)
    cx, cy = x + w / 2, y + h / 2
    return torch.stack([cx / width, cy / height, w / width, h / height], dim=-1)


def cxcywh_norm_to_xyxy_abs(boxes, width, height):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height], dim=-1
    )


def build_examples(dataset_split, max_samples):
    examples = []
    for ex in dataset_split:
        objects = ex["objects"]
        if len(objects["bbox"]) == 0:
            continue
        examples.append((ex["image"].convert("RGB"), objects))
        if len(examples) >= max_samples:
            break
    return examples


class Collator:
    def __init__(self, processor, model_type, categories):
        self.processor = processor
        self.model_type = model_type
        self.categories = categories
        self.text_prompt = ". ".join(categories) + "."

    def __call__(self, batch):
        images = [img for img, _ in batch]
        labels = []
        for img, objects in batch:
            w, h = img.size
            labels.append(
                {
                    "class_labels": torch.tensor(objects["category"], dtype=torch.long),
                    "boxes": coco_to_cxcywh_norm(objects["bbox"], w, h),
                    "orig_size": torch.tensor([h, w]),
                }
            )
        if self.model_type == "grounding_dino":
            enc = self.processor(images=images, text=[self.text_prompt] * len(images), return_tensors="pt")
        elif self.model_type in ("owlvit", "owlv2"):
            enc = self.processor(images=images, text=[self.categories] * len(images), return_tensors="pt")
        elif self.model_type == "omdet_turbo":
            task = "Detect {}.".format(", ".join(self.categories))
            enc = self.processor(
                images=images, text=[self.categories] * len(images), task=[task] * len(images), return_tensors="pt"
            )
        else:
            raise ValueError(self.model_type)
        return enc, labels


def predictions_from_outputs(model_type, outputs, labels, processor, enc, cat_to_id):
    """Convert raw model outputs to torchmetrics format: list of {boxes(xyxy abs), scores, labels}."""
    preds = []
    if model_type in ("owlvit", "owlv2", "omdet_turbo"):
        if model_type == "omdet_turbo":
            logits, boxes = outputs.decoder_class_logits, outputs.decoder_coord_logits
        else:
            logits, boxes = outputs.logits, outputs.pred_boxes
        scores, label_ids = logits.sigmoid().max(dim=-1)
        for i, lab in enumerate(labels):
            h, w = lab["orig_size"].tolist()
            preds.append(
                {"boxes": cxcywh_norm_to_xyxy_abs(boxes[i], w, h), "scores": scores[i], "labels": label_ids[i]}
            )
    elif model_type == "grounding_dino":
        target_sizes = torch.stack([lab["orig_size"] for lab in labels]).to(outputs.logits.device)
        results = processor.post_process_grounded_object_detection(
            outputs, enc["input_ids"], threshold=0.0, text_threshold=0.0, target_sizes=target_sizes
        )
        for res in results:
            text_labels = res.get("text_labels") or res.get("labels")
            device = res["boxes"].device
            if len(text_labels) and isinstance(text_labels[0], str):
                label_ids = torch.tensor([cat_to_id.get(t, 0) for t in text_labels], dtype=torch.long, device=device)
            else:
                label_ids = torch.as_tensor(text_labels, dtype=torch.long, device=device)
            preds.append({"boxes": res["boxes"], "scores": res["scores"], "labels": label_ids})
    return preds


def targets_for_metric(labels):
    tgts = []
    for lab in labels:
        h, w = lab["orig_size"].tolist()
        tgts.append({"boxes": cxcywh_norm_to_xyxy_abs(lab["boxes"], w, h), "labels": lab["class_labels"]})
    return tgts


def move_to_device(enc, labels, device):
    enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    labels = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in lab.items()} for lab in labels]
    return enc, labels


def render_sample(model, processor, model_type, image, categories, cat_to_id, device, score_threshold=0.3):
    """Run inference on one PIL image and draw boxes above the score threshold; returns a copy with drawings."""
    model.eval()
    collator = Collator(processor, model_type, categories)
    enc, labels = collator([(image, {"bbox": [[0, 0, 1, 1]], "category": [0]})])
    enc, labels = move_to_device(enc, labels, device)
    with torch.no_grad():
        out = model(**enc, labels=None) if model_type != "grounding_dino" else model(**enc)
    preds = predictions_from_outputs(model_type, out, labels, processor, enc, cat_to_id)[0]
    preds = {k: v.cpu() for k, v in preds.items()}
    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    keep = preds["scores"] >= score_threshold
    for box, score, lab in zip(preds["boxes"][keep], preds["scores"][keep], preds["labels"][keep]):
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle((x0, y0, x1, y1), outline="red", width=3)
        name = categories[int(lab)] if int(lab) < len(categories) else str(int(lab))
        draw.text((x0, max(0, y0 - 10)), f"{name}:{float(score):.2f}", fill="red")
    return drawn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=list(MODEL_DEFAULTS))
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--dataset_name", default="rishitdagli/cppe-5")
    parser.add_argument("--max_train_samples", type=int, default=400)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_sample_images", type=int, default=3)
    parser.add_argument("--output_dir", default="zsod-finetune")
    parser.add_argument("--push_to_hub_repo", default=None, help="Optional HF dataset repo id to upload artifacts to.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_name_or_path or MODEL_DEFAULTS[args.model_type]
    print(f"[{args.model_type}] model={model_id} device={device}", flush=True)

    dataset = load_dataset(args.dataset_name)
    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.15, seed=args.seed)
        dataset["train"], dataset["validation"] = split["train"], split["test"]
    categories = dataset["train"].features["objects"]["category"].feature.names
    cat_to_id = {c: i for i, c in enumerate(categories)}

    train_examples = build_examples(dataset["train"], args.max_train_samples)
    eval_examples = build_examples(dataset["validation"], args.max_eval_samples)
    print(f"train={len(train_examples)} eval={len(eval_examples)} num_classes={len(categories)}", flush=True)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, ignore_mismatched_sizes=True).to(device)

    # Vanilla (pretrained, before fine-tuning) sample inference, for the before/after comparison.
    sample_images = [img for img, _ in eval_examples[: args.num_sample_images]]
    for i, img in enumerate(sample_images):
        render_sample(model, processor, args.model_type, img, categories, cat_to_id, device).save(
            os.path.join(args.output_dir, f"sample_{i}_vanilla.png")
        )

    collator = Collator(processor, args.model_type, categories)
    train_loader = DataLoader(
        train_examples, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collator
    )
    eval_loader = DataLoader(
        eval_examples, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=collator
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    history = []
    for epoch in range(args.num_train_epochs):
        model.train()
        train_losses = []
        for enc, labels in train_loader:
            enc, labels = move_to_device(enc, labels, device)
            optimizer.zero_grad()
            out = model(**enc, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            train_losses.append(float(out.loss.detach()))

        model.eval()
        eval_losses = []
        metric = MeanAveragePrecision(box_format="xyxy")
        with torch.no_grad():
            for enc, labels in eval_loader:
                enc, labels = move_to_device(enc, labels, device)
                out = model(**enc, labels=labels)
                if out.loss is not None:
                    eval_losses.append(float(out.loss.detach()))
                preds = [
                    {k: v.cpu() for k, v in p.items()}
                    for p in predictions_from_outputs(args.model_type, out, labels, processor, enc, cat_to_id)
                ]
                tgts = [{k: v.cpu() for k, v in t.items()} for t in targets_for_metric(labels)]
                metric.update(preds, tgts)

        m = metric.compute()
        row = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(train_losses)), 4),
            "eval_loss": round(float(np.mean(eval_losses)), 4) if eval_losses else None,
            "eval_map": round(float(m["map"]), 4),
            "eval_map_50": round(float(m["map_50"]), 4),
        }
        history.append(row)
        print(f"[{args.model_type}] {row}", flush=True)

    # Fine-tuned sample inference, for the before/after comparison.
    for i, img in enumerate(sample_images):
        render_sample(model, processor, args.model_type, img, categories, cat_to_id, device).save(
            os.path.join(args.output_dir, f"sample_{i}_finetuned.png")
        )

    summary = {"model_type": args.model_type, "model_id": model_id, "history": history}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.model_type}] wrote {args.output_dir}/metrics.json", flush=True)

    if args.push_to_hub_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=args.output_dir,
            path_in_repo=args.model_type,
            repo_id=args.push_to_hub_repo,
            repo_type="dataset",
        )
        print(f"[{args.model_type}] uploaded artifacts to {args.push_to_hub_repo}/{args.model_type}", flush=True)


if __name__ == "__main__":
    main()
