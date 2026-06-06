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
Small-scale fine-tuning benchmark for the zero-shot object detectors that support a training loss
(Grounding DINO, OWL-ViT, OWLv2, OmDet-Turbo).

Key choices (see PR discussion):
  * backbones are frozen by default (only the detection heads/encoder-decoder are trained) -> far less
    over-fitting on small datasets and much faster/cheaper training;
  * evaluation uses each processor's `post_process_*` with `target_sizes`, so predicted boxes are mapped
    back to the original image and labels come out as proper class indices (this is what fixes the AP);
  * Grounding DINO training labels come from the processor's COCO-annotation handling (correct box frame),
    the other models use manually built normalized boxes (their processors only square-resize, no padding).
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


MODEL_DEFAULTS = {
    "grounding_dino": "IDEA-Research/grounding-dino-tiny",
    "owlvit": "google/owlvit-base-patch32",
    "owlv2": "google/owlv2-base-patch16-ensemble",
    "omdet_turbo": "omlab/omdet-turbo-swin-tiny-hf",
}
# Substrings of parameter names to freeze (image + text backbones) per model.
FREEZE_SUBSTRINGS = {
    "grounding_dino": ["backbone"],  # matches model.backbone.* and model.text_backbone.*
    "owlvit": ["vision_model", "text_model"],
    "owlv2": ["vision_model", "text_model"],
    "omdet_turbo": ["vision_backbone", "language_backbone"],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_name(name):
    return name.replace("_", " ").strip().lower()


def hflip(image, bbox, width):
    """Horizontally flip a PIL image and a list of COCO [x, y, w, h] boxes."""
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    bbox = [[width - x - w, y, w, h] for (x, y, w, h) in bbox]
    return image, bbox


def coco_to_cxcywh_norm(bboxes, width, height):
    boxes = torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([(x + w / 2) / width, (y + h / 2) / height, w / width, h / height], dim=-1)


def coco_to_xyxy_abs(bboxes, width, height):
    boxes = torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def build_examples(split, max_samples, drop_first_class):
    """Materialize (image RGB, coco_bboxes, category_ids); drop the RF100 meta-class 0 and reindex."""
    examples = []
    for ex in split:
        o = ex["objects"]
        bbox, cats = [], []
        for b, c in zip(o["bbox"], o["category"]):
            if drop_first_class:
                if c == 0:
                    continue
                c = c - 1
            bbox.append([float(v) for v in b])
            cats.append(int(c))
        if not bbox:
            continue
        examples.append((ex["image"].convert("RGB"), bbox, cats))
        if len(examples) >= max_samples:
            break
    return examples


class Collator:
    def __init__(self, processor, model_type, categories, augment=False):
        self.processor = processor
        self.model_type = model_type
        self.categories = categories
        self.augment = augment
        self.text_prompt = ". ".join(categories) + "."

    def __call__(self, batch):
        images, bboxes, cats = [], [], []
        for img, bbox, cat in batch:
            w, h = img.size
            if self.augment and random.random() < 0.5:
                img, bbox = hflip(img, bbox, w)
            images.append(img)
            bboxes.append(bbox)
            cats.append(cat)

        sizes = [img.size for img in images]  # (w, h)

        # Manual normalized (cxcywh) labels work for all four: OWL-ViT/OWLv2/OmDet square-resize, and Grounding
        # DINO uses aspect-preserving resize with no padding at batch size 1, so normalized coords are invariant.
        labels = []
        for bbox, cat, (w, h) in zip(bboxes, cats, sizes):
            labels.append(
                {
                    "class_labels": torch.tensor(cat, dtype=torch.long),
                    "boxes": coco_to_cxcywh_norm(bbox, w, h),
                    "orig_size": torch.tensor([h, w]),
                }
            )
        if self.model_type == "grounding_dino":
            enc = self.processor(images=images, text=[self.text_prompt] * len(images), return_tensors="pt")
        elif self.model_type in ("owlvit", "owlv2"):
            enc = self.processor(images=images, text=[self.categories] * len(images), return_tensors="pt")
        else:  # omdet_turbo
            task = "Detect {}.".format(", ".join(self.categories))
            enc = self.processor(
                images=images, text=[self.categories] * len(images), task=[task] * len(images), return_tensors="pt"
            )
        return enc, labels


def freeze_backbones(model, model_type):
    frozen, total = 0, 0
    for name, p in model.named_parameters():
        total += p.numel()
        if any(s in name for s in FREEZE_SUBSTRINGS[model_type]):
            p.requires_grad_(False)
            frozen += p.numel()
    print(f"froze {frozen / 1e6:.1f}M / {total / 1e6:.1f}M params", flush=True)


@torch.no_grad()
def postprocess_predictions(model_type, outputs, labels, processor, enc, categories):
    """Use the model's own post-processing -> boxes in original-image xyxy + class-index labels."""
    target_sizes = torch.stack([lab["orig_size"] for lab in labels]).to(outputs_device(outputs))
    bs = len(labels)
    if model_type in ("owlvit", "owlv2"):
        results = processor.post_process_grounded_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)
        return [{"boxes": r["boxes"], "scores": r["scores"], "labels": r["labels"]} for r in results]
    if model_type == "omdet_turbo":
        results = processor.post_process_grounded_object_detection(
            outputs, text_labels=[categories] * bs, threshold=0.0, nms_threshold=0.5, target_sizes=target_sizes
        )
        return [{"boxes": r["boxes"], "scores": r["scores"], "labels": r["labels"]} for r in results]
    # grounding_dino: labels come back as text -> map robustly to class ids.
    # A non-zero text_threshold is required: with 0.0 every box decodes to the whole prompt and all
    # predictions collapse onto a single class.
    results = processor.post_process_grounded_object_detection(
        outputs, enc["input_ids"], threshold=0.05, text_threshold=0.25, target_sizes=target_sizes
    )
    cat_clean = [c.lower() for c in categories]
    preds = []
    for r in results:
        text_labels = r.get("text_labels") or r.get("labels")
        ids = []
        for t in text_labels:
            t = str(t).strip().lower()
            mid = next((i for i, c in enumerate(cat_clean) if t and (t == c or t in c or c in t)), 0)
            ids.append(mid)
        preds.append(
            {"boxes": r["boxes"], "scores": r["scores"], "labels": torch.tensor(ids, device=r["boxes"].device)}
        )
    return preds


def outputs_device(outputs):
    for v in outputs.values():
        if torch.is_tensor(v):
            return v.device
    return torch.device("cpu")


def targets_for_metric(labels):
    tgts = []
    for lab in labels:
        h, w = lab["orig_size"].tolist()
        cx, cy, bw, bh = lab["boxes"].unbind(-1)
        boxes = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], dim=-1)
        tgts.append({"boxes": boxes, "labels": lab["class_labels"]})
    return tgts


def move_to_device(enc, labels, device):
    enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    labels = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in lab.items()} for lab in labels]
    return enc, labels


def render_sample(model, processor, model_type, image, bbox, cat, categories, device, score_threshold=0.3):
    model.eval()
    collator = Collator(processor, model_type, categories, augment=False)
    enc, labels = collator([(image, bbox, cat)])
    enc, labels = move_to_device(enc, labels, device)
    with torch.no_grad():
        out = model(**enc)
    pred = postprocess_predictions(model_type, out, labels, processor, enc, categories)[0]
    pred = {k: v.cpu() for k, v in pred.items()}
    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    keep = pred["scores"] >= score_threshold
    for box, score, lab in zip(pred["boxes"][keep], pred["scores"][keep], pred["labels"][keep]):
        x0, y0, x1, y1 = [float(v) for v in box]
        draw.rectangle((x0, y0, x1, y1), outline="red", width=3)
        name = categories[int(lab)] if int(lab) < len(categories) else str(int(lab))
        draw.text((x0, max(0, y0 - 10)), f"{name}:{float(score):.2f}", fill="red")
    return drawn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=list(MODEL_DEFAULTS))
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--dataset_name", default="Francesco/excavators-czvg9")
    parser.add_argument("--drop_first_class", action="store_true", help="Drop the RF100 meta-class at index 0.")
    parser.add_argument("--max_train_samples", type=int, default=400)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--freeze_backbones", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbones", dest="freeze_backbones", action="store_false")
    parser.add_argument("--num_sample_images", type=int, default=3)
    parser.add_argument("--output_dir", default="zsod-finetune")
    parser.add_argument("--push_to_hub_repo", default=None)
    parser.add_argument("--push_subdir", default=None, help="Subdir in the hub repo (defaults to model_type).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = args.model_name_or_path or MODEL_DEFAULTS[args.model_type]
    print(f"[{args.model_type}] model={model_id} device={device} freeze={args.freeze_backbones}", flush=True)

    dataset = load_dataset(args.dataset_name)
    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.15, seed=args.seed)
        dataset["train"], dataset["validation"] = split["train"], split["test"]
    raw_names = dataset["train"].features["objects"]["category"].feature.names
    categories = [clean_name(n) for n in (raw_names[1:] if args.drop_first_class else raw_names)]
    print(f"categories={categories}", flush=True)

    train_examples = build_examples(dataset["train"], args.max_train_samples, args.drop_first_class)
    eval_examples = build_examples(dataset["validation"], args.max_eval_samples, args.drop_first_class)
    print(f"train={len(train_examples)} eval={len(eval_examples)}", flush=True)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, ignore_mismatched_sizes=True).to(device)
    if args.freeze_backbones:
        freeze_backbones(model, args.model_type)

    sample = eval_examples[: args.num_sample_images]
    for i, (img, bbox, cat) in enumerate(sample):
        render_sample(model, processor, args.model_type, img, bbox, cat, categories, device).save(
            os.path.join(args.output_dir, f"sample_{i}_vanilla.png")
        )

    train_collator = Collator(processor, args.model_type, categories, augment=True)
    eval_collator = Collator(processor, args.model_type, categories, augment=False)
    train_loader = DataLoader(
        train_examples, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=train_collator
    )
    eval_loader = DataLoader(
        eval_examples, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=eval_collator
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-4)

    history = []
    for epoch in range(args.num_train_epochs):
        model.train()
        train_losses = []
        for enc, labels in train_loader:
            enc, labels = move_to_device(enc, labels, device)
            optimizer.zero_grad()
            out = model(**enc, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.1)
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
                    for p in postprocess_predictions(args.model_type, out, labels, processor, enc, categories)
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

    for i, (img, bbox, cat) in enumerate(sample):
        render_sample(model, processor, args.model_type, img, bbox, cat, categories, device).save(
            os.path.join(args.output_dir, f"sample_{i}_finetuned.png")
        )

    summary = {"model_type": args.model_type, "model_id": model_id, "dataset": args.dataset_name, "history": history}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.model_type}] wrote metrics.json", flush=True)

    if args.push_to_hub_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=args.output_dir,
            path_in_repo=args.push_subdir or args.model_type,
            repo_id=args.push_to_hub_repo,
            repo_type="dataset",
        )
        print(f"[{args.model_type}] uploaded artifacts", flush=True)


if __name__ == "__main__":
    main()
