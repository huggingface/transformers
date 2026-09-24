#!/usr/bin/env python
# coding=utf-8
import argparse, json, os, inspect
from dataclasses import dataclass
from typing import Dict, List
import numpy as np, torch
from datasets import load_dataset
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments, set_seed,
)

# Global matplotlib import (no reassignments later!)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

def sigmoid(x: np.ndarray) -> np.ndarray: return 1/(1+np.exp(-x))
def binarize_probs(p: np.ndarray, th: float) -> np.ndarray: return (p>=th).astype(np.int64)
def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str,float]:
    return {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
    }

@dataclass
class DatasetColumns:
    text: str
    labels: str

def build_one_hot_fn(n:int):
    def fn(ids:List[int])->List[float]:
        arr = np.zeros(n, dtype=np.float32)
        if ids is not None:
            for i in ids:
                if 0<=i<n: arr[i]=1.0
        return arr.tolist()
    return fn

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="prajjwal1/bert-tiny")
    p.add_argument("--dataset_name", type=str, default="go_emotions")
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--validation_split", type=str, default="validation")
    p.add_argument("--test_split", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")
    p.add_argument("--labels_column", type=str, default="labels")
    # training
    p.add_argument("--output_dir", type=str, default="./mlc_out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--do_predict", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--max_predict_samples", type=int, default=None)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--per_device_eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    # thresholds/plots
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--tune_thresholds", action="store_true")
    p.add_argument("--plot_threshold_curve", action="store_true")
    # inference
    p.add_argument("--predict_texts", type=str, nargs="*", default=None)
    args=p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("[STEP] started, output_dir:", args.output_dir, flush=True)
    set_seed(args.seed)

    print("[STEP] loading dataset…", flush=True)
    raw = load_dataset(args.dataset_name, args.dataset_config_name)
    if args.train_split not in raw: raise ValueError("missing train split")
    if args.do_eval and args.validation_split not in raw: raise ValueError("missing validation split")

    label_feature = raw[args.train_split].features[args.labels_column].feature
    label_names: List[str] = list(label_feature.names)
    num_labels=len(label_names)
    print(f"[STEP] dataset ok, num_labels={num_labels}", flush=True)

    cols = DatasetColumns(text=args.text_column, labels=args.labels_column)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    config.problem_type = "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    to_one_hot = build_one_hot_fn(num_labels)
    def preprocess(batch):
        enc = tokenizer(batch[cols.text], truncation=True, padding=False)
        enc["labels"] = [to_one_hot(ids) for ids in batch[cols.labels]]
        return enc
    def maybe_select(ds,n): return ds if n is None else ds.select(range(min(n,len(ds))))

    train_ds = maybe_select(raw[args.train_split], args.max_train_samples) if args.do_train else None
    eval_ds  = maybe_select(raw[args.validation_split], args.max_eval_samples) if args.do_eval else None

    if args.do_train: train_ds=train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    if args.do_eval:  eval_ds =eval_ds .map(preprocess, batched=True, remove_columns=eval_ds .column_names)

    # collator that casts labels to float32 for BCEWithLogitsLoss
    _base_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    def data_collator(features):
        batch = _base_collator(features)
        if "labels" in batch:
            batch["labels"] = batch["labels"].to(dtype=torch.float32)
        return batch

    base_threshold = args.threshold
    def compute_metrics(ev):
        logits, labels = ev
        probs = sigmoid(logits)
        preds = binarize_probs(probs, base_threshold)
        return multilabel_metrics(labels.astype(int), preds)

    # ---- TrainingArguments (v4/v5 compatible) ----
    print("[STEP] building trainer…", flush=True)
    strategy = "epoch" if args.do_eval else "no"
    ta_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        load_best_model_at_end=bool(args.do_eval),
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )
    sig = inspect.signature(TrainingArguments)
    if "eval_strategy" in sig.parameters:
        ta_kwargs["eval_strategy"] = strategy
    elif "evaluation_strategy" in sig.parameters:
        ta_kwargs["evaluation_strategy"] = strategy
    if "save_strategy" in sig.parameters:
        ta_kwargs["save_strategy"] = strategy
    training_args = TrainingArguments(**ta_kwargs)

    # ---- Trainer (v4/v5 compatible tokenizer kwarg) ----
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds if args.do_train else None,
        eval_dataset=eval_ds if args.do_eval else None,
        data_collator=data_collator,
    )
    if args.do_eval:
        trainer_kwargs["compute_metrics"] = compute_metrics
    if "tokenizer" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    results: Dict[str, Dict] = {}
    if args.do_train:
        print("[STEP] training…", flush=True)
        train_out = trainer.train()
        trainer.save_model()
        results["train"]=train_out.metrics

    if args.do_eval:
        print("[STEP] evaluating…", flush=True)
        eval_out = trainer.evaluate()
        results["eval_base_threshold"]=eval_out
        print("[STEP] eval done.", flush=True)

        if args.tune_thresholds:
            print("[STEP] sweeping thresholds…", flush=True)
            preds_output = trainer.predict(eval_ds)
            logits = preds_output.predictions
            labels = preds_output.label_ids.astype(int)
            probs = sigmoid(logits)
            ths = np.linspace(0.05,0.95,19)
            f1s=[]; best={"threshold":base_threshold,"f1_micro":-1.0,"metrics":None}
            for th in ths:
                mets = multilabel_metrics(labels, binarize_probs(probs, th))
                f1s.append(mets["f1_micro"])
                if mets["f1_micro"]>best["f1_micro"]:
                    best={"threshold":float(th),"f1_micro":mets["f1_micro"],"metrics":mets}
            results["threshold_tuning"]=best
            print(f"[STEP] best threshold: {best['threshold']:.2f} f1_micro={best['f1_micro']:.4f}", flush=True)

            if args.plot_threshold_curve and plt is not None:
                print("[STEP] plotting curve…", flush=True)
                os.makedirs(args.output_dir, exist_ok=True)
                plt.figure(figsize=(6,4))
                plt.plot(ths, f1s, marker="o")
                plt.xlabel("Threshold"); plt.ylabel("F1-micro")
                plt.title("Validation F1-micro vs Threshold"); plt.grid(True,alpha=0.3)
                plt.savefig(os.path.join(args.output_dir,"threshold_sweep.png"), dpi=160, bbox_inches="tight")
                print("[STEP] plot saved.", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir,"results_multilabel.json"),"w") as f:
        json.dump(results, f, indent=2)
    print("[STEP] results saved to", os.path.join(args.output_dir,"results_multilabel.json"), flush=True)

if __name__=="__main__":
    main()
