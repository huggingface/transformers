#!/usr/bin/env python
# coding=utf-8
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
Simple Sentiment Analysis Example for Beginners

This is a beginner-friendly introduction to text classification using transformers.
It demonstrates the basic workflow of fine-tuning a pre-trained model on the IMDB
movie review dataset for binary sentiment classification.

This script is intentionally simpler than run_glue.py and run_classification.py
to serve as an educational entry point for those new to transformers.

Key Learning Points:
- Loading and preprocessing datasets
- Using pre-trained models for sequence classification
- Fine-tuning with the Trainer API
- Evaluating model performance
- Making predictions on new text

Requirements:
    pip install transformers datasets torch scikit-learn

Usage:
    python run_simple_sentiment.py
    
    # For smaller/faster demo:
    python run_simple_sentiment.py --max_train_samples 1000 --max_eval_samples 200
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple sentiment analysis example using IMDB dataset"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="distilbert-base-uncased",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization",
    )
    
    # Data arguments
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit the number of training samples (useful for quick testing)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit the number of evaluation samples (useful for quick testing)",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./imdb_sentiment_output",
        help="Output directory for model checkpoints and predictions",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size per device during training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size per device during evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def compute_metrics(eval_pred):
    """
    Compute accuracy and F1 score for evaluation.
    
    Args:
        eval_pred: EvalPrediction object containing predictions and labels
        
    Returns:
        Dictionary with computed metrics
    """
    from sklearn.metrics import accuracy_score, f1_score
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    
    return {
        "accuracy": accuracy,
        "f1": f1,
    }


def preprocess_function(examples, tokenizer, max_length):
    """
    Tokenize the text data.
    
    Args:
        examples: Batch of examples containing 'text' field
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def main():
    """Main training and evaluation function."""
    
    # Parse arguments
    args = parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("=" * 80)
    logger.info("Simple Sentiment Analysis with Transformers")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Step 1: Load dataset
    logger.info("\n Step 1: Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    
    # Optionally limit dataset size for faster experimentation
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))
        logger.info(f"   Limited training samples to {args.max_train_samples}")
    
    if args.max_eval_samples:
        dataset["test"] = dataset["test"].select(range(args.max_eval_samples))
        logger.info(f"   Limited test samples to {args.max_eval_samples}")
    
    logger.info(f"   Training samples: {len(dataset['train'])}")
    logger.info(f"   Test samples: {len(dataset['test'])}")
    
    # Step 2: Load tokenizer and model
    logger.info(f"\n Step 2: Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,  # Binary classification: positive/negative
    )
    logger.info(f"    Loaded {args.model_name_or_path}")
    
    # Step 3: Tokenize dataset
    logger.info("\n Step 3: Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        desc="Tokenizing",
    )
    logger.info("    Tokenization complete")
    
    # Step 4: Setup training
    logger.info("\n  Step 4: Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=100,
        seed=args.seed,
        report_to="none",  # Disable wandb/tensorboard for simplicity
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    logger.info("    Trainer initialized")
    
    # Step 5: Train model
    logger.info("\n Step 5: Training model...")
    logger.info(f"   Training for {args.num_train_epochs} epochs")
    train_result = trainer.train()
    
    # Save model
    trainer.save_model()
    logger.info(f"    Model saved to {args.output_dir}")
    
    # Step 6: Evaluate
    logger.info("\n Step 6: Evaluating model...")
    metrics = trainer.evaluate()
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    for key, value in metrics.items():
        logger.info(f"   {key}: {value:.4f}")
    
    # Step 7: Example predictions
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE PREDICTIONS")
    logger.info("=" * 80)
    
    example_texts = [
        "This movie was absolutely fantastic! Best film I've seen all year.",
        "Terrible waste of time. I want my money back.",
        "An okay movie, nothing special but not terrible either.",
    ]
    
    for text in example_texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=args.max_length,
        )
        
        # Move to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probs[0][prediction].item()
        
        logger.info(f"\nText: {text}")
        logger.info(f"Prediction: {sentiment} (confidence: {confidence:.2%})")
    
    logger.info("\n" + "=" * 80)
    logger.info(" Training and evaluation completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()