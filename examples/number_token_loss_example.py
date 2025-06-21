#!/usr/bin/env python3
"""
Example script demonstrating the use of Number Token Loss (NTL) in transformers.

This script shows how to:
1. Use NTL-WAS and NTL-MSE losses with a language model
2. Compare the performance with standard cross-entropy loss
3. Demonstrate the benefits on numerical tasks
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.loss import ForCausalLMWithNTLWAS, ForCausalLMWithNTLMSE
from datasets import Dataset
import numpy as np


def create_math_dataset(num_examples=1000):
    """
    Create a simple math dataset for demonstration.
    
    Args:
        num_examples: Number of examples to generate
        
    Returns:
        Dataset with math problems and solutions
    """
    examples = []
    
    for i in range(num_examples):
        # Generate simple arithmetic problems
        a = np.random.randint(1, 100)
        b = np.random.randint(1, 100)
        operation = np.random.choice(['+', '-', '*'])
        
        if operation == '+':
            result = a + b
        elif operation == '-':
            result = a - b
        else:  # '*'
            result = a * b
        
        # Create the problem text
        problem = f"What is {a} {operation} {b}? The answer is {result}."
        examples.append({"text": problem})
    
    return Dataset.from_list(examples)


def custom_loss_function(loss_name, tokenizer=None, alpha=0.1):
    """
    Create a custom loss function based on the specified loss type.
    
    Args:
        loss_name: Type of loss ('ce', 'ntl_was', 'ntl_mse')
        tokenizer: Tokenizer for NTL losses
        alpha: Weight for NTL loss component
        
    Returns:
        Loss function
    """
    if loss_name == 'ce':
        # Standard cross-entropy loss
        return nn.CrossEntropyLoss()
    elif loss_name == 'ntl_was':
        # NTL with Wasserstein-1 distance
        def ntl_was_loss(logits, labels):
            return ForCausalLMWithNTLWAS(
                logits, labels, 
                vocab_size=tokenizer.vocab_size,
                tokenizer=tokenizer,
                alpha=alpha
            )
        return ntl_was_loss
    elif loss_name == 'ntl_mse':
        # NTL with MSE
        def ntl_mse_loss(logits, labels):
            return ForCausalLMWithNTLMSE(
                logits, labels,
                vocab_size=tokenizer.vocab_size,
                tokenizer=tokenizer,
                alpha=alpha
            )
        return ntl_mse_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_name}")


class CustomTrainer(Trainer):
    """Custom trainer that supports different loss functions."""
    
    def __init__(self, loss_function, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss using the custom loss function.
        """
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get labels from inputs
        labels = inputs.get("labels")
        if labels is None:
            # If no labels provided, use input_ids shifted by 1
            labels = inputs["input_ids"].clone()
            labels[:, :-1] = inputs["input_ids"][:, 1:]
            labels[:, -1] = -100  # Ignore last token
        
        # Compute loss using custom loss function
        if isinstance(self.loss_function, nn.Module):
            # Standard loss function (e.g., CrossEntropyLoss)
            loss = self.loss_function(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            # Custom loss function (e.g., NTL)
            loss = self.loss_function(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def main():
    """Main function demonstrating Number Token Loss usage."""
    print("Number Token Loss (NTL) Example")
    print("=" * 50)
    
    # Load tokenizer and model
    model_name = "gpt2"  # Using GPT-2 for demonstration
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print("Creating math dataset...")
    dataset = create_math_dataset(num_examples=500)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./ntl_example_output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=100,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=5e-5,
        warmup_steps=100,
        remove_unused_columns=False,
    )
    
    # Test different loss functions
    loss_functions = {
        'Cross-Entropy': custom_loss_function('ce'),
        'NTL-WAS': custom_loss_function('ntl_was', tokenizer, alpha=0.1),
        'NTL-MSE': custom_loss_function('ntl_mse', tokenizer, alpha=0.1),
    }
    
    results = {}
    
    for loss_name, loss_function in loss_functions.items():
        print(f"\nTraining with {loss_name} loss...")
        
        # Create trainer with custom loss
        trainer = CustomTrainer(
            loss_function=loss_function,
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Evaluate on a simple test
        test_text = "What is 15 + 27? The answer is"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # Get logits for last position
            probs = torch.softmax(logits, dim=-1)
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probs, 5)
            
            print(f"\nTop 5 predictions for '{test_text}':")
            for i in range(5):
                token = tokenizer.decode([top_indices[0][i]])
                prob = top_probs[0][i].item()
                print(f"  {token}: {prob:.4f}")
        
        results[loss_name] = {
            'model': model,
            'final_loss': trainer.state.log_history[-1]['train_loss'] if trainer.state.log_history else None
        }
    
    # Print summary
    print("\n" + "=" * 50)
    print("Training Summary:")
    print("=" * 50)
    for loss_name, result in results.items():
        print(f"{loss_name}: Final Loss = {result['final_loss']:.4f}")
    
    print("\nNote: This is a demonstration. For real applications:")
    print("- Use larger models and datasets")
    print("- Tune hyperparameters (alpha, learning rate, etc.)")
    print("- Evaluate on proper test sets")
    print("- Consider the computational cost of NTL")


if __name__ == "__main__":
    main() 