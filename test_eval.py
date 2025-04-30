"""
This script is used to test the SmolLM2-135M model.

Usage:
python test.py
# or using torchrun
torchrun --nproc_per_node=1 test.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
import torch.distributed as dist

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    # this is what we use to initialize torch.distributed
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Log distributed information
    logger.info(f"Distributed setup - Rank: {rank}, World size: {world_size}, Local rank: {local_rank}")
    
    # Load model and tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M"
    logger.info(f"Loading model and tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, tp_plan="auto")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Input text
    input_text = "Hello, my name is"
    logger.info(f"Input text: {input_text}")

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get logits
    logits = outputs.logits

    # Print shape and sample of logits
    logger.info(f"Logits shape: {logits.shape}")
    logger.info(f"Last token logits (first 10 values): {logits[0, -1, :10]}")

    # Get top 5 predictions for the next token
    next_token_logits = logits[0, -1, :]
    top_k_values, top_k_indices = torch.topk(next_token_logits, 5)

    logger.info("\nTop 5 next token predictions:")
    for i, (value, idx) in enumerate(zip(top_k_values.tolist(), top_k_indices.tolist())):
        token = tokenizer.decode([idx])
        logger.info(f"{i+1}. Token: '{token}', Score: {value:.4f}")
    
    # Clean up distributed environment
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed process group")

if __name__ == "__main__":
    main()
