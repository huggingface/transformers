import logging
import collections
from dict_to_obj import DictToObj
import numpy as np
import torch
import pandas as pd
import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig, set_seed

start = "<|startoftext|> "
sep = " <|sep|>"


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0 or 0 < max_sequence_length < length:
        length = max_sequence_length
    elif length < 0:
        length = MAX_LENGTH
    return length


def generate(args, tokenizer, model, prompt):
    args.length = adjust_length_to_model(
        args.length, max_sequence_length=model.config.max_position_embeddings
    )
    prompt_text = start + prompt + sep
    encoded_prompt = tokenizer.encode(
        prompt_text, add_special_tokens=False, return_tensors="pt"
    )
    encoded_prompt = encoded_prompt.to(args.device)

    input_ids = None if encoded_prompt.size()[-1] == 0 else encoded_prompt
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=args.length + len(encoded_prompt[0]),
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=1,
    )
    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()
        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]
        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text
            + text[
                len(
                    tokenizer.decode(
                        encoded_prompt[0], clean_up_tokenization_spaces=True
                    )
                ) :
            ]
        )
        generated_sequences.append(total_sequence)
        # print(total_sequence)
    return generated_sequences


# Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set max generation length
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

# Define model class
MODEL_CLASSES = {"gpt2": (GPT2LMHeadModel, GPT2Tokenizer)}

# Generation arguments
args = collections.defaultdict(
    model_type="gpt2",
    model_name_or_path="path/to/model",
    prompt="",
    length=512,
    stop_token="<|endoftext|>",
    temperature=1.0,
    repetition_penalty=1.0,
    k=0,
    p=0.90,  # use nucleus sampling
    seed=42,
    no_cuda=False,
    num_return_sequences=1,
    device=torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    ),
    n_gpu=torch.cuda.device_count(),
)

# Convert dict to object
args = DictToObj(args)

# Set seed
set_seed(args.seed)

# Load tokenizer and model
args.model_type = args.model_type.lower()
config_class = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=None)
model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config_class,
    cache_dir=None,
)
model = model_class.from_pretrained(args.model_name_or_path)
model.to(args.device)


# Generate
text = generate(args, tokenizer, model, "Language Models are Few-Shot Learners")
print(text)
