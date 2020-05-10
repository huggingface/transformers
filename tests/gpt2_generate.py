import os
from time import time

import jax.numpy as jnp
import jax.random as rng
import numpy as np
import onnx
import onnxruntime as ort
import torch
from jax import jit, vmap
from jax.nn import softmax
from numpyro.distributions import MultinomialLogits
from torch.onnx import export
from tqdm import trange

from transformers import GPT2LMHeadModel, GPT2TokenizerFast


@jit
def nucleus_sampling(logits: np.ndarray, k: int = 50, p: float = 0.9, theta: float = 1.0):
    INF = np.finfo("f4").min

    # Sort descending
    sorted_indices = np.flip(jnp.argsort(logits, axis=-1), axis=-1)
    reverse_indices = jnp.argsort(sorted_indices, axis=-1)

    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)

    # Filter out top-k
    sorted_logits = jnp.where(sorted_logits < sorted_logits[k], logits, INF)

    # Compute the cumulative distribution from individual distribution
    candidates_dist = jnp.cumsum(softmax(sorted_logits / theta), axis=-1)

    # Filter out less probable candidates (Keep only p-% of the mass)
    sorted_logits = jnp.where(candidates_dist > p, sorted_logits, INF)

    # Sample
    filtered_logits = jnp.take_along_axis(sorted_logits, reverse_indices, axis=-1)
    return MultinomialLogits(filtered_logits / theta).sample(rng_key)


if __name__ == "__main__":
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    inputs = tokenizer.encode_plus("My name is", return_attention_mask=False, return_token_type_ids=False,)

    inputs = {k: torch.tensor([v], dtype=torch.int64) for k, v in inputs.items()}

    # Load ONNX model
    onnx_options = ort.SessionOptions()
    onnx_options.intra_op_num_threads = 1
    onnx_options.log_severity_level = 3
    onnx_options.log_verbosity_level = 3
    onnx_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    onnx_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    onnx_f = ort.InferenceSession("onnx/gpt2-large_ort_cpu.onnx", onnx_options, providers=["CPUExecutionProvider"])

    # Generate
    content = "And what is love?"
    content += (
        " It is a doll dressed up. For idleness to cosset, nurse, and dandle; A thing of soft misnomers, so divine"
    )

    content_generated = tokenizer.encode(content)
    content_tokens = np.array(content_generated, dtype="i8")

    # Ensure we have a batch axis
    content_tokens = np.atleast_2d(content_tokens)

    rng_seed = 0
    rng_key = rng.PRNGKey(rng_seed)

    for _ in trange(30):
        outputs = onnx_f.run(["word_probs"], {"input_ids": content_tokens})
        words_log_prob = outputs[0]

        # Nucleus sampling (Top-P) to get candidates
        next_word_token = nucleus_sampling(words_log_prob[:, -1], 50, 0.9, 0.7)
        next_word_token = jnp.atleast_2d(next_word_token[:, 0]).astype("i8")

        # Append the token to the next forward call
        content_tokens = np.ascontiguousarray(np.concatenate([content_tokens, next_word_token], axis=-1))

        # Append the token to the list of token generated so far
        content_generated.append(next_word_token.item())

    # Decode the generated tokens
    content_generated = tokenizer.decode(content_generated)

    print(content_generated)
