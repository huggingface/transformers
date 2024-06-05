import os
import time
import argparse

import torch
import torch._dynamo.config
import torch._inductor.config
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import QuantizedCacheConfig, HQQQuantizedCacheStatic

os.environ["TOKENIZERS_PARALLELISM"] = "0"

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision('high')

torch._logging.set_logs(graph_breaks=True, recompiles=True)
torch._dynamo.config.cache_size_limit = 256

CBOLD = '\033[1m'
CRED = '\033[91m'
CEND = '\033[0m'

def run_compile(model, tokenizer, max_new_tokens=20, verbose=True):
    prompts = [
        #"The sun dipped below the horizon, painting the sky in red.",
        "I almost missed the bus this morning, but luckily the driver saw me. Tomorrow",
    ]

    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    inputs_length = inputs.input_ids.shape[1]
    generate_kwargs = {
        "pad_token_id": tokenizer.pad_token_id,
        "min_new_tokens": max_new_tokens,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "output_logits": True,
        "return_dict_in_generate": True,
        }

    cache_config = QuantizedCacheConfig(
        backend="HQQ",
        nbits=4,
        axis_key=0,
        axis_value=0,
        compute_dtype=torch.float16,
        device=model.device
    )
    cache = HQQQuantizedCacheStatic(
        config=model.config, cache_config=cache_config, max_batch_size=inputs.input_ids.shape[0],
        max_cache_len=inputs.input_ids.shape[1] + max_new_tokens
    )
    model._cache = cache
    #model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    out = model.generate(
        **inputs, **generate_kwargs,
        past_key_values=cache,
        #cache_implementation="quantized",
        #cache_config={"backend": "HQQ", "axis_key": 1, "axis_value": 1, "nbits": 4, "residual_length": 1}
    )

    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end) / 1000)

    text = tokenizer.batch_decode(
        out.sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    if verbose:
        print(f"{CBOLD}Generated output:{CEND} {text}")
        print("-" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", type=str, default="fp16")

    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--verbose", action="store_false")

    args = parser.parse_args()

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype
    ).to("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=bool(args.trust_remote_code), padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    run_compile(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
