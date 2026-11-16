"""Export Llama to an ExecuTorch XNNPACK `.pte` with `ExecutorchExporter.export_for_generation` and
`GenerationConfig(use_cache_buffer=True)`, then decode with the `.pte`.

The static cache is kept as the model's buffers, so the exported decode step takes no cache and returns logits only;
the cache updates are `BUFFER_MUTATION`s held inside the program.

    python export_llama_cache_buffer.py --max_seq_len 128
"""

import argparse
from collections import Counter

import torch
from executorch.runtime import Runtime

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.exporters import ExecutorchConfig, ExecutorchExporter


def main(args):
    layer_override = {} if args.num_layers is None else {"num_hidden_layers": args.num_layers}
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype="float32", **layer_override).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    inputs = tokenizer(args.prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0].tolist()
    generation_config = GenerationConfig(
        cache_implementation="static", max_cache_len=args.max_seq_len, use_cache_buffer=True
    )

    with torch.no_grad():
        eager_tokens = model.generate(
            **inputs, generation_config=generation_config, max_new_tokens=args.max_seq_len - len(prompt_ids)
        )[0].tolist()
        program = ExecutorchExporter().export_for_generation(
            model, dict(inputs), ExecutorchConfig(backend="xnnpack"), generation_config=generation_config
        )["decode"]
    signature = program.exported_program().graph_signature
    input_names = [spec.arg.name for spec in signature.input_specs if spec.kind.name == "USER_INPUT"]
    print("user inputs:", input_names)
    print("outputs:", dict(Counter(spec.kind.name for spec in signature.output_specs)))
    with open(args.pte_path, "wb") as f:
        program.write_to_file(f)
    print(f"wrote {args.pte_path}")

    method = Runtime.get().load_program(args.pte_path).load_method("forward")
    slots = torch.arange(args.max_seq_len)
    tokens = list(prompt_ids)
    next_token = tokens[0]
    for position in range(args.max_seq_len):
        step_inputs = {
            "input_ids": torch.tensor([[next_token]]),
            "attention_mask": (slots <= position)[None, None, None],
            "position_ids": torch.tensor([[position]]),
        }
        logits = method.execute([step_inputs[name] for name in input_names])[0]
        next_token = logits[0, -1].argmax().item()
        if position + 1 < len(prompt_ids):
            next_token = prompt_ids[position + 1]
        elif position + 1 < args.max_seq_len:
            tokens.append(next_token)
    print(repr(tokenizer.decode(tokens)))
    print("matches eager generate():", tokens == eager_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--prompt", default="Simply put, the theory of relativity states that")
    parser.add_argument("--pte_path", default="llama_cache_buffer.pte")
    parser.add_argument("--num_layers", type=int, default=None, help="Override num_hidden_layers for quick runs.")
    main(parser.parse_args())
