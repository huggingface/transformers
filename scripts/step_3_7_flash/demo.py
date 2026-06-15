"""Demo: load mini Step-3.7-Flash and run a text-only forward pass with debug tracing.

Build the mini checkpoint first:
  PYTHONPATH=src python scripts/step_3_7_flash/build_mini_model.py

Then run this demo:
  PYTHONPATH=src python scripts/step_3_7_flash/demo.py

Debug traces are written to ./debug_step_3_7_flash/:
  - Step3p7ForConditionalGeneration_debug_tree_SUMMARY.json
  - Step3p7ForConditionalGeneration_debug_tree_FULL_TENSORS.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("./mini-step-3-7-flash"))
    parser.add_argument("--debug-path", type=str, default=str(Path(__file__).parent / "debug_output"))
    args = parser.parse_args()

    from transformers import model_addition_debugger_context
    from transformers.models.step_3_7_flash.configuration_step3p7 import Step3p7Config
    from transformers.models.step_3_7_flash.modeling_step3p7 import Step3p7ForConditionalGeneration

    print(f"Loading mini model from {args.model_dir} ...")
    cfg = Step3p7Config.from_pretrained(args.model_dir)
    model = Step3p7ForConditionalGeneration.from_pretrained(args.model_dir, config=cfg).eval()

    vocab_size = cfg.text_config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 16))

    print(f"input_ids shape: {input_ids.shape}")
    print(f"Writing debug trace to: {args.debug_path}")
    with model_addition_debugger_context(model, debug_path=args.debug_path, do_prune_layers=False):
        out = model.forward(input_ids=input_ids, use_cache=False)

    print(f"logits shape:    {out.logits.shape}")
    print("Forward pass OK.")


if __name__ == "__main__":
    main()
