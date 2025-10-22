#!/usr/bin/env python3
"""
Example: Safe vs Regular Text Generation with Transformers Safety

This example demonstrates how to compare regular generation to generation with
real-time safety filtering (toxicity) using Transformers' safety utilities.

"""

import os
import platform
import sys
from pathlib import Path

import torch


# Add safe_generation to path to import BasicToxicityChecker
sys.path.insert(0, str(Path(__file__).parent / "safe_generation"))

from safe_generation import BasicToxicityChecker

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.safety import SafetyConfig, SafetyLogitsProcessor, SafetyStoppingCriteria


def get_device():
    if torch.backends.mps.is_available() and platform.system() == "Darwin":
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """Compare regular generation vs safety-filtered generation."""

    # Load a pre-trained model and tokenizer
    model_name = os.environ.get("SAFETY_EXAMPLE_MODEL", "Qwen/Qwen2.5-3B")
    print(f"Loading model: {model_name}")
    print("(Using Qwen for consistent text generation)")

    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build safety components with strict threshold
    toxicity_threshold = 0.7  # Strict threshold for demo
    safety_checker = BasicToxicityChecker(
        model_name="s-nlp/roberta_toxicity_classifier", threshold=toxicity_threshold, device="cpu"
    )

    # Create safety config from checker
    safety_config = SafetyConfig.from_checker(safety_checker)

    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import StoppingCriteriaList

    safety_processor = SafetyLogitsProcessor(
        safety_checker=safety_checker,
        tokenizer=tokenizer,
        safety_config=safety_config,
        check_interval=1,
    )
    safety_stopper = SafetyStoppingCriteria(
        safety_checker=safety_checker,
        tokenizer=tokenizer,
        safety_config=safety_config,
        check_final_only=False,
    )

    safety_processors = LogitsProcessorList([safety_processor])
    safety_criteria = StoppingCriteriaList([safety_stopper])

    # Base generation configuration - deterministic for comparison
    base_gen = GenerationConfig(
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompts = [
        'He went on a long tirade spouting insults and obscenities: "',
    ]

    print("\n" + "=" * 60)
    print("SAFE TEXT GENERATION EXAMPLES (Regular vs Safe)")
    print("=" * 60)
    print("(Errors will raise; no soft fallbacks)")

    for i, prompt in enumerate(prompts, 1):
        print(f"\nExample {i}: '{prompt}'")
        print("-" * 40)

        # Tokenize input and move to model device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 1) Regular generation (no safety)
        with torch.no_grad():
            outputs_regular = model.generate(
                **inputs,
                generation_config=base_gen,
            )
        text_regular = tokenizer.decode(outputs_regular[0], skip_special_tokens=True)
        cont_regular = text_regular[len(prompt) :].strip()
        print(f"Regular: {cont_regular}")

        # 2) Safe generation (with safety filtering)
        with torch.no_grad():
            outputs_safe = model.generate(
                **inputs,
                generation_config=base_gen,
                logits_processor=safety_processors,
                stopping_criteria=safety_criteria,
            )
        text_safe = tokenizer.decode(outputs_safe[0], skip_special_tokens=True)
        cont_safe = text_safe[len(prompt) :].strip()
        print(f"Safe:    {cont_safe}")

        if cont_regular != cont_safe:
            print("Safety filtering applied - outputs differ")
        else:
            print("No safety filtering needed - outputs identical")

            # Verify safety checker would detect issues in the output
            regular_safety_result = safety_checker.check_safety(text_regular)
            if not regular_safety_result.is_safe:
                print("   WARNING: Safety checker detected violations in output but filtering didn't occur!")
                print(f"   Violations: {[v.category for v in regular_safety_result.violations]}")
                print(f"   Confidence: {regular_safety_result.confidence:.3f}")

    print("\n" + "=" * 60)
    print("HOW IT WORKS:")
    print("=" * 60)
    print(
        """
1. SafetyLogitsProcessor blocks ALL tokens when unsafe content is detected
2. SafetyStoppingCriteria can halt generation if unsafe content is detected
3. Both work during generation, stopping output when safety violations occur
4. Deterministic generation allows direct comparison of safe vs regular outputs
        """
    )

    print("\nDifferent Safety Levels:")
    print("- strict:   threshold=0.5 (more restrictive)")
    print("- moderate: threshold=0.7 (balanced)")
    print("- lenient:  threshold=0.9 (less restrictive)")
    print("\nCurrent demo uses: threshold=0.7 for reliable blocking")
    print("\nTo use predefined presets:")
    print("from transformers.generation.safety import STRICT_PRESET")
    print("config = SafetyConfig.from_checker(checker, **STRICT_PRESET)")


if __name__ == "__main__":
    main()
