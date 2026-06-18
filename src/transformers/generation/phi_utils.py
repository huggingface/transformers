import math
from typing import Optional, Tuple, List
import torch

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
PHI_SQ = PHI * PHI

def phi_temperature(base_temp: float = 0.7) -> float:
    return base_temp * PHI_INV + 1.0 * (1 - PHI_INV)

def phi_top_p() -> float:
    return 1.0 - PHI_INV * PHI_INV

def phi_top_k() -> int:
    return int(round(PHI * PHI * PHI * 10))

def phi_repetition_penalty() -> float:
    return 1.0 + PHI_INV

def phi_max_tokens() -> int:
    return int(round(PHI ** 8))


class PhiRecursiveGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.phi = PHI
        self.phi_inv = PHI_INV

    def generate(
        self,
        input_text: str,
        max_iterations: int = 3,
        verbose: bool = False,
        **kwargs
    ) -> str:
        inputs = self.tokenizer(input_text, return_tensors="pt")
        temp = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        top_k = kwargs.get('top_k', 40)

        best_output = None
        best_score = -float('inf')

        for i in range(max_iterations):
            temp_i = phi_temperature(temp)
            top_p_i = phi_top_p()
            top_k_i = phi_top_k()

            if verbose:
                print(f"Iteration {i+1}: temp={temp_i:.3f}, top_p={top_p_i:.3f}, top_k={top_k_i}")

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    temperature=temp_i,
                    top_p=top_p_i,
                    top_k=top_k_i,
                    do_sample=True,
                    max_new_tokens=phi_max_tokens(),
                    repetition_penalty=phi_repetition_penalty(),
                    **kwargs
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            score = self._evaluate_quality(generated_text)

            if score > best_score:
                best_score = score
                best_output = generated_text

            if verbose:
                print(f"  Score: {score:.3f}")
                print(f"  Output: {generated_text[:100]}...")

            temp = temp_i
            top_p = top_p_i
            top_k = top_k_i

        return best_output

    def _evaluate_quality(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        unique_ratio = len(set(words)) / len(words)
        length_score = 1.0 - abs(len(words) - 100) / 200
        length_score = max(0.0, min(1.0, length_score))
        return unique_ratio * 0.6 + length_score * 0.4


# DECLARATION: Expose the class at module level
__all__ = ['PhiRecursiveGenerator', 'phi_temperature', 'phi_top_p', 'phi_top_k', 'phi_repetition_penalty', 'phi_max_tokens']
