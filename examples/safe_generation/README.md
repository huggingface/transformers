# Safe Generation Example Implementations

This directory contains reference implementations of safety checkers for the transformers safe generation feature.

## Overview

The core transformers library provides **infrastructure only**:
- `SafetyChecker` abstract base class
- `SafetyLogitsProcessor` and `SafetyStoppingCriteria`
- `SafetyConfig` configuration system
- `SafetyResult` and `SafetyViolation` data structures

**Concrete implementations** like `BasicToxicityChecker` are provided here as examples.

This follows the same pattern as watermarking in transformers - the core provides infrastructure, users provide or choose implementations.

## Usage

### Basic Usage with Pipeline

```python
from examples.safe_generation import BasicToxicityChecker
from transformers import pipeline
from transformers.generation.safety import SafetyConfig

# Create a safety checker
checker = BasicToxicityChecker(threshold=0.7)

# Option 1: Use with SafetyConfig
config = SafetyConfig.from_checker(checker)
pipe = pipeline("text-generation", model="gpt2", safety_config=config)

# Option 2: Direct generation with model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Attach tokenizer to model (required for safety processors)
model.tokenizer = tokenizer

inputs = tokenizer("Hello, I want to", return_tensors="pt")
outputs = model.generate(**inputs, safety_config=config, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

### Using Preset Configurations

SafetyConfig provides three preset configurations for different safety/performance trade-offs:

```python
from examples.safe_generation import BasicToxicityChecker
from transformers.generation.safety import SafetyConfig, STRICT_PRESET, MODERATE_PRESET, LENIENT_PRESET

checker = BasicToxicityChecker(threshold=0.7)

# STRICT preset - Maximum safety, more overhead
# - Smaller caches (50 entries, 500 unsafe hash limit)
# - Returns violations and metadata for debugging
config_strict = SafetyConfig.from_checker(checker, **STRICT_PRESET)

# MODERATE preset - Balanced approach (default)
# - Medium caches (100 entries, 1000 unsafe hash limit)
# - No extra metadata (better performance)
config_moderate = SafetyConfig.from_checker(checker, **MODERATE_PRESET)

# LENIENT preset - Performance-optimized
# - Larger caches (200 entries, 2000 unsafe hash limit)
# - No extra metadata
config_lenient = SafetyConfig.from_checker(checker, **LENIENT_PRESET)

# Custom preset - Mix and match
config_custom = SafetyConfig.from_checker(
    checker,
    cache_size=150,
    unsafe_hash_limit=1500,
    return_violations=True,  # Get detailed violation info
    return_metadata=False    # Skip extra metadata
)
```

**Preset Comparison:**

| Preset | cache_size | unsafe_hash_limit | return_violations | return_metadata | Use Case |
|--------|-----------|-------------------|-------------------|-----------------|----------|
| STRICT | 50 | 500 | True | True | High-risk applications, debugging |
| MODERATE | 100 | 1000 | False | False | General use (balanced) |
| LENIENT | 200 | 2000 | False | False | Performance-critical, trusted content |

### Customizing the BasicToxicityChecker

```python
from examples.safe_generation import BasicToxicityChecker

# Use different threshold
strict_checker = BasicToxicityChecker(threshold=0.5)  # More strict

# Use different model
custom_checker = BasicToxicityChecker(
    model_name="unitary/toxic-bert",
    threshold=0.7,
    device="cuda"  # Force specific device
)
```

## Implementing Custom Safety Checkers

You can create your own safety checkers by inheriting from `SafetyChecker`:

```python
from transformers.generation.safety import SafetyChecker, SafetyResult, SafetyViolation

class MyCustomChecker(SafetyChecker):
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        # Your initialization here

    @property
    def supported_categories(self) -> list[str]:
        """Declare what categories this checker detects."""
        return ["violence", "hate_speech", "custom_category"]

    def check_safety(self, text, **kwargs):
        """
        Check text for safety violations.

        Args:
            text: Single string or list of strings to check

        Returns:
            SafetyResult or list of SafetyResult
        """
        if isinstance(text, str):
            return self._check_single(text)
        return [self._check_single(t) for t in text]

    def _check_single(self, text: str) -> SafetyResult:
        # Your safety checking logic here
        is_safe = True  # Your determination
        violations = []  # List of SafetyViolation objects

        if not is_safe:
            violations.append(SafetyViolation(
                category="your_category",
                confidence=0.9,
                severity="high",
                description="Why this is unsafe"
            ))

        return SafetyResult(
            is_safe=is_safe,
            confidence=0.95,
            violations=violations,
            metadata={"checker": "MyCustomChecker"}
        )
```

### LLM-Based Safety Checker Example

```python
import openai
import json
from transformers.generation.safety import SafetyChecker, SafetyResult, SafetyViolation

class LLMSafetyChecker(SafetyChecker):
    """Safety checker using an LLM API for multi-category detection."""

    def __init__(self, api_key: str, categories: list[str] = None):
        self.client = openai.OpenAI(api_key=api_key)
        self.categories = categories or ["violence", "self_harm", "hate_speech", "sexual"]

    @property
    def supported_categories(self) -> list[str]:
        return self.categories

    def check_safety(self, text, **kwargs):
        if isinstance(text, str):
            return self._check_single(text)
        return [self._check_single(t) for t in text]

    def _check_single(self, text: str) -> SafetyResult:
        prompt = f"""Evaluate if this text contains any of: {', '.join(self.categories)}

Text: {text}

Respond with JSON: {{"is_safe": true/false, "category": "...", "confidence": 0.0-1.0}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)

            violations = []
            if not result["is_safe"]:
                violations.append(SafetyViolation(
                    category=result.get("category", "unknown"),
                    confidence=result["confidence"],
                    severity="high" if result["confidence"] > 0.8 else "medium",
                    description=f"Detected {result['category']} content"
                ))

            return SafetyResult(
                is_safe=result["is_safe"],
                confidence=result["confidence"],
                violations=violations,
                metadata={"model": "gpt-4", "categories_checked": self.categories}
            )
        except Exception as e:
            # Fail-safe: assume unsafe on error
            return SafetyResult(
                is_safe=False,
                confidence=0.0,
                violations=[SafetyViolation("error", 0.0, "high", str(e))],
                metadata={"error": str(e)}
            )

# Usage
llm_checker = LLMSafetyChecker(api_key="your-api-key")
config = SafetyConfig.from_checker(llm_checker)
```

## Performance Optimization

For high-latency checkers (like LLM APIs), use SafetyConfig.from_checker() with custom performance settings:

```python
from transformers.generation.safety import SafetyConfig

# For high-latency checkers, optimize with larger caches and sliding windows
config = SafetyConfig.from_checker(
    your_checker,            # Your checker instance
    cache_size=500,          # Large cache for API responses
    unsafe_hash_limit=5000,  # Track more unsafe patterns
    sliding_window_size=512, # Limit tokens sent to API
    incremental_checking=True, # Avoid re-processing same content
    return_violations=False, # Disable for better performance
    return_metadata=False    # Disable for better performance
)
```

## Files in This Directory

- `checkers.py`: Reference implementation of `BasicToxicityChecker`
- `__init__.py`: Exports for easy importing
- `README.md`: This file - usage guide and examples

## Further Reading

- [Safe Generation Design Document](../../docs/0.safe_generation_design.md)
- [Extensibility and Checker Strategy](../../docs/6.extensibility_and_checker_strategy.md)
- [Core Safety Infrastructure](../../docs/1.core_safety_infrastructure.md)
