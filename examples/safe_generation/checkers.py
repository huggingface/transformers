# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Union

import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.generation.safety import SafetyChecker, SafetyResult, SafetyViolation
from transformers.utils import is_torch_available, logging


if not is_torch_available():
    raise ImportError("PyTorch is required to use safety checkers. Please install PyTorch: pip install torch")


logger = logging.get_logger(__name__)


class BasicToxicityChecker(SafetyChecker):
    """
    Toxicity checker using the s-nlp/roberta_toxicity_classifier model.

    This checker uses a pre-trained RoBERTa model to detect toxic content in text. It supports both
    single text and batch processing, with configurable thresholds and automatic device selection.

    This is a reference implementation provided in the examples directory to demonstrate how to
    implement custom safety checkers. The core transformers library provides only the infrastructure
    (SafetyChecker abstract base class, processors, configuration).

    Args:
        model_name (`str`, *optional*, defaults to `"s-nlp/roberta_toxicity_classifier"`):
            The name of the pre-trained model to use for toxicity detection.
        threshold (`float`, *optional*, defaults to `0.7`):
            The toxicity score threshold above which content is considered unsafe.
        device (`str`, *optional*):
            The device to run the model on. If None, automatically selects CUDA if available, else CPU.

    Examples:
    ```python
    >>> from examples.safe_generation import BasicToxicityChecker
    >>> from transformers.generation.safety import SafetyConfig
    >>> from transformers import pipeline

    >>> # Create checker
    >>> checker = BasicToxicityChecker(threshold=0.7)

    >>> # Use with SafetyConfig
    >>> config = SafetyConfig.from_checker(checker)
    >>> pipe = pipeline("text-generation", model="gpt2", safety_config=config)
    ```
    """

    def __init__(
        self,
        model_name: str = "s-nlp/roberta_toxicity_classifier",
        threshold: float = 0.7,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer with error handling
        try:
            logger.info(f"Loading toxicity model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded toxicity model on {self.device}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load toxicity model '{model_name}'. "
                f"Please ensure the model exists and you have internet connectivity. "
                f"Original error: {e}"
            )

    @property
    def supported_categories(self) -> list[str]:
        """Return list of safety categories this checker supports."""
        return ["toxicity"]

    def check_safety(self, text: Union[str, list[str]], **kwargs) -> Union[SafetyResult, list[SafetyResult]]:
        """
        Check text(s) for toxicity violations.

        Args:
            text (`Union[str, List[str]]`):
                Single text string or list of texts to check for toxicity.
            **kwargs:
                Additional parameters (currently unused).

        Returns:
            `Union[SafetyResult, List[SafetyResult]]`:
                SafetyResult for single text input, List[SafetyResult] for multiple texts.
        """
        if isinstance(text, str):
            return self._check_single_text(text, **kwargs)
        elif isinstance(text, list):
            return [self._check_single_text(t, **kwargs) for t in text]
        else:
            raise TypeError(f"Expected string or list of strings, got {type(text)}")

    def _check_single_text(self, text: str, **kwargs) -> SafetyResult:
        """
        Check single text for toxicity.

        Args:
            text (`str`): Text to check for toxicity.
            **kwargs: Additional parameters (currently unused).

        Returns:
            `SafetyResult`: Result of the safety check.
        """
        # Input validation
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, got {type(text)}")

        # Handle empty text
        if not text.strip():
            return SafetyResult(
                is_safe=True,
                confidence=1.0,
                violations=[],
                metadata={"reason": "empty_text", "model_name": self.model_name},
            )

        # Handle very long text
        original_length = len(text)
        max_length = 10000  # Reasonable limit
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated from {original_length} to {max_length} characters")

        # Tokenize and run inference
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)

            # Extract toxicity probability (assuming binary classification: [non-toxic, toxic])
            toxicity_score = probabilities[0][1].item()  # Toxic class probability

        except Exception as e:
            logger.error(f"Error during toxicity inference: {e}")
            raise RuntimeError(f"Toxicity detection failed: {e}")

        # Determine safety
        is_safe = toxicity_score < self.threshold
        violations = []

        if not is_safe:
            violations.append(
                SafetyViolation(
                    category="toxicity",
                    confidence=toxicity_score,
                    severity=self._get_severity(toxicity_score),
                    description=f"Detected toxic content with {toxicity_score:.2%} confidence",
                )
            )

        # Prepare metadata
        metadata = {
            "model_name": self.model_name,
            "toxicity_score": toxicity_score,
            "threshold": self.threshold,
            "device": self.device,
        }

        if original_length > max_length:
            metadata["truncated"] = True
            metadata["original_length"] = original_length
            metadata["processed_length"] = max_length

        return SafetyResult(
            is_safe=is_safe,
            confidence=max(toxicity_score, 1.0 - toxicity_score),
            violations=violations,
            metadata=metadata,
        )

    def _get_severity(self, score: float) -> str:
        """
        Determine severity based on toxicity score.

        Args:
            score (`float`): Toxicity score from 0.0 to 1.0.

        Returns:
            `str`: Severity level ("low", "medium", "high", "critical").
        """
        if score >= 0.95:
            return "critical"
        elif score >= 0.85:
            return "high"
        elif score >= 0.75:
            return "medium"
        else:
            return "low"

    def get_config(self) -> dict[str, Any]:
        """
        Return checker configuration for serialization.

        Returns:
            `Dict[str, Any]`: Dictionary containing the checker's configuration.
        """
        return {
            "checker_type": "BasicToxicityChecker",
            "model_name": self.model_name,
            "threshold": self.threshold,
            "device": self.device,
        }
