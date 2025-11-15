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

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from .base import SafetyChecker


# Constants for validation warnings
WARNING_CACHE_SIZE_LIMIT = 10000
WARNING_UNSAFE_HASH_LIMIT = 100000


@dataclass
class SafetyConfig:
    """
    Configuration for safety checking in text generation.

    This configuration class stores settings for safety checking and accepts a user-provided
    safety checker instance. The transformers library provides the infrastructure
    (SafetyChecker abstract base, processors, configuration), while users implement
    concrete checkers for their specific safety requirements.

    Args:
        enabled (`bool`, *optional*, defaults to `False`):
            Whether safety checking is enabled.
        checker (`SafetyChecker`, *optional*, defaults to `None`):
            The safety checker instance to use. Must be provided by the user.
            See examples/safe_generation/ for reference implementations.
        device (`str`, *optional*):
            Device to run models on. If None, automatically selects CUDA if available.
        cache_size (`int`, *optional*, defaults to `100`):
            Maximum number of safety check results to cache. Larger values use more memory
            but can improve performance for repetitive content.
        unsafe_hash_limit (`int`, *optional*, defaults to `1000`):
            Maximum number of unsafe sequence hashes to remember. Prevents memory leaks
            in long-running applications with many unsafe sequences.
        sliding_window_size (`int`, *optional*, defaults to `512`):
            Maximum number of tokens to check for safety instead of the full sequence.
            Helps improve performance for long sequences while maintaining safety effectiveness.
            Set to -1 to disable sliding window (check full sequence).
        incremental_checking (`bool`, *optional*, defaults to `True`):
            Whether to enable incremental safety checking that tracks state between checks
            to avoid redundant processing. Improves performance for long generations.
        return_violations (`bool`, *optional*, defaults to `False`):
            Whether to return detailed violation information in results.
        return_metadata (`bool`, *optional*, defaults to `False`):
            Whether to return additional metadata in results.

    Examples:
    ```python
    # Using a reference implementation from examples directory
    # Note: You need to add examples/ to your Python path first:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("examples")))

    from safe_generation import BasicToxicityChecker
    from transformers.generation.safety import SafetyConfig

    # Create checker instance
    checker = BasicToxicityChecker(threshold=0.7)

    # Option 1: Create config with from_checker() (recommended)
    config = SafetyConfig.from_checker(checker)

    # Option 2: Create config directly
    config = SafetyConfig(enabled=True, checker=checker)

    # Use with generation
    from transformers import pipeline
    pipe = pipeline("text-generation", model="gpt2", safety_config=config)
    ```
    """

    # Checker configuration
    enabled: bool = False
    checker: Optional[SafetyChecker] = None

    # Device configuration
    device: Optional[str] = None

    # Performance configuration
    cache_size: int = 100
    unsafe_hash_limit: int = 1000
    sliding_window_size: int = 512
    incremental_checking: bool = True
    prefix_lengths: list[int] = field(default_factory=lambda: [100, 75, 50])
    min_text_length_for_prefix: int = 50

    # Output configuration
    return_violations: bool = False
    return_metadata: bool = False

    def __post_init__(self):
        """Perform immediate validation after initialization."""
        # Basic type checking for critical parameters
        if not isinstance(self.cache_size, int):
            raise TypeError(f"cache_size must be an integer, got {type(self.cache_size).__name__}")

        if not isinstance(self.unsafe_hash_limit, int):
            raise TypeError(f"unsafe_hash_limit must be an integer, got {type(self.unsafe_hash_limit).__name__}")

        # Range validation
        if self.cache_size < 1:
            raise ValueError("cache_size must be a positive integer")

        if self.unsafe_hash_limit < 1:
            raise ValueError("unsafe_hash_limit must be a positive integer")

        # Validate sliding window size
        if not isinstance(self.sliding_window_size, int):
            raise TypeError(f"sliding_window_size must be an integer, got {type(self.sliding_window_size).__name__}")

        if self.sliding_window_size < -1 or self.sliding_window_size == 0:
            raise ValueError("sliding_window_size must be a positive integer or -1 to disable")

        # Validate incremental checking
        if not isinstance(self.incremental_checking, bool):
            raise TypeError(f"incremental_checking must be a boolean, got {type(self.incremental_checking).__name__}")

        # Validate prefix configuration
        if not isinstance(self.prefix_lengths, list):
            raise TypeError(f"prefix_lengths must be a list, got {type(self.prefix_lengths).__name__}")

        if not all(isinstance(length, int) and length > 0 for length in self.prefix_lengths):
            raise ValueError("All prefix_lengths must be positive integers")

        if not isinstance(self.min_text_length_for_prefix, int) or self.min_text_length_for_prefix < 1:
            raise ValueError("min_text_length_for_prefix must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Note: The checker instance is not serialized. You must recreate it when
        deserializing.

        Returns:
            `Dict[str, Any]`: Dictionary representation of the configuration.
        """
        return {
            "enabled": self.enabled,
            "device": self.device,
            "cache_size": self.cache_size,
            "unsafe_hash_limit": self.unsafe_hash_limit,
            "sliding_window_size": self.sliding_window_size,
            "incremental_checking": self.incremental_checking,
            "prefix_lengths": self.prefix_lengths,
            "min_text_length_for_prefix": self.min_text_length_for_prefix,
            "return_violations": self.return_violations,
            "return_metadata": self.return_metadata,
            # Note: checker is not serialized - must be provided when deserializing
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> SafetyConfig:
        """
        Create SafetyConfig from dictionary.

        Args:
            config_dict (`Dict[str, Any]`): Dictionary containing configuration parameters.

        Returns:
            `SafetyConfig`: Instance created from the dictionary.
        """
        return cls(**config_dict)

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        # Validate enabled is boolean
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")

        # Warn about potentially inefficient configurations (validation done in __post_init__)
        if self.cache_size > WARNING_CACHE_SIZE_LIMIT:
            warnings.warn(
                f"cache_size > {WARNING_CACHE_SIZE_LIMIT} may use excessive memory", UserWarning, stacklevel=2
            )

        if self.unsafe_hash_limit > WARNING_UNSAFE_HASH_LIMIT:
            warnings.warn(
                f"unsafe_hash_limit > {WARNING_UNSAFE_HASH_LIMIT} may use excessive memory", UserWarning, stacklevel=2
            )

        # Validate output configuration
        if not isinstance(self.return_violations, bool):
            raise ValueError("return_violations must be a boolean")

        if not isinstance(self.return_metadata, bool):
            raise ValueError("return_metadata must be a boolean")

    def construct_checker(self) -> SafetyChecker:
        """
        Retrieve the safety checker from the configuration.

        Returns the user-provided checker instance that was specified when creating
        the configuration.

        Returns:
            `SafetyChecker`: The safety checker instance.

        Raises:
            ValueError: If no checker instance is provided.

        Examples:
        ```python
        # See examples/safe_generation/ for reference implementations
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path("examples")))

        from safe_generation import BasicToxicityChecker
        from transformers.generation.safety import SafetyConfig

        # Create checker
        checker = BasicToxicityChecker(threshold=0.7)

        # Create config with checker
        config = SafetyConfig.from_checker(checker)

        # Construct checker (returns the same instance)
        safety_checker = config.construct_checker()
        ```
        """
        if self.checker is None:
            raise ValueError(
                "SafetyConfig requires a checker instance. "
                "You must provide a SafetyChecker when creating the configuration. "
                "See examples/safe_generation/ for reference implementations:\n\n"
                "  from examples.safe_generation import BasicToxicityChecker\n"
                "  checker = BasicToxicityChecker(threshold=0.7)\n"
                "  config = SafetyConfig.from_checker(checker)\n\n"
                "Or implement your own custom checker by inheriting from SafetyChecker."
            )
        return self.checker

    @classmethod
    def from_checker(cls, checker: SafetyChecker, **kwargs) -> SafetyConfig:
        """
        Create a SafetyConfig from a safety checker instance.

        This is the recommended way to create a SafetyConfig.

        Args:
            checker (`SafetyChecker`): The safety checker instance to use.
            **kwargs: Additional configuration parameters to override defaults.

        Returns:
            `SafetyConfig`: A SafetyConfig instance with the provided checker.

        Examples:
        ```python
        # See examples/safe_generation/ for reference implementations
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path("examples")))

        from safe_generation import BasicToxicityChecker
        from transformers.generation.safety import SafetyConfig

        # Create checker
        checker = BasicToxicityChecker(threshold=0.7)

        # Create config from checker
        config = SafetyConfig.from_checker(checker)

        # With additional parameters
        config = SafetyConfig.from_checker(
            checker,
            cache_size=200,
            return_violations=True
        )
        ```
        """
        return cls(enabled=True, checker=checker, **kwargs)


# Preset configuration kwargs for convenience
# These replace the deprecated create_default() method
# Usage: SafetyConfig.from_checker(checker, **STRICT_PRESET)

STRICT_PRESET = {
    "cache_size": 50,
    "unsafe_hash_limit": 500,
    "return_violations": True,
    "return_metadata": True,
}

MODERATE_PRESET = {
    "cache_size": 100,
    "unsafe_hash_limit": 1000,
    "return_violations": False,
    "return_metadata": False,
}

LENIENT_PRESET = {
    "cache_size": 200,
    "unsafe_hash_limit": 2000,
    "return_violations": False,
    "return_metadata": False,
}
