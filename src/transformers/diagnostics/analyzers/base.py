"""
Base analyzer interface for diagnostics collection.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseAnalyzer(ABC):
    """
    Abstract base class for diagnostics analyzers.

    Analyzers are responsible for collecting specific types of metrics
    (memory, timing, activations) during model forward passes.
    """

    def __init__(self):
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Whether this analyzer is currently enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @abstractmethod
    def before_forward(self, module: nn.Module, layer_name: str, inputs: tuple) -> dict[str, Any]:
        """
        Called before a layer's forward pass.

        Args:
            module: The PyTorch module being executed
            layer_name: Name of the layer in the model
            inputs: Input tensors to the layer

        Returns:
            State dict to pass to after_forward
        """
        pass

    @abstractmethod
    def after_forward(
        self, module: nn.Module, layer_name: str, inputs: tuple, outputs: Any, state: dict[str, Any]
    ) -> Any:
        """
        Called after a layer's forward pass.

        Args:
            module: The PyTorch module that was executed
            layer_name: Name of the layer in the model
            inputs: Input tensors to the layer
            outputs: Output tensors from the layer
            state: State dict from before_forward

        Returns:
            Collected metrics/statistics
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset any accumulated state."""
        pass

    def get_device(self, module: nn.Module) -> torch.device | None:
        """Get the device of a module's parameters."""
        try:
            param = next(module.parameters())
            return param.device
        except StopIteration:
            # Module has no parameters, try buffers
            try:
                buf = next(module.buffers())
                return buf.device
            except StopIteration:
                return None

    def is_cuda(self, module: nn.Module) -> bool:
        """Check if module is on CUDA."""
        device = self.get_device(module)
        return device is not None and device.type == "cuda"
