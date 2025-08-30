# Temporary compatibility shim: legacy import path expected by tests
# Provides LogitsProcessor and LogitsProcessorList re-exported from logits_process.
from .logits_process import LogitsProcessor, LogitsProcessorList  # noqa: I001

__all__ = ["LogitsProcessor", "LogitsProcessorList"]
