# Temporary compatibility shim: legacy import path expected by tests
# Provides LogitsProcessor and LogitsProcessorList re-exported from logits_process.
from .logits_process import LogitsProcessor, LogitsProcessorList

__all__ = ["LogitsProcessor", "LogitsProcessorList"]
