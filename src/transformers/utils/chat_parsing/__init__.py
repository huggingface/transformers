# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Response parsing: extract structured message dicts from LLM output.

Given a declarative `response_format` spec (pure data, shipped in a
model's tokenizer_config.json), convert model-emitted text into the
assistant-message dict used by chat templates.

Public API:
    parse_response(text, response_format)  -> dict
    ResponseEventStream(response_format)   -> stateful streamer (V1: region-level events)
"""

from .executor import ResponseEventStream, parse_response
from .spec import load_spec, validate_spec


__all__ = ["ResponseEventStream", "load_spec", "parse_response", "validate_spec"]
