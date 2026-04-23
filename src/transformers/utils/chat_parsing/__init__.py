# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Response parsing: convert model-emitted text into the assistant-message
dict used by chat templates, driven by a declarative `response_format` spec."""

from .executor import ResponseEventStream, parse_response


__all__ = ["ResponseEventStream", "parse_response"]
