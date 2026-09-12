# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Telling a continuous batching forward apart from an ordinary one, by what the call carries."""


def is_paged_call(kwargs: dict) -> bool:
    """Whether these forward kwargs carry a continuous batching paged cache.

    Continuous batching passes its cache, and the packed-input metadata that goes with it, through the model
    forward kwargs. That makes paged-ness a property of the call rather than of the model, so the engine does
    not have to switch the model to a `paged|` implementation, and the model stays usable for an ordinary
    forward while the engine runs.
    """
    cache = kwargs.get("cache")
    if cache is None:
        return False
    from ..generation.continuous_batching import PagedAttentionCache

    return isinstance(cache, PagedAttentionCache)
