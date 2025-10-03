# Copyright 2025 ModelCloud.ai team and The HuggingFace Inc. team.
##
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

"""Thread-safe utilities and module wrappers used across Transformers."""

from __future__ import annotations

import threading
from functools import wraps
from types import ModuleType

import regex as _regex


__all__ = ["ThreadSafe", "SafeRegex", "regex"]


class ThreadSafe(ModuleType):
    """Generic proxy that exposes a module through a shared (non-reentrant) lock."""

    def __init__(self, module: ModuleType):
        super().__init__(module.__name__)
        self._module = module
        self._lock = threading.Lock()
        self._callable_cache: dict[str, object] = {}
        # Keep core module metadata available so tools relying on attributes
        # like __doc__ or __spec__ see the original values.
        self.__dict__.update(
            {
                "__doc__": module.__doc__,
                "__package__": module.__package__,
                "__file__": getattr(module, "__file__", None),
                "__spec__": getattr(module, "__spec__", None),
            }
        )

    def __getattr__(self, name: str):
        attr = getattr(self._module, name)
        if callable(attr):
            cached = self._callable_cache.get(name)
            if cached is not None and getattr(cached, "__wrapped__", None) is attr:
                return cached

            @wraps(attr)
            def locked(*args, **kwargs):
                with self._lock:
                    return attr(*args, **kwargs)

            locked.__wrapped__ = attr
            self._callable_cache[name] = locked
            return locked
        return attr

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._module)))


class _ThreadSafeProxy:
    """Lightweight proxy that serializes access to an object with a shared lock."""

    def __init__(self, value, lock):
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_lock", lock)
        object.__setattr__(self, "_callable_cache", {})
        object.__setattr__(self, "__wrapped__", value)

    def __getattr__(self, name: str):
        attr = getattr(self._value, name)
        if callable(attr):
            cached = self._callable_cache.get(name)
            if cached is not None and getattr(cached, "__wrapped__", None) is attr:
                return cached

            @wraps(attr)
            def locked(*args, **kwargs):
                with self._lock:
                    return attr(*args, **kwargs)

            locked.__wrapped__ = attr
            self._callable_cache[name] = locked
            return locked
        return attr

    def __setattr__(self, name, value):
        setattr(self._value, name, value)

    def __dir__(self):
        return dir(self._value)

    def __repr__(self):
        return repr(self._value)


class SafeRegex(ThreadSafe):
    """Proxy module that exposes ``regex`` through a shared lock."""

    # We must bind the shared regex lock to any objects returned here since
    # compiled patterns expose methods (e.g. pattern.match) that must also be
    # serialized.

    def compile(self, *args, **kwargs):
        pattern = self._module.compile(*args, **kwargs)
        return _ThreadSafeProxy(pattern, self._lock)

    def Regex(self, *args, **kwargs):
        pattern = self._module.Regex(*args, **kwargs)
        return _ThreadSafeProxy(pattern, self._lock)


regex = SafeRegex(_regex)
