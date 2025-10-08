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

"""Thread-safe utilities and module wrappers usable across Transformers."""

from __future__ import annotations

import threading
from functools import wraps
from types import ModuleType

import regex as _regex


__all__ = ["ThreadSafe", "SafeRegex", "regex"]


class ThreadSafe(ModuleType):
    """Generic proxy that exposes a module through a shared lock."""

    def __init__(self, module: ModuleType):
        super().__init__(module.__name__)
        # `_hf_safe_` prefix is used to avoid colliding with the wrapped object namespace.
        self._hf_safe_module = module
        # Callable execution lock (re-entrant so wrapped code can re-enter safely)
        self._hf_safe_lock = threading.RLock()
        # Cache dict lock
        self._hf_safe_callable_cache_lock = threading.Lock()
        self._hf_safe_callable_cache: dict[str, object] = {}
        # Retain module metadata so introspection tools relying on attributes
        # like __doc__, __spec__, etc, can see the original values.
        metadata = {"__doc__": module.__doc__}
        for attr in ("__package__", "__file__", "__spec__"):
            if hasattr(module, attr):
                metadata[attr] = getattr(module, attr)
        self.__dict__.update(metadata)

    def __getattr__(self, name: str):
        attr = getattr(self._hf_safe_module, name)
        if callable(attr):
            with self._hf_safe_callable_cache_lock:
                cached = self._hf_safe_callable_cache.get(name)
                if cached is not None and getattr(cached, "__wrapped__", None) is attr:
                    return cached

                @wraps(attr)
                def _hf_safe_locked(*args, **kwargs):
                    with self._hf_safe_lock:
                        return attr(*args, **kwargs)

                _hf_safe_locked.__wrapped__ = attr
                self._hf_safe_callable_cache[name] = _hf_safe_locked
                return _hf_safe_locked
        return attr

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._hf_safe_module)))


class _ThreadSafeProxy:
    """Lightweight proxy that serializes access to an object with a shared lock."""

    def __init__(self, value, lock):
        # `_hf_safe_` prefix is used to avoid colliding with the wrapped object namespace.
        object.__setattr__(self, "_hf_safe_value", value)
        object.__setattr__(self, "_hf_safe_lock", lock)
        object.__setattr__(self, "_hf_safe_cache_lock", threading.Lock())
        object.__setattr__(self, "_hf_safe_callable_cache", {})
        object.__setattr__(self, "__wrapped__", value)

    def __getattr__(self, name: str):
        attr = getattr(self._hf_safe_value, name)
        if callable(attr):
            with self._hf_safe_cache_lock:
                cached = self._hf_safe_callable_cache.get(name)
                if cached is not None and getattr(cached, "__wrapped__", None) is attr:
                    return cached

                @wraps(attr)
                def _hf_safe_locked(*args, **kwargs):
                    with self._hf_safe_lock:
                        return attr(*args, **kwargs)

                _hf_safe_locked.__wrapped__ = attr
                self._hf_safe_callable_cache[name] = _hf_safe_locked
                return _hf_safe_locked
        return attr

    def __setattr__(self, name, value):
        with self._hf_safe_lock:
            setattr(self._hf_safe_value, name, value)

    def __delattr__(self, name):
        with self._hf_safe_lock:
            delattr(self._hf_safe_value, name)

    def __dir__(self):
        with self._hf_safe_lock:
            return dir(self._hf_safe_value)

    def __repr__(self):
        with self._hf_safe_lock:
            return repr(self._hf_safe_value)

    def __call__(self, *args, **kwargs):
        with self._hf_safe_lock:
            return self._hf_safe_value(*args, **kwargs)


class SafeRegex(ThreadSafe):
    """Proxy module that exposes ``regex`` through a shared lock."""

    # We must proxy the shared regex lock to any objects returned here since
    # compiled patterns expose methods that may call regex itself. Also,
    # non-cached compiled pattern is also unsafe for threaded execution as unit
    # tests have shown (segfault): test_safe_crash.py

    def compile(self, *args, **kwargs):
        pattern = self._hf_safe_module.compile(*args, **kwargs)
        return _ThreadSafeProxy(pattern, self._hf_safe_lock)

    def Regex(self, *args, **kwargs):
        pattern = self._hf_safe_module.Regex(*args, **kwargs)
        return _ThreadSafeProxy(pattern, self._hf_safe_lock)


regex = SafeRegex(_regex)
