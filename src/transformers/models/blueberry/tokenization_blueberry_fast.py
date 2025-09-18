# coding=utf-8
# Copyright 2025 Dustin Loring
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

from .tokenization_blueberry import BlueberryTokenizerFast as _BlueberryTokenizerFast


# Re-export the fast tokenizer from the main tokenization module so that
# AutoTokenizer import resolution can find it under the conventional
# `tokenization_blueberry_fast` module path.
BlueberryTokenizerFast = _BlueberryTokenizerFast

__all__ = ["BlueberryTokenizerFast"]

