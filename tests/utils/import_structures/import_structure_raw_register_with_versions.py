# Copyright 2024 The HuggingFace Team. All rights reserved.
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

# fmt: off

from transformers.utils.import_utils import requires


@requires(backends=("torch>=2.5",))
class D0:
    def __init__(self):
        pass


@requires(backends=("torch>=2.5",))
def d0():
    pass


@requires(backends=("torch>2.5",))
class D1:
    def __init__(self):
        pass


@requires(backends=("torch>2.5",))
def d1():
    pass


@requires(backends=("torch<=2.5",))
class D2:
    def __init__(self):
        pass


@requires(backends=("torch<=2.5",))
def d2():
    pass

@requires(backends=("torch<2.5",))
class D3:
    def __init__(self):
        pass


@requires(backends=("torch<2.5",))
def d3():
    pass


@requires(backends=("torch==2.5",))
class D4:
    def __init__(self):
        pass


@requires(backends=("torch==2.5",))
def d4():
    pass


@requires(backends=("torch!=2.5",))
class D5:
    def __init__(self):
        pass


@requires(backends=("torch!=2.5",))
def d5():
    pass

@requires(backends=("torch>=2.5", "accelerate<0.20"))
class D6:
    def __init__(self):
        pass


@requires(backends=("torch>=2.5", "accelerate<0.20"))
def d6():
    pass
