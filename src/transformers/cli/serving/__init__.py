# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from ...utils.import_utils import is_serve_available
from .utils import Modality


# The server machinery depends on the serve extras (fastapi, openai, ...).
# Importing this package must stay safe without them — `transformers.cli.serve`
# imports `.utils` at module level (and therefore runs this __init__) just to
# build the CLI, and only pulls in the server lazily when `serve` actually runs.
if is_serve_available():
    from .model_manager import ModelManager
    from .server import build_server
