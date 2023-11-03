# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import warnings

from ..trainer import Trainer
from ..utils import logging


logger = logging.get_logger(__name__)


class SageMakerTrainer(Trainer):
    def __init__(self, args=None, **kwargs):
        warnings.warn(
            "`SageMakerTrainer` is deprecated and will be removed in v5 of Transformers. You can use `Trainer` "
            "instead.",
            FutureWarning,
        )
        super().__init__(args=args, **kwargs)
