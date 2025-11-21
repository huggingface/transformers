# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import os
import tempfile
import unittest

from transformers.models.marian.convert_marian_tatoeba_to_pytorch import DEFAULT_REPO, TatoebaConverter
from transformers.testing_utils import slow
from transformers.utils import cached_property


@unittest.skipUnless(os.path.exists(DEFAULT_REPO), "Tatoeba directory does not exist.")
class TatoebaConversionTester(unittest.TestCase):
    @cached_property
    def resolver(self):
        tmp_dir = tempfile.mkdtemp()
        return TatoebaConverter(save_dir=tmp_dir)

    @slow
    def test_resolver(self):
        self.resolver.convert_models(["heb-eng"])

    @slow
    def test_model_card(self):
        content, mmeta = self.resolver.write_model_card("opus-mt-he-en", dry_run=True)
        assert mmeta["long_pair"] == "heb-eng"
