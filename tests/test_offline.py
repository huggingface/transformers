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

import socket
import unittest
from unittest.mock import patch

from transformers import BertConfig, BertModel, BertTokenizer
from transformers.testing_utils import mockenv_context


class OfflineTests(unittest.TestCase):
    def test_offline_mode(self):
        def experiment():
            mname = "lysandre/tiny-bert-random"
            _ = BertConfig.from_pretrained(mname)
            _ = BertModel.from_pretrained(mname)
            _ = BertTokenizer.from_pretrained(mname)

        # should succeed
        experiment()

        def offline_socket(*args, **kwargs):
            raise socket.error("Offline mode is enabled.")

        with patch("socket.socket", offline_socket):
            # should fail
            self.assertRaises(ValueError, experiment)

            with mockenv_context(TRANSFORMERS_OFFLINE="1"):
                # should succeed
                experiment()
