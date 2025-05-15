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
import json
import os
import shutil
import unittest

from huggingface_hub import hf_hub_download

import transformers
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    TRANSFORMERS_CACHE,
)
from transformers.utils.hub import get_checkpoint_shard_files


RANDOM_BERT_SHARDED = "hf-internal-testing/tiny-random-bert-sharded"
CACHE_DIR = os.path.join(TRANSFORMERS_CACHE, "models--hf-internal-testing--tiny-random-bert-sharded")
FULL_COMMIT_HASH = "04a52fc6ff50bf21639d65be441bd2bd8410ef5d"

CHECK_POINT_EXISTS_FUNC = transformers.utils.hub.checkpoint_exists


class GetFromCacheTestsParallel(unittest.TestCase):
    def setUp(self) -> None:
        if os.path.exists(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)

        # NOTE mock checkpoint_exists so it's a function that returns False
        self._original_checkpoint_exists = CHECK_POINT_EXISTS_FUNC
        # Mock to always make it return False
        transformers.utils.hub.checkpoint_exists = lambda *args, **kwargs: False

        os.environ["HF_ENABLE_PARALLEL_DOWNLOADING"] = "true"
        os.environ["HF_PARALLEL_DOWNLOADING_WORKERS"] = "8"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    def tearDown(self) -> None:
        # Restore the original function after the test
        transformers.utils.hub.checkpoint_exists = self._original_checkpoint_exists

        del os.environ["HF_ENABLE_PARALLEL_DOWNLOADING"]
        del os.environ["HF_PARALLEL_DOWNLOADING_WORKERS"]
        del os.environ["HF_HUB_ENABLE_HF_TRANSFER"]

    def test_get_checkpoint_shard_files(self):
        hf_hub_download(
            RANDOM_BERT_SHARDED,
            filename=SAFE_WEIGHTS_INDEX_NAME,
        )

        index_filename = os.path.join(CACHE_DIR, "snapshots", FULL_COMMIT_HASH, SAFE_WEIGHTS_INDEX_NAME)

        cached_filenames, sharded_metadata = get_checkpoint_shard_files(
            RANDOM_BERT_SHARDED, index_filename, revision=FULL_COMMIT_HASH
        )

        # Should have downloaded the file in here
        self.assertTrue(os.path.isdir(CACHE_DIR))

        # make sure the files we were supposed to download were downloaded
        with open(index_filename, "r") as f:
            index = json.loads(f.read())

        weight_map_file_names = sorted(set(index["weight_map"].values()))

        # make sure we have the same number of caches files as the number of files in the weight map
        self.assertTrue(len(weight_map_file_names), len(cached_filenames))

        for index, cached_filename in enumerate(cached_filenames):
            # now make sure each file exists
            exists = os.path.exists(cached_filename)
            self.assertTrue(exists)

            # now make sure each file was in the set of files we told the function to download
            filename = cached_filename.split("/").pop()

            # make sure they are both sorted the same way
            name_in_set = weight_map_file_names[index]
            self.assertTrue(name_in_set == filename)

        # for extra safety we now perform an integration test for the cached data
        model = transformers.AutoModel.from_pretrained(
            "hf-internal-testing/tiny-random-bert-sharded",
        )
        self.assertIsNotNone(model)

    def test_get_checkpoint_shard_files_integration(self):
        model = transformers.AutoModel.from_pretrained(
            "hf-internal-testing/tiny-random-bert-sharded",
        )
        self.assertIsNotNone(model)
