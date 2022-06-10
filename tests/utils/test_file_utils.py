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

import contextlib
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import transformers

# Try to import everything from transformers to ensure every object can be loaded.
from transformers import *  # noqa F406
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER
from transformers.utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    filename_to_url,
    find_labels,
    get_file_from_repo,
    get_from_cache,
    has_file,
    hf_bucket_url,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)


MODEL_ID = DUMMY_UNKNOWN_IDENTIFIER
# An actual model hosted on huggingface.co

REVISION_ID_DEFAULT = "main"
# Default branch name
REVISION_ID_ONE_SPECIFIC_COMMIT = "f2c752cfc5c0ab6f4bdec59acea69eefbee381c2"
# One particular commit (not the top of `main`)
REVISION_ID_INVALID = "aaaaaaa"
# This commit does not exist, so we should 404.

PINNED_SHA1 = "d9e9f15bc825e4b2c9249e9578f884bbcb5e3684"
# Sha-1 of config.json on the top of `main`, for checking purposes
PINNED_SHA256 = "4b243c475af8d0a7754e87d7d096c92e5199ec2fe168a2ee7998e3b8e9bcb1d3"
# Sha-256 of pytorch_model.bin on the top of `main`, for checking purposes


# Dummy contexts to test `ContextManagers`
@contextlib.contextmanager
def context_en():
    print("Welcome!")
    yield
    print("Bye!")


@contextlib.contextmanager
def context_fr():
    print("Bonjour!")
    yield
    print("Au revoir!")


class TestImportMechanisms(unittest.TestCase):
    def test_module_spec_available(self):
        # If the spec is missing, importlib would not be able to import the module dynamically.
        assert transformers.__spec__ is not None
        assert importlib.util.find_spec("transformers") is not None


class GetFromCacheTests(unittest.TestCase):
    def test_bogus_url(self):
        # This lets us simulate no connection
        # as the error raised is the same
        # `ConnectionError`
        url = "https://bogus"
        with self.assertRaisesRegex(ValueError, "Connection error"):
            _ = get_from_cache(url)

    def test_file_not_found(self):
        # Valid revision (None) but missing file.
        url = hf_bucket_url(MODEL_ID, filename="missing.bin")
        with self.assertRaisesRegex(EntryNotFoundError, "404 Client Error"):
            _ = get_from_cache(url)

    def test_model_not_found_not_authenticated(self):
        # Invalid model id.
        url = hf_bucket_url("bert-base", filename="pytorch_model.bin")
        with self.assertRaisesRegex(RepositoryNotFoundError, "401 Client Error"):
            _ = get_from_cache(url)

    @unittest.skip("No authentication when testing against prod")
    def test_model_not_found_authenticated(self):
        # Invalid model id.
        url = hf_bucket_url("bert-base", filename="pytorch_model.bin")
        with self.assertRaisesRegex(RepositoryNotFoundError, "404 Client Error"):
            _ = get_from_cache(url, use_auth_token="hf_sometoken")
            # ^ TODO - if we decide to unskip this: use a real / functional token

    def test_revision_not_found(self):
        # Valid file but missing revision
        url = hf_bucket_url(MODEL_ID, filename=CONFIG_NAME, revision=REVISION_ID_INVALID)
        with self.assertRaisesRegex(RevisionNotFoundError, "404 Client Error"):
            _ = get_from_cache(url)

    def test_standard_object(self):
        url = hf_bucket_url(MODEL_ID, filename=CONFIG_NAME, revision=REVISION_ID_DEFAULT)
        filepath = get_from_cache(url, force_download=True)
        metadata = filename_to_url(filepath)
        self.assertEqual(metadata, (url, f'"{PINNED_SHA1}"'))

    def test_standard_object_rev(self):
        # Same object, but different revision
        url = hf_bucket_url(MODEL_ID, filename=CONFIG_NAME, revision=REVISION_ID_ONE_SPECIFIC_COMMIT)
        filepath = get_from_cache(url, force_download=True)
        metadata = filename_to_url(filepath)
        self.assertNotEqual(metadata[1], f'"{PINNED_SHA1}"')
        # Caution: check that the etag is *not* equal to the one from `test_standard_object`

    def test_lfs_object(self):
        url = hf_bucket_url(MODEL_ID, filename=WEIGHTS_NAME, revision=REVISION_ID_DEFAULT)
        filepath = get_from_cache(url, force_download=True)
        metadata = filename_to_url(filepath)
        self.assertEqual(metadata, (url, f'"{PINNED_SHA256}"'))

    def test_has_file(self):
        self.assertTrue(has_file("hf-internal-testing/tiny-bert-pt-only", WEIGHTS_NAME))
        self.assertFalse(has_file("hf-internal-testing/tiny-bert-pt-only", TF2_WEIGHTS_NAME))
        self.assertFalse(has_file("hf-internal-testing/tiny-bert-pt-only", FLAX_WEIGHTS_NAME))

    def test_get_file_from_repo_distant(self):
        # `get_file_from_repo` returns None if the file does not exist
        self.assertIsNone(get_file_from_repo("bert-base-cased", "ahah.txt"))

        # The function raises if the repository does not exist.
        with self.assertRaisesRegex(EnvironmentError, "is not a valid model identifier"):
            get_file_from_repo("bert-base-case", "config.json")

        # The function raises if the revision does not exist.
        with self.assertRaisesRegex(EnvironmentError, "is not a valid git identifier"):
            get_file_from_repo("bert-base-cased", "config.json", revision="ahaha")

        resolved_file = get_file_from_repo("bert-base-cased", "config.json")
        # The name is the cached name which is not very easy to test, so instead we load the content.
        config = json.loads(open(resolved_file, "r").read())
        self.assertEqual(config["hidden_size"], 768)

    def test_get_file_from_repo_local(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = Path(tmp_dir) / "a.txt"
            filename.touch()
            self.assertEqual(get_file_from_repo(tmp_dir, "a.txt"), str(filename))

            self.assertIsNone(get_file_from_repo(tmp_dir, "b.txt"))


class GenericUtilTests(unittest.TestCase):
    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_context_managers_no_context(self, mock_stdout):
        with ContextManagers([]):
            print("Transformers are awesome!")
        # The print statement adds a new line at the end of the output
        self.assertEqual(mock_stdout.getvalue(), "Transformers are awesome!\n")

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_context_managers_one_context(self, mock_stdout):
        with ContextManagers([context_en()]):
            print("Transformers are awesome!")
        # The output should be wrapped with an English welcome and goodbye
        self.assertEqual(mock_stdout.getvalue(), "Welcome!\nTransformers are awesome!\nBye!\n")

    @unittest.mock.patch("sys.stdout", new_callable=io.StringIO)
    def test_context_managers_two_context(self, mock_stdout):
        with ContextManagers([context_fr(), context_en()]):
            print("Transformers are awesome!")
        # The output should be wrapped with an English and French welcome and goodbye
        self.assertEqual(mock_stdout.getvalue(), "Bonjour!\nWelcome!\nTransformers are awesome!\nBye!\nAu revoir!\n")

    def test_find_labels(self):
        if is_torch_available():
            from transformers import BertForPreTraining, BertForQuestionAnswering, BertForSequenceClassification

            self.assertEqual(find_labels(BertForSequenceClassification), ["labels"])
            self.assertEqual(find_labels(BertForPreTraining), ["labels", "next_sentence_label"])
            self.assertEqual(find_labels(BertForQuestionAnswering), ["start_positions", "end_positions"])

        if is_tf_available():
            from transformers import TFBertForPreTraining, TFBertForQuestionAnswering, TFBertForSequenceClassification

            self.assertEqual(find_labels(TFBertForSequenceClassification), ["labels"])
            self.assertEqual(find_labels(TFBertForPreTraining), ["labels", "next_sentence_label"])
            self.assertEqual(find_labels(TFBertForQuestionAnswering), ["start_positions", "end_positions"])

        if is_flax_available():
            # Flax models don't have labels
            from transformers import (
                FlaxBertForPreTraining,
                FlaxBertForQuestionAnswering,
                FlaxBertForSequenceClassification,
            )

            self.assertEqual(find_labels(FlaxBertForSequenceClassification), [])
            self.assertEqual(find_labels(FlaxBertForPreTraining), [])
            self.assertEqual(find_labels(FlaxBertForQuestionAnswering), [])
