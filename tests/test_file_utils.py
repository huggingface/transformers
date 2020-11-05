import unittest

import requests
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME, filename_to_url, get_from_cache, hf_bucket_url
from transformers.testing_utils import DUMMY_UNKWOWN_IDENTIFIER


MODEL_ID = DUMMY_UNKWOWN_IDENTIFIER
REVISION_ID = "main"
PINNED_SHA1 = "d9e9f15bc825e4b2c9249e9578f884bbcb5e3684"
PINNED_SHA256 = "4b243c475af8d0a7754e87d7d096c92e5199ec2fe168a2ee7998e3b8e9bcb1d3"


class GetFromCacheTests(unittest.TestCase):
    def test_bogus_url(self):
        # This lets us simulate no connection
        # as the error raised is the same
        # `ConnectionError`
        url = "https://bogus"
        with self.assertRaisesRegex(ValueError, "Connection error"):
            _ = get_from_cache(url)

    def test_not_found(self):
        url = hf_bucket_url(MODEL_ID, filename="missing.bin")
        with self.assertRaisesRegex(requests.exceptions.HTTPError, "404 Client Error"):
            _ = get_from_cache(url)

    def test_standard_object(self):
        url = hf_bucket_url(MODEL_ID, filename=CONFIG_NAME, revision=REVISION_ID)
        filepath = get_from_cache(url, force_download=True)
        metadata = filename_to_url(filepath)
        self.assertEqual(metadata, (url, f'"{PINNED_SHA1}"'))

    def test_lfs_object(self):
        url = hf_bucket_url(MODEL_ID, filename=WEIGHTS_NAME, revision=REVISION_ID)
        filepath = get_from_cache(url, force_download=True)
        metadata = filename_to_url(filepath)
        self.assertEqual(metadata, (url, f'"{PINNED_SHA256}"'))
