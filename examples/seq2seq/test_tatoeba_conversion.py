import os
import tempfile
import unittest

from transformers.convert_marian_tatoeba_to_pytorch import DEFAULT_REPO, TatoebaConverter
from transformers.file_utils import cached_property
from transformers.testing_utils import slow


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
