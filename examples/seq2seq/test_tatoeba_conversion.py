import os
import tempfile
import unittest

from transformers.file_utils import cached_property
from transformers.models.marian.convert_marian_tatoeba_to_pytorch import DEFAULT_REPO, TatoebaConverter
from transformers.testing_utils import require_torch_non_multi_gpu_but_fix_me, slow


@unittest.skipUnless(os.path.exists(DEFAULT_REPO), "Tatoeba directory does not exist.")
class TatoebaConversionTester(unittest.TestCase):
    @cached_property
    def resolver(self):
        tmp_dir = tempfile.mkdtemp()
        return TatoebaConverter(save_dir=tmp_dir)

    @slow
    @require_torch_non_multi_gpu_but_fix_me
    def test_resolver(self):
        self.resolver.convert_models(["heb-eng"])

    @slow
    @require_torch_non_multi_gpu_but_fix_me
    def test_model_card(self):
        content, mmeta = self.resolver.write_model_card("opus-mt-he-en", dry_run=True)
        assert mmeta["long_pair"] == "heb-eng"
