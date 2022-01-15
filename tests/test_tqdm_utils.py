from unittest.mock import patch

from transformers import AutoModel
from transformers.testing_utils import require_torch
from transformers.utils.tqdm_utils import set_progress_bar_enabled


TINY_MODEL = "hf-internal-testing/tiny-random-distilbert"


@require_torch
def test_set_progress_bar_enabled():
    with patch("tqdm.auto.tqdm") as mock_tqdm:
        set_progress_bar_enabled(True)
        _ = AutoModel.from_pretrained(TINY_MODEL, force_download=True)
        mock_tqdm.assert_called()

        mock_tqdm.reset_mock()

        set_progress_bar_enabled(False)
        _ = AutoModel.from_pretrained(TINY_MODEL, force_download=True)
        mock_tqdm.assert_not_called()
