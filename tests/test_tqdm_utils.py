from unittest.mock import patch

from transformers import AutoModel
from transformers.utils.tqdm_utils import set_progress_bar_enabled


def test_set_progress_bar_enabled():
    with patch("tqdm.auto.tqdm") as mock_tqdm:
        set_progress_bar_enabled(True)
        _ = AutoModel.from_pretrained("distilbert-base-uncased", force_download=True)
        mock_tqdm.assert_called()

        mock_tqdm.reset_mock()

        set_progress_bar_enabled(False)
        _ = AutoModel.from_pretrained("distilbert-base-uncased", force_download=True)
        mock_tqdm.assert_not_called()
