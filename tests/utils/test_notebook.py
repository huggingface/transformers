import pytest
from transformers.utils.notebook import NotebookProgressBar, NotebookTrainingTracker
from unittest.mock import patch

pytestmark = [pytest.mark.flaky(reruns=3, reruns_delay=2), pytest.mark.is_staging_test]

@pytest.fixture(autouse=True)
def mock_ipython():
    """Mock IPython display to avoid requiring IPython installation"""
    with patch("transformers.utils.notebook.disp") as mock_display:
        yield mock_display

def test_notebook_progress_bar_display():
    pbar = NotebookProgressBar(100)
    with patch('IPython.display.display') as mock_display:
        pbar.display()
        assert mock_display.called

    with patch('IPython.display.display') as mock_display:
        mock_display.side_effect = [OSError(), None]
        pbar.display()
        assert mock_display.called
        assert len(mock_display.call_args_list) == 2

def test_notebook_training_tracker_display():
    tracker = NotebookTrainingTracker(100, ["Step", "Training Loss"])
    with patch('IPython.display.display') as mock_display:
        tracker.display()
        assert mock_display.called

    with patch('IPython.display.display') as mock_display:
        child_bar = tracker.add_child(10)
        child_bar.display()
        assert mock_display.called
