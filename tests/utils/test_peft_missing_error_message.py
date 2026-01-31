import pytest

from transformers import AutoModel, BertConfig
from transformers.testing_utils import require_torch
from transformers.utils import ADAPTER_CONFIG_NAME, is_peft_available


@require_torch
def test_adapter_repo_without_peft_has_helpful_error(tmp_path):
    if is_peft_available():
        pytest.skip("peft is installed in this environment")

    BertConfig().save_pretrained(tmp_path)
    (tmp_path / ADAPTER_CONFIG_NAME).write_text("{}", encoding="utf-8")

    with pytest.raises(OSError) as excinfo:
        AutoModel.from_pretrained(tmp_path)

    msg = str(excinfo.value).lower()
    assert "peft" in msg
    assert "pip install peft" in msg
    assert "adapter" in msg

