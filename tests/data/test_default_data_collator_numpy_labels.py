import numpy as np

from transformers import default_data_collator


def test_numpy_integer_labels_stay_integers():
    for label_type in (int, np.int32, np.int64):
        batch = default_data_collator([{"label": label_type(3)}, {"label": label_type(4)}])
        assert batch["labels"].dtype == np.dtype("int64") or str(batch["labels"].dtype) == "torch.int64"

    batch = default_data_collator([{"label": np.float32(1.5)}])
    assert "float" in str(batch["labels"].dtype)

    numpy_batch = default_data_collator([{"label": np.int64(3)}], return_tensors="np")
    assert numpy_batch["labels"].dtype == np.int64
