import numpy as np
import importlib.util, pathlib

# Load the example module directly from its file path (hyphen-safe)
PATH = pathlib.Path("examples/pytorch/text-classification/run_multilabel_classification.py")
spec = importlib.util.spec_from_file_location("mlc_example", str(PATH))
mlc = importlib.util.module_from_spec(spec)
assert spec and spec.loader, "Could not load spec for example module"
spec.loader.exec_module(mlc)

def test_sigmoid_binarize_shapes():
    x = np.array([0.0, 10.0, -10.0])
    p = mlc.sigmoid(x)
    assert p.shape == (3,)
    assert np.all((p > 0) & (p < 1)), "sigmoid outputs must be in (0,1)"
    y = mlc.binarize_probs(p.reshape(1, -1), 0.5)
    assert y.shape == (1, 3)
    assert set(y.ravel()) <= {0, 1}

def test_metrics_ranges_and_keys():
    y_true = np.array([[1,0,1],[0,1,0],[1,1,0]])
    y_pred = np.array([[1,0,1],[0,1,1],[1,0,0]])
    m = mlc.multilabel_metrics(y_true, y_pred)
    assert set(m) == {"f1_micro","f1_macro","hamming_loss","subset_accuracy"}
    for v in m.values():
        assert 0.0 <= v <= 1.0
