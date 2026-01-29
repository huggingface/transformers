import numpy as np
from scipy.special import expit


def test_multilabel_prediction_thresholding():
    # Simulated logits from a multi-label model
    logits = np.array([
        [-1.2, 0.3, 2.1],
        [-0.7, -0.2, 1.5],
    ])

    predictions = np.array(
        [np.where(expit(p) > 0.5, 1, 0) for p in logits]
    )

    # Ensure shape is preserved
    assert predictions.shape == logits.shape

    # Ensure at least one positive label is predicted
    assert predictions.sum() > 0

