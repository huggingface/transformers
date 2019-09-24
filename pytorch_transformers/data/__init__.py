from .processors import (InputExample, InputFeatures, DataProcessor,
                         glue_output_modes, glue_convert_examples_to_features, glue_processors)
from .metrics import is_sklearn_available

if is_sklearn_available():
    from .metrics import glue_compute_metrics
