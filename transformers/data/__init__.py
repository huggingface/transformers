from .processors import InputExample, InputFeatures, DataProcessor
from .processors import glue_output_modes, glue_processors, glue_tasks_num_labels, glue_convert_examples_to_features
from .processors import xnli_output_modes, xnli_processors, xnli_tasks_num_labels

from .metrics import is_sklearn_available
if is_sklearn_available():
    from .metrics import glue_compute_metrics, xnli_compute_metrics
