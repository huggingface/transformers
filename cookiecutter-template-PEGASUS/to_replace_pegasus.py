## Copyright 2020 The HuggingFace Team. All rights reserved.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

## This file is made so that specific statements may be copied inside existing files. This is useful to copy
## import statements in __init__.py, or to complete model lists in the AUTO files.
##
## It is to be used as such:
## Put '# To replace in: "FILE_PATH"' in order to indicate the contents will be copied in the file at path FILE_PATH
## Put '# Below: "STATEMENT"' in order to copy the contents below **the first occurence** of that line in the file at FILE_PATH
## Put '# Replace with:' followed by the lines containing the content to define the content
## End a statement with '# End.'. If starting a new statement without redefining the FILE_PATH, it will continue pasting
## content in that file.
##
## Put '## COMMENT' to comment on the file.


# To replace in: "src/transformers/__init__.py"
# Below: "if is_torch_available():" if generating PyTorch
# Replace with:

    from .models.pegasus import (
        PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST,
        PegasusForConditionalGeneration,
        PegasusForQuestionAnswering,
        PegasusForSequenceClassification,
        PegasusModel,
    )
# End.

# Below: "if is_tf_available():" if generating TensorFlow
# Replace with:

    from .models.pegasus import (
        TFPegasusForConditionalGeneration,
        TFPegasusModel,
        TFPegasusPreTrainedModel,
    )
# End.

# Below: "if is_tokenizers_available():"
# Replace with:
    from .models.pegasus import PegasusTokenizerFast
# End.

# Below: "from .models.albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig"
# Replace with:
from .models.pegasus import PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP, PegasusConfig, PegasusTokenizer
# End.



# To replace in: "src/transformers/models/auto/configuration_auto.py"
# Below: "# Add configs here"
# Replace with:
        ("pegasus", PegasusConfig),
# End.

# Below: "# Add archive maps here"
# Replace with:
        PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP,
# End.

# Below: "from ..albert.configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig",
# Replace with:
from ..pegasus.configuration_pegasus import PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP, PegasusConfig
# End.

# Below: "# Add full (and cased) model names here"
# Replace with:
        ("pegasus", "Pegasus"),
# End.



# To replace in: "src/transformers/models/auto/modeling_auto.py" if generating PyTorch
# Below: "from .configuration_auto import ("
# Replace with:
    PegasusConfig,
# End.

# Below: "# Add modeling imports here"
# Replace with:
from ..pegasus.modeling_pegasus import (
    PegasusForConditionalGeneration,
    PegasusForQuestionAnswering,
    PegasusForSequenceClassification,
    PegasusModel,
)
# End.

# Below: "# Base model mapping"
# Replace with:
        (PegasusConfig, PegasusModel),
# End.

# Below: "# Model with LM heads mapping"
# Replace with:

        (PegasusConfig, PegasusForConditionalGeneration),
# End.

# Below: "# Model for Causal LM mapping"
# Replace with:
# End.

# Below: "# Model for Masked LM mapping"
# Replace with:
# End.

# Below: "# Model for Sequence Classification mapping"
# Replace with:
        (PegasusConfig, PegasusForSequenceClassification),
# End.

# Below: "# Model for Question Answering mapping"
# Replace with:
        (PegasusConfig, PegasusForQuestionAnswering),
# End.

# Below: "# Model for Token Classification mapping"
# Replace with:
# End.

# Below: "# Model for Multiple Choice mapping"
# Replace with:
# End.

# Below: "# Model for Seq2Seq Causal LM mapping"
# Replace with:

        (PegasusConfig, PegasusForConditionalGeneration),
# End.

# To replace in: "src/transformers/models/auto/modeling_tf_auto.py" if generating TensorFlow
# Below: "from .configuration_auto import ("
# Replace with:
    PegasusConfig,
# End.

# Below: "# Add modeling imports here"
# Replace with:
from ..pegasus.modeling_tf_pegasus import (
    TFPegasusForConditionalGeneration,
    TFPegasusModel,
)
# End.

# Below: "# Base model mapping"
# Replace with:
        (PegasusConfig, TFPegasusModel),
# End.

# Below: "# Model with LM heads mapping"
# Replace with:

        (PegasusConfig, TFPegasusForConditionalGeneration),
# End.

# Below: "# Model for Causal LM mapping"
# Replace with:
# End.

# Below: "# Model for Masked LM mapping"
# Replace with:
# End.

# Below: "# Model for Sequence Classification mapping"
# Replace with:
# End.

# Below: "# Model for Question Answering mapping"
# Replace with:
# End.

# Below: "# Model for Token Classification mapping"
# Replace with:
# End.

# Below: "# Model for Multiple Choice mapping"
# Replace with:
# End.

# Below: "# Model for Seq2Seq Causal LM mapping"
# Replace with:

        (PegasusConfig, TFPegasusForConditionalGeneration),
# End.

# To replace in: "utils/check_repo.py" if generating PyTorch

# Below: "models to ignore for model xxx mapping"
# Replace with:
"PegasusEncoder",
    "PegasusDecoder",
# End.

# Below: "models to ignore for not tested"
# Replace with:
"PegasusEncoder",  # Building part of bigger (tested) model.
    "PegasusDecoder",  # Building part of bigger (tested) model.
# End.
